import numpy as np
import random
import pickle
import os
from profiling.profiling_class import ProfilingData
from simulator.simulator import CloudEdgeSimulator


class TabularDoubleQLearningAgent:
    """
    Tabular Double Q-Learning for PURE LATENCY MINIMIZATION
    
    GOAL: Minimize total inference latency (execution + communication time)
    
    STATE: [bandwidth, cloud_contention, current_layer, previous_assignment]
        - bandwidth: Available network bandwidth (MBps)
        - cloud_contention: Cloud server waiting time (ms)
        - current_layer: Which DNN layer we're processing (0 to L-1)
        - previous_assignment: Where previous layer's nodes ran (tuple of 0/1 values)
    
    ACTION: Binary assignment for each node in current layer
        - 0: Execute on Edge device
        - 1: Execute on Cloud server
    
    ALGORITHM: Double Q-Learning (off-policy, two Q-tables to reduce overestimation)
    """
    
    def __init__(
        self,
        profiling_data: ProfilingData,
        is_test=False,
        alpha=0.1,           # Learning rate
        gamma=0.95,          # Discount factor
        reward_scale=10.0,   # Scale factor for rewards
        epsilon=1.0,         # Initial exploration rate
        epsilon_min=0.01,    # Minimum exploration rate
        epsilon_decay=0.995, # Decay per episode
    ):
        self.profiling = profiling_data
        self.is_test = is_test
        self.alpha = alpha
        self.gamma = gamma
        self.reward_scale = reward_scale
        
        # Double Q-learning tables
        self.Q1 = {}   # First Q-table: (state_key, action_key) -> value
        self.Q2 = {}   # Second Q-table
        
        # Simulator
        self.simulator = CloudEdgeSimulator(profiling_data)
        
        # DISCRETIZATION BINS
        self.bandwidth_bins = np.linspace(1, 15, 15)      # Bandwidth: 1-15 MBps
        self.cloudtime_bins = np.linspace(0, 100, 20)     # Cloud contention: 0-100 ms
        
        # EXPLORATION PARAMETERS
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # PERFORMANCE TRACKING
        self.best_episode_latency = float('inf')
        self.episodes_since_improvement = 0
        self.stagnant_limit = 500       # Much smaller than Actor-Critic
        self.total_episodes = 0
        self.current_episode_latency = 0.0
        self.current_episode_reward = 0.0
    
    # =========================================================================
    # STATE AND ACTION DISCRETIZATION
    # =========================================================================
    
    def _discretize(self, value, bins):
        """Convert continuous value to discrete bin center."""
        idx = np.digitize([value], bins, right=True)[0] - 1
        return float(bins[max(0, min(idx, len(bins) - 1))])
    
    def _state_to_key(self, state):
        bw, ctime, layer, prev_assign = state
        
        bw_bin = self._discretize(float(bw), self.bandwidth_bins)
        ctime_bin = self._discretize(float(ctime), self.cloudtime_bins)
        
        if prev_assign is None:
            prev_key = None
        else:
            # Extract the assignment column (0/1) from prev_assign
            # prev_assign could be a numpy array (shape nodes x 2) or a tuple/list of pairs
            if hasattr(prev_assign, 'shape') and len(prev_assign.shape) == 2:
                # numpy array: take second column
                assignments = prev_assign[:, 1]
            else:
                # Assume iterable of pairs (node_idx, assignment) or just assignments
                # Try to get second element of each item
                try:
                    assignments = [item[1] for item in prev_assign]
                except (TypeError, IndexError):
                    # If it's already a flat list of assignments, use directly
                    assignments = prev_assign
            prev_key = tuple(int(x) for x in assignments)
        
        return (bw_bin, ctime_bin, int(layer), prev_key)
    
    
    def _action_to_key(self, action):
        """Convert action matrix to hashable key."""
        # action is a numpy array with shape (num_nodes, 2), we take the assignment column
        return tuple(int(x) for x in action[:, 1])
    
    # =========================================================================
    # ACTION SPACE
    # =========================================================================
    
    def _get_possible_actions(self, layer_idx):
        """Get all possible actions for a layer using the simulator."""
        return self.simulator.get_possible_actions(layer_idx)
    
    # =========================================================================
    # Q-VALUE ACCESS (with default 0.0)
    # =========================================================================
    
    def _get_q1(self, state_key, action_key):
        """Get Q1 value, default 0.0."""
        return self.Q1.get((state_key, action_key), 0.0)
    
    def _get_q2(self, state_key, action_key):
        """Get Q2 value, default 0.0."""
        return self.Q2.get((state_key, action_key), 0.0)
    
    def _get_q_avg(self, state_key, action_key):
        """Average of both Q-tables (used for action selection)."""
        q1 = self._get_q1(state_key, action_key)
        q2 = self._get_q2(state_key, action_key)
        return (q1 + q2) / 2.0
    
    # =========================================================================
    # ACTION SELECTION (ε-greedy on average Q)
    # =========================================================================
    
    def choose_action(self, state):
        """Select action using ε-greedy policy based on average Q-values."""
        layer = int(state[2])   # Current layer index
        actions = self._get_possible_actions(layer)
        state_key = self._state_to_key(state)
        
        # Exploration: random action
        if not self.is_test and random.random() < self.epsilon:
            return random.choice(actions)
        
        # Exploitation: choose action maximizing average Q
        best_action = None
        best_value = -float('inf')
        
        for action in actions:
            action_key = self._action_to_key(action)
            q_avg = self._get_q_avg(state_key, action_key)
            if q_avg > best_value:
                best_value = q_avg
                best_action = action
        
        # If all actions have equal value (e.g., all zero), pick first
        if best_action is None:
            best_action = actions[0]
        
        return best_action
    
    # =========================================================================
    # ENVIRONMENT INTERACTION
    # =========================================================================
    
    def step(self, current_state):
        """
        Execute one step in the environment.
        
        Returns:
            action: Chosen action
            reward: Immediate reward
            latency: Raw latency value (seconds)
            next_state: Next state
            done: Whether episode is complete
        """
        # Step 1: Choose action
        action = self.choose_action(current_state)
        
        # Step 2: Get cloud waiting time for next layer (required by simulator)
        layer = int(current_state[2])
        next_layer = min(layer + 1, len(self.profiling.layers) - 1)
        
        cloud_waiting_time = self.simulator.get_next_state_cloud_waiting_time(
            next_layer=next_layer,
            current_action=action,
            isAllCloud=False,
        )
        
        # Step 3: Compute latency
        latency_s = self.simulator.compute_latency(
            current_state=current_state,
            current_action=action,
            cloud_pending_ms=cloud_waiting_time,
        )
        
        # Step 4: Calculate reward (pure latency minimization)
        reward = self.simulator.calculate_latency_reward(
            latency_s, 
            scale_factor=self.reward_scale
        )
        
        # Step 5: Get next state
        next_state, done = self.simulator.get_next_state(
            current_state=current_state,
            action=action,
            new_cloud_pending=cloud_waiting_time,
        )
        
        # Step 6: Track metrics
        latency_ms = latency_s * 1000
        self.current_episode_latency += latency_ms
        self.current_episode_reward += reward
        
        return action, reward, latency_s, next_state, done
    
    # =========================================================================
    # DOUBLE Q-LEARNING UPDATE
    # =========================================================================
    
    def update(self, state, action, reward, next_state, done):
        """
        One-step Double Q-learning update.
        Randomly chooses which Q-table to update, uses the other for the target.
        """
        state_key = self._state_to_key(state)
        action_key = self._action_to_key(action)
        next_state_key = self._state_to_key(next_state)
        
        # Decide which Q-table to update (random)
        if random.random() < 0.5:
            Q_current = self.Q1
            Q_other = self.Q2
        else:
            Q_current = self.Q2
            Q_other = self.Q1
        
        # Compute target using Double Q-learning
        if done:
            target = reward
        else:
            # Get possible actions in next state
            next_layer = int(next_state[2])
            next_actions = self._get_possible_actions(next_layer)
            
            # Step 1: Select greedy action using the current Q-table
            best_action_key = None
            best_q_current = -float('inf')
            for next_action in next_actions:
                next_action_key = self._action_to_key(next_action)
                q_val = Q_current.get((next_state_key, next_action_key), 0.0)
                if q_val > best_q_current:
                    best_q_current = q_val
                    best_action_key = next_action_key
            
            # Step 2: Evaluate that action using the other Q-table
            if best_action_key is None:
                # Fallback (should not happen)
                target = reward
            else:
                q_other_val = Q_other.get((next_state_key, best_action_key), 0.0)
                target = reward + self.gamma * q_other_val
        
        # TD error
        old_q = Q_current.get((state_key, action_key), 0.0)
        td_error = target - old_q
        td_error = np.clip(td_error, -10000.0, 10000.0)
        
        # Update the chosen Q-table
        new_q = old_q + self.alpha * td_error
        Q_current[(state_key, action_key)] = new_q
        
        return td_error
    
    # =========================================================================
    # EPISODE MANAGEMENT
    # =========================================================================
    
    def start_episode(self):
        """Reset episode tracking and simulator time."""
        self.current_episode_latency = 0.0
        self.current_episode_reward = 0.0
        self.simulator.reset_episode_time()
    
    def end_episode(self):
        """
        Update exploration parameters based on performance.
        
        Returns:
            tuple: (total_latency_ms, total_reward)
        """
        total_latency = self.current_episode_latency
        total_reward = self.current_episode_reward
        
        self.total_episodes += 1
        
        # Check for improvement (lower latency = better)
        if total_latency < self.best_episode_latency:
            self.best_episode_latency = total_latency
            self.episodes_since_improvement = 0
        else:
            self.episodes_since_improvement += 1
            
            # Boost exploration if stagnant
            if self.episodes_since_improvement >= self.stagnant_limit:
                self.epsilon = min(0.5, self.epsilon * 1.5)  # Boost exploration
                self.episodes_since_improvement = 0
                print(f"🔥 Epsilon boosted to {self.epsilon:.3f}")
            else:
                # Normal decay
                self.epsilon = max(
                    self.epsilon_min,
                    self.epsilon * self.epsilon_decay
                )
        
        return total_latency, total_reward
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save(self, file="double_q_tables.pkl"):
        """Save both Q-tables."""
        with open(file, "wb") as f:
            pickle.dump((self.Q1, self.Q2), f)
        print(f"💾 Double Q-learning agent saved to {file}")
    
    def load(self, file="double_q_tables.pkl"):
        """Load both Q-tables."""
        if os.path.exists(file):
            with open(file, "rb") as f:
                self.Q1, self.Q2 = pickle.load(f)
            print(f"✅ Double Q-learning agent loaded from {file}")
        else:
            print(f"⚠️ File {file} not found. Starting from scratch.")