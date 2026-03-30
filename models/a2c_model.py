import numpy as np
import random
import pickle
import os
from profiling.profiling_class import ProfilingData
from simulator.simulator import CloudEdgeSimulator


class TabularActorCriticAgent:
    """
    Tabular Actor–Critic for PURE LATENCY MINIMIZATION
    
    GOAL: Minimize total inference latency (execution + communication time)
    
    STATE: [bandwidth, cloud_contention, current_layer, previous_assignment]
        - bandwidth: Available network bandwidth (MBps)
        - cloud_contention: Cloud server waiting time (ms)
        - current_layer: Which DNN layer we're processing (0 to L-1)
        - previous_assignment: Where previous layer's nodes ran (tuple of 0/1 values)
    
    ACTION: Binary assignment for each node in current layer
        - 0: Execute on Edge device
        - 1: Execute on Cloud server
    
    ALGORITHM: One-step Actor-Critic (online updates, no trajectory)
    """
    
    def __init__(
        self,
        profiling_data: ProfilingData,
        is_test=False,
        alpha_actor=0.02,      # Actor learning rate
        alpha_critic=0.05,     # Critic learning rate  
        gamma=0.95,            # Discount factor
        reward_scale=10.0,     # Scale factor for rewards
    ):
        self.profiling = profiling_data
        self.is_test = is_test
        self.gamma = gamma
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.reward_scale = reward_scale
        
        # TABLES
        self.policy_table = {}   # (state_key, action_key) → preference score
        self.value_table = {}    # state_key → state value V(s)
        
        # Simulator
        self.simulator = CloudEdgeSimulator(profiling_data)
        
        # DISCRETIZATION BINS
        self.bandwidth_bins = np.linspace(1, 15, 15)      # Bandwidth: 1-15 MBps
        self.cloudtime_bins = np.linspace(0, 100, 20)     # Cloud contention: 0-100 ms
        
        # EXPLORATION PARAMETERS
        # self.temperature = 1.0
        # self.temperature_min = 0.025
        # self.temperature_decay = 0.999
        # self.temperature_boost = 1.35
        # self.epsilon_min = 0.05
        # EXPLORATION PARAMETERS

        self.temperature = 0.01
        self.temperature_min = 0.01
        self.temperature_decay = 0.9  # decay
        self.temperature_boost = 0.1   # More reasonable boost
        self.epsilon_min = 0

        # PERFORMANCE TRACKING
        self.best_episode_latency = float('inf')
        self.episodes_since_improvement = 0
        self.stagnant_limit = 10000
        self.total_episodes = 0
        self.current_episode_latency = 0.0   # Total latency (ms)
        self.current_episode_reward = 0.0    # Total reward
    
    # =========================================================================
    # STATE AND ACTION DISCRETIZATION
    # =========================================================================
    
    def _discretize(self, value, bins):
        """Convert continuous value to discrete bin center."""
        idx = np.digitize([value], bins, right=True)[0] - 1
        return float(bins[max(0, min(idx, len(bins) - 1))])
    
    def _state_to_key(self, state):
        """
        Convert state to hashable key.
        
        State format: [bandwidth, cloud_contention, layer, previous_assignment]
        """
        bw, ctime, layer, prev_assign = state
        
        return (
            self._discretize(float(bw), self.bandwidth_bins),
            self._discretize(float(ctime), self.cloudtime_bins),
            int(layer),
        )
    
    def _action_to_key(self, action):
        """Convert action matrix to hashable key."""
        return tuple(int(x) for x in action[:, 1])
    
    # =========================================================================
    # ACTION SPACE
    # =========================================================================
    
    def _get_possible_actions(self, layer_idx):
        """Get all possible actions for a layer using the simulator."""
        return self.simulator.get_possible_actions(layer_idx)
    
    # =========================================================================
    # ACTION SELECTION
    # =========================================================================
    
    def choose_action(self, state):
        """Select action using epsilon-greedy + softmax."""
        layer = int(state[2])  # Current layer index
        actions = self._get_possible_actions(layer)
        state_key = self._state_to_key(state)
        
        # Forced exploration
        if not self.is_test and random.random() < self.epsilon_min:
            return random.choice(actions)
        
        # Get preference scores
        preferences = []
        for action in actions:
            action_key = self._action_to_key(action)
            pref = self.policy_table.get((state_key, action_key), 0.0)
            preferences.append(pref)
        
        preferences = np.array(preferences)
        
        # Softmax with temperature
        scaled_prefs = preferences / max(self.temperature, 1e-6)
        scaled_prefs -= np.max(scaled_prefs)
        
        probs = np.exp(scaled_prefs)
        probs /= np.sum(probs)
        
        # Test mode: greedy
        if self.is_test:
            best_idx = int(np.argmax(probs))
            return actions[best_idx]
        
        # Training mode: sample
        chosen_idx = np.random.choice(len(actions), p=probs)
        return actions[chosen_idx]
    
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
        
        # Step 2: Get cloud waiting time for next layer
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
    # ONLINE UPDATE
    # =========================================================================
    
    def update(self, state, action, reward, next_state, done):
        """
        One-step TD update for Actor-Critic.
        """
        state_key = self._state_to_key(state)
        action_key = self._action_to_key(action)
        next_state_key = self._state_to_key(next_state)
        
        # Current value
        V_current = self.value_table.get(state_key, 0.0)
        
        # TD Target
        if done:
            target = reward
        else:
            V_next = self.value_table.get(next_state_key, 0.0)
            target = reward + self.gamma * V_next
        
        # TD Error
        td_error = target - V_current
        td_error = np.clip(td_error, -10000.0, 10000.0)
        
        # Update Critic
        self.value_table[state_key] = V_current + self.alpha_critic * td_error
        
        # Update Actor
        old_pref = self.policy_table.get((state_key, action_key), 0.0)
        new_pref = old_pref + self.alpha_actor * td_error
        self.policy_table[(state_key, action_key)] = np.clip(new_pref, -500.0, 500.0)
        
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
            
            # Reduce exploration when improving
            self.temperature = max(
                self.temperature_min,
                self.temperature * 0.995
            )
        else:
            self.episodes_since_improvement += 1
            
            # Boost exploration if stagnant
            if self.episodes_since_improvement >= self.stagnant_limit:
                self.temperature = min(
                    2.0,
                    self.temperature * self.temperature_boost
                )
                self.episodes_since_improvement = 0
                print(f"🔥 Temperature boosted to {self.temperature:.3f}")
            else:
                # Slight decay
                self.temperature = max(
                    self.temperature_min,
                    self.temperature * self.temperature_decay
                )
        
        return total_latency, total_reward
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save(self, file="a2c_tables.pkl"):
        """Save learned tables."""
        with open(file, "wb") as f:
            pickle.dump((self.policy_table, self.value_table), f)
        print(f"💾 Agent saved to {file}")
    
    def load(self, file="a2c_tables.pkl"):
        """Load learned tables."""
        if os.path.exists(file):
            with open(file, "rb") as f:
                self.policy_table, self.value_table = pickle.load(f)
            print(f"✅ Agent loaded from {file}")
        else:
            print(f"⚠️ File {file} not found. Starting from scratch.")