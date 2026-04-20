import numpy as np
import random
import pickle
import os
from itertools import product
from profiling.profiling_class import ProfilingData
from simulator.simulator import CloudEdgeSimulator

class TabularActorCriticAgent:
    """
    Tabular Actor–Critic with GROUPING support.
    
    Groups are contiguous node sets. All nodes in a group share the same assignment.
    The agent chooses a group‑level action (tuple of 0/1, length = num_groups),
    then expands it to a per‑node action matrix.
    """
    
    def __init__(self, profiling_data: ProfilingData, is_test=False,
                 alpha_actor=0.02, alpha_critic=0.05, gamma=0.95, reward_scale=10.0):
        self.profiling = profiling_data
        self.is_test = is_test
        self.gamma = gamma
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.reward_scale = reward_scale
        
        self.policy_table = {}
        self.value_table = {}
        self.simulator = CloudEdgeSimulator(profiling_data)
        
        # Discretization
        self.bandwidth_bins = np.linspace(1, 15, 15)
        self.cloudtime_bins = np.linspace(0, 100, 20)
        
        # Exploration
        self.temperature = 0.5
        self.temperature_min = 0.01
        self.temperature_decay = 0.999
        self.temperature_boost = 1.5          # FIXED: boost >1 to increase exploration
        self.epsilon_min = 0
        
        # Performance tracking
        self.best_episode_latency = float('inf')
        self.episodes_since_improvement = 0
        self.stagnant_limit = 10000
        self.total_episodes = 0
        self.current_episode_latency = 0.0
        self.current_episode_reward = 0.0
        
        # For grouping update
        self.current_layer = None
        self.current_num_groups = None
        self.last_group_action_key = None
    
    # -------------------------------------------------------------------------
    # Helper: get number of nodes for a layer
    # -------------------------------------------------------------------------
    def _get_num_nodes(self, layer_idx):
        """Return number of nodes in the given layer."""
        # Assuming profiling_data.layers[layer_idx].nodes is a list
        return self.profiling.get_num_nodes(layer_idx)
    
    # -------------------------------------------------------------------------
    # Group creation (contiguous, balanced sizes)
    # -------------------------------------------------------------------------
    def _create_groups(self, num_nodes, num_groups):
        """
        Split nodes into contiguous groups.
        Returns a list of lists, e.g., [[0,1,2], [3,4], [5]] for 6 nodes, 3 groups.
        """
        if num_groups >= num_nodes:
            # Each node its own group
            return [[i] for i in range(num_nodes)]
        base = num_nodes // num_groups
        remainder = num_nodes % num_groups
        groups = []
        start = 0
        for i in range(num_groups):
            size = base + (1 if i < remainder else 0)
            groups.append(list(range(start, start + size)))
            start += size
        return groups
    
    # -------------------------------------------------------------------------
    # Action space for groups
    # -------------------------------------------------------------------------
    def _get_possible_group_actions(self, num_groups):
        """Return all possible group assignment tuples (length = num_groups)."""
        return list(product([0, 1], repeat=num_groups))
    
    def _expand_group_action(self, group_action, groups, num_nodes):
        """
        Convert group_action (tuple of 0/1) to node action matrix (nodes x 2).
        Column 0 = node index, column 1 = assignment (0 edge, 1 cloud).
        """
        action_mat = np.zeros((num_nodes, 2), dtype=int)
        action_mat[:, 0] = np.arange(num_nodes)
        for g_idx, node_list in enumerate(groups):
            assign = group_action[g_idx]
            for node in node_list:
                action_mat[node, 1] = assign
        return action_mat
    
    # -------------------------------------------------------------------------
    # State discretization (unchanged, but note previous_assignment is ignored)
    # -------------------------------------------------------------------------
    def _discretize(self, value, bins):
        idx = np.digitize([value], bins, right=True)[0] - 1
        return float(bins[max(0, min(idx, len(bins) - 1))])
    
    def _state_to_key(self, state):
        bw, ctime, layer, _ = state
        return (
            self._discretize(float(bw), self.bandwidth_bins),
            self._discretize(float(ctime), self.cloudtime_bins),
            int(layer),
        )
    
    # -------------------------------------------------------------------------
    # Action selection with grouping
    # -------------------------------------------------------------------------
    def choose_action(self, state, num_groups):
        """
        Select a group assignment, then expand to node action matrix.
        Returns the node‑level action matrix.
        """
        layer = int(state[2])
        num_nodes = self._get_num_nodes(layer)
        groups = self._create_groups(num_nodes, num_groups)
        possible_actions = self._get_possible_group_actions(num_groups)
        
        state_key = self._state_to_key(state)
        
        # Compute preferences for each possible group action
        preferences = []
        for group_act in possible_actions:
            # Use the tuple itself as action key
            pref = self.policy_table.get((state_key, group_act), 0.0)
            preferences.append(pref)
        
        preferences = np.array(preferences)
        
        # Softmax with temperature
        scaled = preferences / max(self.temperature, 1e-6)
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        probs /= np.sum(probs)
        
        if self.is_test:
            idx = int(np.argmax(probs))
        else:
            # epsilon-greedy floor (if epsilon_min > 0)
            if random.random() < self.epsilon_min:
                idx = random.randrange(len(possible_actions))
            else:
                idx = np.random.choice(len(possible_actions), p=probs)
        
        chosen_group_action = possible_actions[idx]
        
        # Expand to node matrix
        action_matrix = self._expand_group_action(chosen_group_action, groups, num_nodes)
        
        # Store for later update
        self.last_group_action_key = chosen_group_action
        self.current_layer = layer
        self.current_num_groups = num_groups
        
        return action_matrix
    
    # -------------------------------------------------------------------------
    # Environment step (now accepts numberOfGroups)
    # -------------------------------------------------------------------------
    def step(self, current_state, numberOfGroups=399):
        """Execute one step, using grouping."""
        # Choose action with given number of groups
        action = self.choose_action(current_state, numberOfGroups)
        
        layer = int(current_state[2])
        next_layer = min(layer + 1, len(self.profiling.layers) - 1)
        
        cloud_waiting_time = self.simulator.get_next_state_cloud_waiting_time(
            next_layer=next_layer,
            current_action=action,
            isAllCloud=False,
        )
        
        latency_s = self.simulator.compute_latency(
            current_state=current_state,
            current_action=action,
            cloud_pending_ms=cloud_waiting_time,
        )
        
        reward = self.simulator.calculate_latency_reward(latency_s, scale_factor=self.reward_scale)
        
        next_state, done = self.simulator.get_next_state(
            current_state=current_state,
            action=action,
            new_cloud_pending=cloud_waiting_time,
        )
        
        latency_ms = latency_s * 1000
        self.current_episode_latency += latency_ms
        self.current_episode_reward += reward
        
        return action, reward, latency_s, next_state, done
    
    # -------------------------------------------------------------------------
    # Update (uses stored group action key)
    # -------------------------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        """One-step TD update for Actor-Critic using group action key."""
        state_key = self._state_to_key(state)
        action_key = self.last_group_action_key   # tuple of group assignments
        next_state_key = self._state_to_key(next_state)
        
        V_current = self.value_table.get(state_key, 0.0)
        if done:
            target = reward
        else:
            V_next = self.value_table.get(next_state_key, 0.0)
            target = reward + self.gamma * V_next
        
        td_error = np.clip(target - V_current, -10000.0, 10000.0)
        
        # Critic update
        self.value_table[state_key] = V_current + self.alpha_critic * td_error
        
        # Actor update
        old_pref = self.policy_table.get((state_key, action_key), 0.0)
        new_pref = old_pref + self.alpha_actor * td_error
        self.policy_table[(state_key, action_key)] = np.clip(new_pref, -500.0, 500.0)
        
        return td_error
    
    # -------------------------------------------------------------------------
    # Episode management (unchanged)
    # -------------------------------------------------------------------------
    def start_episode(self):
        self.current_episode_latency = 0.0
        self.current_episode_reward = 0.0
        self.simulator.reset_episode_time()
    
    def end_episode(self):
        total_latency = self.current_episode_latency
        total_reward = self.current_episode_reward
        self.total_episodes += 1
        
        if total_latency < self.best_episode_latency:
            self.best_episode_latency = total_latency
            self.episodes_since_improvement = 0
            self.temperature = max(self.temperature_min, self.temperature * 0.995)
        else:
            self.episodes_since_improvement += 1
            if self.episodes_since_improvement >= self.stagnant_limit:
                self.temperature = min(2.0, self.temperature * self.temperature_boost)
                self.episodes_since_improvement = 0
                print(f"🔥 Temperature boosted to {self.temperature:.3f}")
            else:
                self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)
        
        return total_latency, total_reward
    
    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def save(self, file="a2c_tables.pkl"):
        with open(file, "wb") as f:
            pickle.dump((self.policy_table, self.value_table), f)
        print(f"💾 Agent saved to {file}")
    
    def load(self, file="a2c_tables.pkl"):
        if os.path.exists(file):
            with open(file, "rb") as f:
                self.policy_table, self.value_table = pickle.load(f)
            print(f"✅ Agent loaded from {file}")
        else:
            print(f"⚠️ File {file} not found. Starting from scratch.")