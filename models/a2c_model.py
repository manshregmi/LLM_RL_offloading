import time
import numpy as np
import random
import pickle
import os
from profiling.profiling_class import ProfilingData
from simulator.simulator import CloudEdgeSimulator
from collections import Counter


class TabularActorCriticAgent:
    """
    Tabular Actor–Critic with optional group action caching.
    """
    
    def __init__(
        self,
        profiling_data: ProfilingData,
        is_test=False,
        alpha_actor=0.02,
        alpha_critic=0.05,
        gamma=0.95,
        reward_scale=10.0,
        total_pipelines=1
    ):
        self.profiling = profiling_data
        self.is_test = is_test
        self.gamma = gamma
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.reward_scale = reward_scale
        
        self.policy_table = {}
        self.value_table = {}
        self.simulator = CloudEdgeSimulator(profiling_data,total_pipeline=total_pipelines)
        
        self.bandwidth_bins = np.linspace(1, 15, 15)
        self.cloudtime_bins = np.linspace(0, 100, 20)
        
        self.temperature = 1.0
        self.temperature_min = 0.01
        self.temperature_decay = 0.999
        self.temperature_boost = 1.5
        
        self.best_episode_latency = float('inf')
        self.episodes_since_improvement = 0
        self.stagnant_limit = 10000
        self.total_episodes = 0
        self.current_episode_latency = 0.0
        self.current_episode_reward = 0.0
        
        self.group_range_assignments = {}   # cache for group actions
    
    # -------------------------------------------------------------------------
    # State & Action Helpers
    # -------------------------------------------------------------------------
    def _discretize(self, value, bins):
        idx = np.digitize([value], bins, right=True)[0] - 1
        return float(bins[max(0, min(idx, len(bins) - 1))])
    
    def _state_to_key(self, state):
        bw, ctime, layer, prev_assign = state
        prev_tuple = tuple(int(x) for x in prev_assign) if prev_assign is not None else (0, 0)
        return (
            self._discretize(float(bw), self.bandwidth_bins),
            self._discretize(float(ctime), self.cloudtime_bins),
            int(layer),
            prev_tuple
        )
    
    def _action_to_key(self, action):
        return tuple(int(x) for x in action[:, 1])
    
    def _get_possible_actions(self, layer_idx):
        return self.simulator.get_possible_actions(layer_idx)
    
    # -------------------------------------------------------------------------
    # Group Caching Helpers
    # -------------------------------------------------------------------------
    def _get_chunk_index(self, layer, num_groups):
        total_layers = len(self.profiling.layers)
        layers_per_chunk = max(1, total_layers // num_groups)
        return layer // layers_per_chunk
    
    # -------------------------------------------------------------------------
    # Action Selection (with optional group caching)
    # -------------------------------------------------------------------------

    def choose_action(self, state, num_groups=None):
        original_prev_assignment = state[3] if len(state) > 3 else None
        state_bw = float(state[0])
        state_ctime = float(state[1])
        layer = int(state[2])

        # ---------- Grouped case: use majority voting over the whole chunk ----------
        if num_groups is not None and num_groups > 0:
            chunk_idx = self._get_chunk_index(layer, num_groups)
            cache_key = (num_groups, chunk_idx)

            # 1. if already cached, return the stored assignment
            if cache_key in self.group_range_assignments:
                return self.group_range_assignments[cache_key]

            # 2. not cached → simulate all layers in this chunk, collect majority action
            start_layer, end_layer = self._get_chunk_range(chunk_idx, num_groups)

            # initial state for the first layer of the chunk
            curr_state = (state_bw, state_ctime, start_layer, original_prev_assignment)
            actions_in_chunk = []

            for l in range(start_layer, end_layer + 1):
                # get action for the current state using the policy
                act = self._get_policy_action(curr_state)
                actions_in_chunk.append(act)

                # update state for next layer: keep BW/cloudtime, increment layer, set prev_action = act
                curr_state = (state_bw, state_ctime, l + 1, act)

            # majority vote
            if actions_in_chunk:
                counter = Counter(actions_in_chunk)
                max_count = max(counter.values())
                # collect all actions that have the highest frequency
                majority_actions = [a for a, cnt in counter.items() if cnt == max_count]
                if len(majority_actions) == 1:
                    chosen_action = majority_actions[0]
                else:
                    # tie → fallback to the previous assignment of the first layer
                    chosen_action = original_prev_assignment
            else:
                # empty chunk (should not happen), fallback
                chosen_action = original_prev_assignment

            # cache the result for this chunk and return it
            self.group_range_assignments[cache_key] = chosen_action
            return chosen_action

        # ---------- No grouping: original single‑layer selection ----------
        actions = self._get_possible_actions(layer)
        state_key = self._state_to_key(state)

        preferences = []
        for action in actions:
            akey = self._action_to_key(action)
            pref = self.policy_table.get((state_key, akey), 0.0)
            preferences.append(pref)
        preferences = np.array(preferences)

        scaled = preferences / max(self.temperature, 1e-8)
        scaled -= np.max(scaled)
        probs = np.exp(scaled) / np.sum(np.exp(scaled))

        if self.is_test:
            best_idx = int(np.argmax(probs))
            chosen_action = actions[best_idx]
        else:
            chosen_idx = np.random.choice(len(actions), p=probs)
            chosen_action = actions[chosen_idx]

        return chosen_action

    # ----------------------------------------------------------------------
    # Helper methods (add to your class)
    # ----------------------------------------------------------------------
    def _get_chunk_range(self, chunk_idx, num_groups):
        """Return (start_layer, end_layer) inclusive for the given chunk index."""
        total_layers = len(self.profiling.layers)
        layers_per_chunk = max(1, total_layers // num_groups)
        start = chunk_idx * layers_per_chunk
        end = min((chunk_idx + 1) * layers_per_chunk, total_layers) - 1
        return start, end

    def _get_policy_action(self, state):
        """
        Given a state tuple (bw, cloud_ctime, layer, prev_action),
        return an action according to the current policy (softmax, temperature, test mode).
        """
        actions = self._get_possible_actions(int(state[2]))
        state_key = self._state_to_key(state)

        preferences = []
        for action in actions:
            akey = self._action_to_key(action)
            pref = self.policy_table.get((state_key, akey), 0.0)
            preferences.append(pref)
        preferences = np.array(preferences)

        scaled = preferences / max(self.temperature, 1e-8)
        scaled -= np.max(scaled)
        probs = np.exp(scaled) / np.sum(np.exp(scaled))

        if self.is_test:
            best_idx = int(np.argmax(probs))
            return actions[best_idx]
        else:
            chosen_idx = np.random.choice(len(actions), p=probs)
            return actions[chosen_idx]
    # -------------------------------------------------------------------------
    # Environment Step
    # -------------------------------------------------------------------------
    def step(self, current_state, num_groups=None):

        start_time = time.time()
        action = self.choose_action(current_state, num_groups=num_groups)
        overhead_time_per_step = time.time() - start_time
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
        
        reward = self.simulator.calculate_latency_reward(
            latency_s=latency_s,
        )
        
        next_state, done = self.simulator.get_next_state(
            current_state=current_state,
            action=action,
            new_cloud_pending=cloud_waiting_time,
        )
        
        latency_ms = latency_s * 1000
        self.current_episode_latency += latency_ms
        self.current_episode_reward += reward
        
        return action, reward, latency_s, next_state, done, overhead_time_per_step
    
    # -------------------------------------------------------------------------
    # Update (Improved Policy Gradient)
    # -------------------------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        state_key = self._state_to_key(state)
        action_key = self._action_to_key(action)
        next_state_key = self._state_to_key(next_state)
        
        V_current = self.value_table.get(state_key, 0.0)
        if done:
            target = reward
        else:
            V_next = self.value_table.get(next_state_key, 0.0)
            target = reward + self.gamma * V_next
        
        td_error = np.clip(target - V_current, -10.0, 10.0)
        
        # Update critic
        self.value_table[state_key] = V_current + self.alpha_critic * td_error
        
        # Update actor using full softmax gradient
        layer = int(state[2])
        actions = self._get_possible_actions(layer)
        prefs = []
        for a in actions:
            akey = self._action_to_key(a)
            prefs.append(self.policy_table.get((state_key, akey), 0.0))
        prefs = np.array(prefs)
        
        scaled = prefs / max(self.temperature, 1e-8)
        scaled -= np.max(scaled)
        probs = np.exp(scaled) / np.sum(np.exp(scaled))
        
        chosen_idx = None
        for i, a in enumerate(actions):
            if self._action_to_key(a) == action_key:
                chosen_idx = i
                break
        
        for i, a in enumerate(actions):
            akey = self._action_to_key(a)
            old = self.policy_table.get((state_key, akey), 0.0)
            if i == chosen_idx:
                grad = td_error * (1 - probs[i])
            else:
                grad = -td_error * probs[i]
            new_pref = old + self.alpha_actor * grad
            self.policy_table[(state_key, akey)] = np.clip(new_pref, -500.0, 500.0)
        
        return td_error
    
    # -------------------------------------------------------------------------
    # Episode Management
    # -------------------------------------------------------------------------
    def start_episode(self):
        self.current_episode_latency = 0.0
        self.current_episode_reward = 0.0
        self.simulator.reset_episode_time()
        self.group_range_assignments.clear()   # Clear group cache
    
    def end_episode(self):
        total_latency = self.current_episode_latency
        total_reward = self.current_episode_reward
        self.total_episodes += 1
        
        if total_latency < self.best_episode_latency:
            self.best_episode_latency = total_latency
            self.episodes_since_improvement = 0
            self.temperature = max(self.temperature_min, self.temperature * 0.999)
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