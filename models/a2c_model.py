import time
import numpy as np
import random
import pickle
import os
from profiling.profiling_class import ProfilingData
from simulator.simulator import CloudEdgeSimulator
from collections import Counter
import pandas as pd


class TabularActorCriticAgent:
    """
    Tabular Actor–Critic with optional group action caching (level‑only caching).
    """

    def __init__(
        self,
        profiling_data: ProfilingData,
        is_test=False,
        alpha_actor=0.02,
        alpha_critic=0.05,
        #gamma=0.95,
        reward_scale=10.0,
        total_pipelines=1,
        average_reward_lr=0.15,
        uncertainty_models = None 
    ):
        self.profiling = profiling_data
        self.is_test = is_test
        #self.gamma = gamma
        self.average_reward_lr = average_reward_lr
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.reward_scale = reward_scale
        # Running estimate of the soft-robust average reward
        self.average_reward = 0.0
        #bandwidth distribution 
        self._uncertainty_models_arg = uncertainty_models 

        self.policy_table = {}
        self.value_table = {}
        self.simulator = CloudEdgeSimulator(profiling_data, total_pipeline=total_pipelines)
        bw_path = os.path.join("simulator", "data", "bw_data.csv")
        df = pd.read_csv(bw_path)
        bw_mbps = df["bandwidth_mbps"]

        bw_min_floor = np.floor(np.min(bw_mbps)/8)
        bw_max_ceil = np.ceil(np.max(bw_mbps)/8)
        self.bandwidth_bins = np.linspace(bw_min_floor, bw_max_ceil, 15)
        if self._uncertainty_models_arg is None:
            self.uncertainty_models = self._build_empirical_uncertainty_models(bw_mbps)
        else:
            self.uncertainty_models = [dict(m) for m in self._uncertainty_models_arg]
        

        self.cloudtime_bins = np.linspace(0, 45, 20)

        self.temperature = 1.0
        self.temperature_min = 0.01
        self.temperature_decay = 0.9995
        self.temperature_boost = 1.5

        self.best_episode_latency = float('inf')
        self.episodes_since_improvement = 0
        self.stagnant_limit = 5000
        self.total_episodes = 0
        self.current_episode_latency = 0.0
        self.current_episode_reward = 0.0

        self.group_range_assignments = {}   # cache: key = (num_groups, chunk_idx)

    ##---------
    def _build_empirical_uncertainty_models(self, bw_mbps):
        """
        Build omega from the empirical bandwidth histogram of the trace,
        aligned to the value table's own discretization grid. Each occupied
        bin becomes one uncertainty model: representative bandwidth = mean of
        the trace samples in that bin (re-discretizes back to the same key),
        weight = fraction of trace samples in that bin.
        """
        bw_MBps = np.asarray(bw_mbps, dtype=float) / 8.0  # grid is in MB/s

        groups = {}
        for v in bw_MBps:
            key = self._discretize(float(v), self.bandwidth_bins)
            groups.setdefault(key, []).append(v)

        keys = sorted(groups)
        counts = np.array([len(groups[k]) for k in keys], dtype=float)
        freqs = counts / counts.sum()

        models = []
        for k, w in zip(keys, freqs):
            rep_MBps = float(np.mean(groups[k]))
            models.append({
                "name": f"bw_bin_{k:.3f}_MBps",
                "weight": float(w),
                "bandwidth_mbps": float(rep_MBps * 8.0),
            })
        return models    

    # -------------------------------------------------------------------------
    # State & Action Helpers
    # -------------------------------------------------------------------------
    def _discretize(self, value, bins):
        idx = np.digitize([value], bins, right=True)[0] - 1
        return float(bins[max(0, min(idx, len(bins) - 1))])
    
    def _state_to_key(self, state):
        # Discretise continuous parts for table lookup
        # NEw state = (BW, Contention, model segmet, previous assignment vector )
        # current_bandwidth,      # Updated bandwidth
            # new_cloud_pending,      # Cloud contention for next layer
            # next_layer,             # Next layer index
            # prev_action_pattern,

        bw_disc = self._discretize(float(state[0]), self.bandwidth_bins)
        ctime_disc = self._discretize(float(state[1]), self.cloudtime_bins)
        prev_action = state[3]
        segment = state[2]

        if prev_action is None:
            prev_tuple = ()
        else:
            if isinstance(prev_action, np.ndarray) and prev_action.ndim == 2:
                assignments = prev_action[:, 1]
            elif isinstance(prev_action, (tuple, list)) and len(prev_action) > 0 and not isinstance(prev_action[0], (tuple, list)):
                assignments = prev_action
            else:
                assignments = np.array(prev_action).flatten()
            prev_tuple = tuple(int(x) for x in assignments)
        segment_tuple = ()
        if (isinstance(segment, (list, tuple))):
            for x in np.asarray(segment):
                segment_tuple += (x,)
        return (bw_disc, ctime_disc, segment_tuple, prev_tuple)

    def _action_to_key(self, action):
        return tuple(int(x) for x in action[:, 1])

    def _get_possible_actions(self):
        return self.simulator.get_possible_actions()

    # -------------------------------------------------------------------------
    # Group Caching Helpers (level only)
    # -------------------------------------------------------------------------
    def _get_chunk_index(self, layer, num_groups):
        total_layers = len(self.profiling.layers)
        # Use ceiling division to avoid out-of-range indices
        layers_per_chunk = (total_layers + num_groups - 1) // num_groups
        return min(layer // layers_per_chunk, num_groups - 1)

    def _get_chunk_range(self, chunk_idx, num_groups):
        """Return (start_layer, end_layer) inclusive for the given chunk index."""
        total_layers = len(self.profiling.layers)
        layers_per_chunk = (total_layers + num_groups - 1) // num_groups
        start = chunk_idx * layers_per_chunk
        end = min((chunk_idx + 1) * layers_per_chunk, total_layers) - 1
        return start, end

    def _action_to_hashable(self, action_2d):
        return tuple(int(action_2d[i, 1]) for i in range(action_2d.shape[0]))

    def _get_default_action(self, layer):
        possible = self._get_possible_actions()
        if possible:
            n_nodes = possible[0].shape[0]
        else:
            # Fallback: try to get node count from profiling
            try:
                n_nodes = self.profiling.get_num_nodes(layer)
            except AttributeError:
                n_nodes = 1
        action = np.zeros((n_nodes, 2), dtype=int)
        action[:, 0] = np.arange(n_nodes)
        return action

    # -------------------------------------------------------------------------
    # Action Selection (group caching uses level‑only key)
    # -------------------------------------------------------------------------
    def choose_action(self, state, num_groups=None, count=0, layer_index=0):
        original_prev_assignment = state[3] if len(state) > 3 else None
        state_bw = float(state[0])
        state_ctime = float(state[1])
        layer = self.simulator.get_current_layer_index()

        # ========== GROUPED CASE (level‑only caching) ==========
        if num_groups is not None and num_groups > 0:
            chunk_idx = self._get_chunk_index(layer, num_groups)
            cache_key = (num_groups, chunk_idx)   # no state

            if cache_key in self.group_range_assignments:
                count += 1   # optional debug counter
                grouped_assignment= self.group_range_assignments[cache_key]
                cached_assignment_vector = grouped_assignment[0][1]
                cached_assignments = []
                for _ in range(self.profiling.get_num_nodes(self.simulator.get_current_layer_index())):
                    cached_assignments.append([self.simulator.get_current_layer_index(), (1 if cached_assignment_vector > 0 else 0)])
                return np.array(cached_assignments), count
        
            # Simulate the whole chunk using the current state (raw values)
            start_layer, end_layer = self._get_chunk_range(chunk_idx, num_groups)
            # print(f"start layer and end layer is {start_layer}, {end_layer}")
            curr_state = (state_bw, state_ctime, start_layer, original_prev_assignment)
            actions_in_chunk = []

            for l in range(start_layer, end_layer + 1):
                action_2d = self._get_policy_action(curr_state)
                # print(f"action for layer {l} is {action_2d}")
                hashable = self._action_to_hashable(action_2d)
                actions_in_chunk.append((action_2d, hashable))
                curr_state = (state_bw, state_ctime, l + 1, action_2d)

            # Majority vote on hashable tuples
            if actions_in_chunk:
                counter = Counter(h for _, h in actions_in_chunk)
                max_count = max(counter.values())
                majority_hashables = [h for h, cnt in counter.items() if cnt == max_count]

                if len(majority_hashables) == 1:
                    target_hash = majority_hashables[0]
                    chosen_action = next(act for act, h in actions_in_chunk if h == target_hash)
                else:
                    chosen_action = actions_in_chunk[0][0]   # tie → first action
            else:
                chosen_action = self._get_default_action(layer)

            self.group_range_assignments[cache_key] = chosen_action
            assignment_vector = chosen_action[0][1]
            actions = []
            current_layer = self.simulator.get_current_layer_index()
            for _ in range(self.profiling.get_num_nodes(current_layer)):
                actions.append([current_layer, (1 if assignment_vector > 0 else 0)])
            return np.array(actions), count

        # ========== NON‑GROUPED CASE (original) ==========
        actions = self._get_possible_actions()
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

        return chosen_action, count

    # -------------------------------------------------------------------------
    # Policy action helper (used by both paths)
    # -------------------------------------------------------------------------
    def _get_policy_action(self, state):
        actions = self._get_possible_actions()
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
    def step(self, current_state, num_groups=None, count=0):
        start_time = time.time()
        action, cached_count = self.choose_action(current_state, num_groups=num_groups, count=count, layer_index= self.simulator.get_current_layer_index())
        # print(f"action for label {current_state[2]} is {action}")
        overhead_time_per_step = time.time() - start_time

        layer = self.simulator.get_current_layer_index()
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

        return action, reward, latency_s, next_state, done, overhead_time_per_step, cached_count
    
    #------------------------------------------
    # Expected weighted values for next-state critic 
    #-------------------------------------------
    def _soft_robust_expected_next_value(self,next_state,done,):
        """
        Calculate the weighted next-state critic value
        under the bandwidth uncertainty models.
        """

        if done:
            return 0.0

        cloud_contention = float(
            next_state[1]
        )

        segment_id = next_state[2]
        

        prev_action = next_state[3]

        expected_next_value = 0.0

        for model in self.uncertainty_models:
            # The empirical values are in Mbps.
            # RL bandwidth state is in MB/s.
            scenario_bandwidth = (
                float(model["bandwidth_mbps"])
                / 8.0
            )

            scenario_state = (
                scenario_bandwidth,
                cloud_contention,
                segment_id,
                prev_action,
            )

            scenario_key = self._state_to_key(
                scenario_state
            )

            scenario_value = float(
                self.value_table.get(
                    scenario_key,
                    0.0,
                )
            )

            expected_next_value += (
                float(model["weight"])
                * scenario_value
            )

        return float(expected_next_value)

    # -------------------------------------------------------------------------
    # Update (Policy Gradient)
    # -------------------------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        state_key = self._state_to_key(state)
        action_key = self._action_to_key(action)
        next_state_key = self._state_to_key(next_state)

        V_current = self.value_table.get(state_key, 0.0)
        # if done:
        #     target = reward
        #     # V_next = 0.0
        # else:
        #     V_next = self.value_table.get(next_state_key, 0.0)
        #     target = reward + self.gamma * V_next

        #soft-robust average reward update 
        self.average_reward += (self.average_reward_lr)*(reward - self.average_reward)
        # Weighted value under poor, nominal and good
        # bandwidth conditions.
        expected_next_value = (
            self._soft_robust_expected_next_value(
                next_state=next_state,
                done=done,
            )
        )
        #soft robust td error        
        td_error = (reward-self.average_reward + expected_next_value - V_current)
        td_error = np.clip(td_error, -5000.0, 5000.0)
        # td_error = np.clip(td_error, -10.0, 10.0)
        # Update critic
        self.value_table[state_key] = V_current + self.alpha_critic * td_error

        # Update actor
        # layer = self.simulator.get_current_layer_index()
        actions = self._get_possible_actions()
        # td_error = 0.0
        if (actions != []):
            prefs = []
            for a in actions:
                akey = self._action_to_key(a)
                prefs.append(self.policy_table.get((state_key, akey), 0.0))
            prefs = np.array(prefs)

            scaled = prefs / max(self.temperature, 1e-8)
            scaled -= np.max(scaled)
            probs = np.exp(scaled) / np.sum(np.exp(scaled))
            # Find index of chosen action
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
        self.simulator.reset_layer_count()
        self.group_range_assignments.clear()


    def end_episode(self):
        total_latency = self.current_episode_latency
        total_reward = self.current_episode_reward
        self.total_episodes += 1
        if self.is_test:
            return total_latency, total_reward

        if total_latency < self.best_episode_latency:
            self.best_episode_latency = total_latency
            self.episodes_since_improvement = 0
            self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)
        else:
            self.episodes_since_improvement += 1
            if self.episodes_since_improvement >= self.stagnant_limit:
                self.temperature = self.temperature_boost
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
            pickle.dump((self.policy_table, self.value_table,self.average_reward), f)
        print(f"💾 Agent saved to {file}")

    def load(self, file="a2c_tables.pkl"):
        if os.path.exists(file):
            with open(file, "rb") as f:
                self.policy_table, self.value_table,self.average_reward = pickle.load(f)
            print(f"✅ Agent loaded from {file}")
        else:
            print(f"⚠️ File {file} not found. Starting from scratch.")