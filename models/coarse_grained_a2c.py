import time
import numpy as np
import pickle
import os
from simulator.simulator import CloudEdgeSimulator
from profiling.profiling_class import ProfilingData

class OneShotTabularA2C:
    """
    Tabular one-shot Actor-Critic for latency minimisation.
    - Policy: P(cloud) for each node, stored in a 3D table (bw, cloud, node).
    - Value: expected total latency (negative reward) given state.
    - Action: sample all nodes independently -> full assignment vector.
    - Update: after episode, REINFORCE with baseline (TD error).
    """

    def __init__(self, profiling_data: ProfilingData,
                 alpha_actor=0.02, alpha_critic=0.05, gamma=0.95,
                 reward_scale=10.0, total_pipelines=1):
        self.profiling = profiling_data
        self.gamma = gamma
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.reward_scale = reward_scale   # not directly used, but kept for compatibility

        # Discretisation
        self.bw_bins = np.linspace(1, 15, 15)
        self.cloud_bins = np.linspace(0, 100, 20)   # cloud contention in ms
        self.num_bw = len(self.bw_bins)
        self.num_cloud = len(self.cloud_bins)

        # Total number of nodes across all layers
        self.total_nodes = self._count_total_nodes()
        print(f"OneShotTabularA2C: total_nodes = {self.total_nodes}")

        # Policy table: probability of cloud for each (bw_idx, cloud_idx, node)
        self.policy = np.full((self.num_bw, self.num_cloud, self.total_nodes), 0.5, dtype=np.float32)
        # Value table: estimated total latency (ms) for each state
        self.V = np.zeros((self.num_bw, self.num_cloud), dtype=np.float32)

        # Simulator (reused across episodes)
        self.simulator = CloudEdgeSimulator(profiling_data, total_pipeline=total_pipelines)

        # Episode tracking
        self.current_episode_latency = 0.0
        self.total_episodes = 0

        # Temperature / exploration (optional)
        self.temperature = 1.0
        self.temperature_min = 0.01
        self.temperature_decay = 0.9995
        self.temperature_boost = 1.5
        self.best_episode_latency = float('inf')
        self.episodes_since_improvement = 0
        self.stagnant_limit = 5000

    def _count_total_nodes(self):
        total = 0
        for layer in self.profiling.layers:
            total += len(layer)
        return total

    def _discretize(self, value, bins):
        idx = np.digitize([value], bins, right=True)[0] - 1
        return max(0, min(idx, len(bins)-1))

    def _state_to_indices(self, bandwidth, cloud_contention):
        bw_idx = self._discretize(bandwidth, self.bw_bins)
        cloud_idx = self._discretize(cloud_contention, self.cloud_bins)
        return bw_idx, cloud_idx

    def _get_action_probabilities(self, bw_idx, cloud_idx):
        """Return probability of cloud for each node (1D array)."""
        probs = self.policy[bw_idx, cloud_idx, :].copy()
        # Apply temperature: make probabilities more uniform if temperature>1
        if self.temperature > 1.0:
            # Convert to logits, scale, then sigmoid
            logits = np.log(probs / (1 - probs + 1e-8))
            logits = logits / self.temperature
            probs = 1.0 / (1.0 + np.exp(-logits))
        return probs

    def choose_action(self, bandwidth, cloud_contention):
        """Sample a full assignment vector (binary per node)."""
        bw_idx, cloud_idx = self._state_to_indices(bandwidth, cloud_contention)
        probs = self._get_action_probabilities(bw_idx, cloud_idx)
        # Sample independent Bernoulli
        action = (np.random.random(self.total_nodes) < probs).astype(int)   # 0=edge, 1=cloud
        # Compute log probability of the action
        log_prob = np.sum(
            action * np.log(probs + 1e-8) + (1-action) * np.log(1-probs + 1e-8)
        )
        return action, log_prob, (bw_idx, cloud_idx), probs

    def _build_action_matrices(self, flat_action):
        """Convert flat binary array into per‑layer action matrices for the simulator."""
        action_matrices = []
        idx = 0
        for layer_idx, layer_nodes in enumerate(self.profiling.layers):
            num_nodes = len(layer_nodes)
            # action matrix: (nodes, 2) where column0 = layer index, column1 = assignment
            mat = np.zeros((num_nodes, 2), dtype=int)
            mat[:, 0] = layer_idx
            mat[:, 1] = flat_action[idx:idx+num_nodes]
            action_matrices.append(mat)
            idx += num_nodes
        return action_matrices

    def run_episode(self):
        """Run one episode with a single one‑shot action."""
        # Get initial state
        bandwidth = self.simulator.get_current_bandwidth()
        cloud_contention = 0.0   # no pending cloud tasks at start
        state = (bandwidth, cloud_contention)

        # Choose global assignment
        start_time = time.time()
        flat_action, log_prob, (bw_idx, cloud_idx), probs = self.choose_action(bandwidth, cloud_contention)
        action_matrices = self._build_action_matrices(flat_action)
        overhead_time = (time.time() - start_time) * 1000.0

        # Simulate the whole pipeline using the fixed plan
        total_latency_s = 0.0
        current_cloud_pending = cloud_contention
        # We'll track done flag but we know it's false until last layer
        current_state = (bandwidth, cloud_contention, 0, None)   # layer=0, no prev assignment

        for layer_idx, action in enumerate(action_matrices):
            # Get cloud waiting time for this action
            cloud_waiting = self.simulator.get_next_state_cloud_waiting_time(
                next_layer=layer_idx, current_action=action, isAllCloud=False
            )
            # Compute latency for this layer
            latency_s = self.simulator.compute_latency(
                current_state=current_state,
                current_action=action,
                cloud_pending_ms=cloud_waiting
            )
            total_latency_s += latency_s
            # Move to next state (we don't actually need next_state except for final)
            next_state, done = self.simulator.get_next_state(
                current_state=current_state,
                action=action,
                new_cloud_pending=cloud_waiting
            )
            total_generated_tokens = 0
            if (done):
                total_generated_tokens = self.simulator.get_total_generated_tokens(done)
                cloud_contention = cloud_waiting
                break
            layer_idx = next_state[2]
            current_state = next_state
            cloud_contention = cloud_waiting

        total_latency_ms = total_latency_s * 1000.0
        reward = -total_latency_ms   # minimise latency
        print(f"Episode {self.total_episodes+1}: latency = {total_latency_ms:.2f} ms, reward = {reward:.2f}")

        # Compute advantage (TD error) - no next state because episode ends here
        value = self.V[bw_idx, cloud_idx]
        advantage = reward - value

        # Update critic (value table)
        self.V[bw_idx, cloud_idx] += self.alpha_critic * advantage

        # Update actor (policy table) – REINFORCE with baseline
        # For each node i, gradient = advantage * (action_i - prob_i)
        # We update probabilities directly (not logits) using a small step.
        # This is a standard tabular update for Bernoulli policies.
        grad = advantage * (flat_action - probs)
        new_probs = self.policy[bw_idx, cloud_idx, :] + self.alpha_actor * grad
        # Clip to avoid numerical issues
        new_probs = np.clip(new_probs, 0.01, 0.99)
        self.policy[bw_idx, cloud_idx, :] = new_probs

        # Update temperature (optional, for exploration)
        self.total_episodes += 1
        if total_latency_ms < self.best_episode_latency:
            self.best_episode_latency = total_latency_ms
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

        return total_latency_ms, reward , overhead_time, total_generated_tokens

    def start_episode(self):
        """Reset simulator for a new episode."""
        self.simulator.reset_episode_time()

    def end_episode(self):
        """For compatibility with your runner; returns current episode latency."""
        return self.current_episode_latency, 0.0   # dummy reward

    def save(self, file="one_shot_a2c_tabular.pkl"):
        """Save policy and value tables."""
        with open(file, "wb") as f:
            pickle.dump((self.policy, self.V, self.temperature, self.total_episodes), f)
        print(f"💾 OneShotTabularA2C saved to {file}")

    def load(self, file="one_shot_a2c_tabular.pkl"):
        """Load previously saved tables."""
        if os.path.exists(file):
            with open(file, "rb") as f:
                data = pickle.load(f)
                self.policy, self.V, self.temperature, self.total_episodes = data
            print(f"✅ OneShotTabularA2C loaded from {file}")
        else:
            print(f"⚠️ File {file} not found. Starting from scratch.")