import numpy as np
from profiling.profiling_class import ProfilingData
from simulator.simulator import CloudEdgeSimulator



def edgeshard_dp(profiling_data: ProfilingData, simulator: CloudEdgeSimulator,
                 bandwidth_mbps: float, cloud_pending: float = 0.0) -> list:
    """
    Run EdgeShard DP using the given simulator's compute_latency for costs.
    Temporarily resets cumulative_time to avoid side effects.
    Returns list devices[layer] = 0 (edge) or 1 (cloud).
    """
    layers = profiling_data.layers
    num_layers = len(layers)
    INF = float('inf')
    dp = [[INF, INF] for _ in range(num_layers)]
    choice = [[-1, -1] for _ in range(num_layers)]

    # Helper: cost of a single layer given previous action pattern
    def layer_cost(layer_idx, current_dev, prev_pattern):
        # Build action matrix (all nodes on current_dev)
        num_nodes = profiling_data.get_num_nodes(layer_idx)
        action = np.zeros((num_nodes, 2), dtype=int)
        action[:, 0] = layer_idx
        action[:, 1] = current_dev
        # State for this layer
        state = (bandwidth_mbps, cloud_pending, layer_idx, prev_pattern)
        # Save simulator's time, compute, then restore
        saved_time = simulator.cumulative_time_seconds
        simulator.cumulative_time_seconds = 0.0
        latency_s = simulator.compute_latency(state, action, cloud_pending)
        simulator.cumulative_time_seconds = saved_time
        return latency_s * 1000.0   # convert to ms

    # First layer: must be on edge, no previous action
    dp[0][0] = layer_cost(0, 0, None)
    # dp[0][1] stays INF

    # Fill DP
    for i in range(1, num_layers):
        for j in (0, 1):          # current device
            if i == num_layers - 1 and j != 0:   # last layer forced to edge
                continue
            for k in (0, 1):      # previous device
                if dp[i-1][k] == INF:
                    continue
                # Build previous action pattern for layer i-1 (all nodes on device k)
                prev_num_nodes = profiling_data.get_num_nodes(i-1)
                prev_pattern = tuple([k] * prev_num_nodes)
                cost = layer_cost(i, j, prev_pattern)
                cand = dp[i-1][k] + cost
                if cand < dp[i][j]:
                    dp[i][j] = cand
                    choice[i][j] = k

    # Backtrack
    devices = [0] * num_layers
    devices[-1] = 0
    for i in range(num_layers-1, 0, -1):
        devices[i-1] = choice[i][devices[i]]
    return devices




def run_edgeshard_scheduler(profiling_data: ProfilingData, episodes=10, max_steps=5000):
    """
    Run only the EdgeShard baseline over multiple episodes.
    Returns per‑episode total latencies (ms), total rewards, and the action plan.
    """
    episode_latencies = []
    episode_rewards = []
    all_action_plans = []   # store the list of action matrices for each episode

    simulator = CloudEdgeSimulator(profiling_data)

    for ep in range(episodes):
        simulator.reset_episode_time()
        initial_bandwidth = simulator.get_current_bandwidth()
        initial_cloud_contention = 0.0
        initial_layer = 0
        initial_prev_action = None

        # ---- EdgeShard: compute static device assignment ----
        devices = edgeshard_dp(profiling_data, simulator,
                               bandwidth_mbps=initial_bandwidth,
                               cloud_pending=initial_cloud_contention)
        # Build action matrices for all layers
        actions_list = []
        num_layers = len(profiling_data.layers)
        for layer_idx in range(num_layers):
            num_nodes = profiling_data.get_num_nodes(layer_idx)
            action_mat = np.zeros((num_nodes, 2), dtype=int)
            action_mat[:, 0] = layer_idx
            action_mat[:, 1] = devices[layer_idx]
            actions_list.append(action_mat)
        all_action_plans.append(actions_list)
        # ------------------------------------------------

        state = (initial_bandwidth, initial_cloud_contention, initial_layer, initial_prev_action)
        total_latency_s = 0.0
        total_reward = 0.0
        step_count = 0

        for step in range(max_steps):
            current_layer = state[2]
            if current_layer >= num_layers:
                break   # safety

            action = actions_list[current_layer]

            # Determine if this layer's action is all‑cloud (for cloud waiting time)
            is_all_cloud = False

            # Get cloud waiting time for the next state
            next_cloud_pending = simulator.get_next_state_cloud_waiting_time(
                next_layer=current_layer,
                current_action=action,
                isAllCloud=is_all_cloud
            )

            latency_s = simulator.compute_latency(state, action, next_cloud_pending)
            reward = simulator.calculate_latency_reward(latency_s, scale_factor=10.0)

            next_state, terminal = simulator.get_next_state(
                current_state=state,
                action=action,
                new_cloud_pending=next_cloud_pending
            )

            total_latency_s += latency_s
            total_reward += reward
            state = next_state
            step_count += 1

            if terminal:
                break

        total_latency_ms = total_latency_s * 1000
        episode_latencies.append(total_latency_ms)
        episode_rewards.append(total_reward)

        print(f"Episode {ep+1}/{episodes}: Latency={total_latency_ms:.2f}ms, "
              f"Reward={total_reward:.2f}, Steps={step_count}")

    avg_latency_ms = np.mean(episode_latencies)
    avg_reward = np.mean(episode_rewards)

    print(f"\n{'='*50}")
    print(f"EDGESHARD SCHEDULER")
    print(f"Average Latency: {avg_latency_ms:.2f} ms")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Total Episodes: {episodes}")
    print(f"{'='*50}")

    # Return the actions for the first episode as an example of the format
    # (or return all_action_plans if needed)
    return avg_latency_ms, avg_reward, all_action_plans[0]

# Example usage
if __name__ == "__main__":
    from profiling.initialize_agx_profiling import get_LLM_profiling_data

    profiling_data = get_LLM_profiling_data()
    avg_lat, avg_rew, actions = run_edgeshard_scheduler(profiling_data, episodes=1000)
    print("latency is", avg_lat)