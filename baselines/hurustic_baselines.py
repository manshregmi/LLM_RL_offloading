import random
import numpy as np
import matplotlib.pyplot as plt
# from simulator.simulator import CloudEdgeSimulator  # OLD
from profiling.profiling_class import ProfilingData
# Import your new simulator
from simulator.simulator import CloudEdgeSimulator  # Update import path as needed


def get_random_action(profiling_data: ProfilingData, layer_idx: int):
    """
    Generate a random action for a single layer in the format:
    [[layer_idx, decision], ...]
    """
    num_nodes = profiling_data.get_num_nodes(layer_idx)

    a = np.zeros((num_nodes, 2), dtype=int)
    a[:, 0] = layer_idx

    if layer_idx == (len(profiling_data.layers) - 1):
        a[:, 1] = 0  # last layer forced to edge
    else:
        for node in range(num_nodes):
            a[node, 1] = random.choice([0, 1])

    return a


def get_all_edge_action(profiling_data: ProfilingData, layer_idx: int):
    """All nodes on edge for a single layer."""
    num_nodes = profiling_data.get_num_nodes(layer_idx)
    a = np.zeros((num_nodes, 2), dtype=int)
    a[:, 0] = layer_idx
    a[:, 1] = 0  # all edge
    return a


def get_all_cloud_action(profiling_data: ProfilingData, layer_idx: int):
    """All nodes on cloud for a single layer (last layer forced to edge)."""
    num_nodes = profiling_data.get_num_nodes(layer_idx)
    a = np.zeros((num_nodes, 2), dtype=int)
    a[:, 0] = layer_idx
    if layer_idx == (len(profiling_data.layers) - 1):
        a[:, 1] = 0  # last layer forced to edge
    else:
        a[:, 1] = 1  # all cloud
    return a


def run_scheduler(profiling_data: ProfilingData, episodes=10, max_steps=5000, scheduler_type='random'):
    """
    Run offloading scheduler benchmark over multiple episodes.
    Collect per-episode latency and reward.
    
    Args:
        profiling_data: ProfilingData object
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode (not strictly enforced due to layer jumps)
        scheduler_type: 'random', 'all_edge', or 'all_cloud'
    
    Returns:
        tuple: (avg_latency_ms, avg_reward, deadline_miss_count)
               Note: deadline is now ignored since new simulator focuses on latency
    """
    episode_latencies = []  # in ms
    episode_rewards = []
    deadline_miss_count = 0  # kept for compatibility, but always 0 in new simulator
    
    # Initialize simulator
    simulator = CloudEdgeSimulator(profiling_data)
    
    # Reset episode time for each run
    simulator.reset_episode_time()
    initial_bandwidth = simulator.get_current_bandwidth()
    initial_cloud_contention = 0.0  # No pending cloud work initially
    initial_layer = 0

    for ep in range(episodes):
        # Reset for new episode
        
        # Initial state: [bandwidth, cloud_contention, layer, previous_action_pattern]
        initial_prev_action = None
        state = (initial_bandwidth, initial_cloud_contention, initial_layer, initial_prev_action)
        
        total_latency_s = 0.0
        total_reward = 0.0
        step_count = 0
        
        for step in range(max_steps):
            current_layer = state[2]
            
            # Get action based on scheduler type
            if scheduler_type == 'random':
                action = get_random_action(profiling_data, current_layer)
            elif scheduler_type == 'all_edge':
                action = get_all_edge_action(profiling_data, current_layer)
            elif scheduler_type == 'all_cloud':
                action = get_all_cloud_action(profiling_data, current_layer)
            else:
                raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
            
            # Check if all nodes are going to cloud (for special handling)
            is_all_cloud = (scheduler_type == 'all_cloud') or (np.all(action[:, 1] == 1))
            
            # Get cloud waiting time for next state
            next_cloud_pending = simulator.get_next_state_cloud_waiting_time(
                next_layer=current_layer,  # Wait time for current layer's processing
                current_action=action,
                isAllCloud=is_all_cloud
            )
            
            # Compute latency for this step
            latency_s = simulator.compute_latency(
                current_state=state,
                current_action=action,
                cloud_pending_ms=next_cloud_pending
            )
            
            # Calculate reward
            reward = simulator.calculate_latency_reward(latency_s, scale_factor=10.0)
            
            # Get next state
            next_state, terminal = simulator.get_next_state(
                current_state=state,
                action=action,
                new_cloud_pending=next_cloud_pending
            )

            initial_bandwidth = next_state[0]  # Update bandwidth for next episode start
            initial_cloud_contention = next_state[1]  # Update cloud contention for next episode

            
            # Accumulate totals
            total_latency_s += latency_s
            total_reward += reward
            
            # Update state
            state = next_state
            step_count += 1
            
            # Check if episode is complete
            if terminal:
                break
        
        # Convert latency to ms for reporting
        total_latency_ms = total_latency_s * 1000
        
        episode_latencies.append(total_latency_ms)
        episode_rewards.append(total_reward)
        
        # Optional: Print progress
        print(f"Episode {ep+1}/{episodes}: Latency={total_latency_ms:.2f}ms, "
              f"Reward={total_reward:.2f}, Steps={step_count}")
    
    # Calculate averages
    avg_latency_ms = np.mean(episode_latencies)
    avg_reward = np.mean(episode_rewards)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Scheduler: {scheduler_type.upper()}")
    print(f"Average Latency: {avg_latency_ms:.2f} ms")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Total Episodes: {episodes}")
    print(f"{'='*50}")
    
    # Optional: Plot results
    # plt.figure(figsize=(12, 4))
    # 
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, episodes+1), episode_latencies, marker='o')
    # plt.xlabel('Episode')
    # plt.ylabel('Total Latency (ms)')
    # plt.title(f'{scheduler_type.upper()} Scheduler: Latency per Episode\nAvg: {avg_latency_ms:.2f} ms')
    # plt.grid(True)
    # 
    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, episodes+1), episode_rewards, marker='s')
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.title(f'{scheduler_type.upper()} Scheduler: Reward per Episode\nAvg: {avg_reward:.2f}')
    # plt.grid(True)
    # 
    # plt.tight_layout()
    # plt.show()
    
    return avg_latency_ms, avg_reward, deadline_miss_count


# Backward compatibility functions (if needed)
def run_random_scheduler(profiling_data: ProfilingData, episodes=10, max_steps=20, is_random=True, is_all_cloud=False):
    """
    Backward compatibility wrapper for old function signature.
    """
    if is_random:
        scheduler_type = 'random'
    elif is_all_cloud:
        scheduler_type = 'all_cloud'
    else:
        scheduler_type = 'all_edge'
    
    avg_latency_ms, avg_reward, deadline_miss_count = run_scheduler(
        profiling_data=profiling_data,
        episodes=episodes,
        max_steps=max_steps,
        scheduler_type=scheduler_type
    )
    
    # Return format compatible with old function (energy, completion_time, deadline_miss_count)
    # Note: Energy is always 0 in new simulator
    return 0.0, avg_latency_ms, deadline_miss_count


# Example usage
if __name__ == "__main__":
    # Load profiling data
    # profiling_data = ProfilingData(...)  # Load your profiling data
    
    # Run different schedulers
    # avg_latency, avg_reward, misses = run_scheduler(profiling_data, episodes=5, scheduler_type='random')
    # avg_latency, avg_reward, misses = run_scheduler(profiling_data, episodes=5, scheduler_type='all_edge')
    # avg_latency, avg_reward, misses = run_scheduler(profiling_data, episodes=5, scheduler_type='all_cloud')
    pass