import asyncio
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from Grouping_RL import GroupingRL
from profiling.profiling_class import ProfilingData
from models.a2c_model import TabularActorCriticAgent
# Adjust import for your profiling data function as needed
# from profiling.llm_profiling import get_LLM_profiling_data

def create_initial_state(simulator):
    """
    Create initial state for an episode.
    
    State format: [bandwidth, cloud_contention, layer, previous_assignment]
    """
    bandwidth = simulator.get_current_bandwidth()
    cloud_contention = 0.0  # No pending cloud tasks at start
    layer = 0  # Start at first layer
    previous_assignment = None  # No previous layer
    
    return (bandwidth, cloud_contention, layer, previous_assignment)

def train_a2c_agent(profiling_data: ProfilingData, episodes=50000, is_test=False, verbose=True): 
    """Main training loop."""
    
    # Create agent
    agent = TabularActorCriticAgent(
        profiling_data=profiling_data,
        is_test=is_test,
        alpha_actor=0.02,
        alpha_critic=0.05,
        gamma=0.95,
        reward_scale=10.0,  # Scale reward magnitude
    )

    grouping_RL_agent = GroupingRL()
    
    # Training parameters
    NUM_EPISODES = episodes
    PRINT_INTERVAL = 10
    
    # Tracking
    episode_latencies = []
    episode_rewards = []
    best_latency = float('inf')
    best_episode = 0
    
    print("=" * 80)
    print("TRAINING: Pure Latency Minimization")
    print("=" * 80)
    print("Goal: Minimize total inference latency")
    print("No deadlines, no surplus, no negative counts")
    print("=" * 80)
    print(f"Total episodes: {NUM_EPISODES}")
    print(f"Learning rates: actor={agent.alpha_actor}, critic={agent.alpha_critic}")
    print(f"Discount factor: γ={agent.gamma}")
    print(f"Reward scale: {agent.reward_scale}")
    print("=" * 80)
    state = create_initial_state(agent.simulator)
    bandwidth = state[0]
    cloud_contention = state[1]
    last_pipeline_contention = []
    average_last_pipeline_contention = 0.0
    state = (bandwidth, cloud_contention, 0, None)  # Start at layer 0 with no previous assignment
    agent.load()  # Load existing model if available
    grouping_RL_agent.load()  # Load grouping agent if it has a saved state
    episode_overhead_time = []
    average_step_overhead_times = []
    for episode in range(NUM_EPISODES):
        rewards_ep = 0
        state = (bandwidth, cloud_contention, 0, None)
        # Start episode
        agent.start_episode()
        done = False
        step_count = 0
        td_errors = []
        step_overhead_time = []
        last_pipeline_contention.append(state[1])
        
        number_of_groups = grouping_RL_agent.train(bandwidth, average_last_pipeline_contention)
        # print("number of groups: ", number_of_groups)
        # print("average_last_pipeline_contention: ", average_last_pipeline_contention)

        # Run episode
        action_array = []
        while not done:
            action, reward, latency_s, next_state, done, overhead_time_per_step = agent.step(state, num_groups=number_of_groups)
            step_overhead_time.append(overhead_time_per_step)
            action_array.append(action)
            rewards_ep += reward
            last_pipeline_contention.append(next_state[1])
            bandwidth = next_state[0]
            cloud_contention = next_state[1]
            td_error = agent.update(state, action, reward, next_state, done)
            td_errors.append(td_error)
            state = next_state
            step_count += 1
        episode_overhead_time.append(np.sum(step_overhead_time))
        average_step_overhead_times.append(np.mean(step_overhead_time))

        average_last_pipeline_contention = np.mean(last_pipeline_contention) if last_pipeline_contention else 0.0
        last_pipeline_contention = []
        # asyncio.run(grouping_RL_agent.push_reward(rewards_ep))
        # asyncio.run(grouping_RL_agent.get_reward())
        asyncio.run(grouping_RL_agent.get_reward(rewards_ep))
        if (episode % 100)== 0:
            grouping_RL_agent.save()  # Save grouping agent state after each episode
        # End episode
        total_latency_ms, total_reward = agent.end_episode()
        episode_latencies.append(total_latency_ms)
        episode_rewards.append(total_reward)
        
        # Update best
        if total_latency_ms < best_latency:
            best_latency = total_latency_ms
            best_episode = episode
        
        # Logging
        if episode % PRINT_INTERVAL == 0:
            avg_latency = np.mean(episode_latencies[-PRINT_INTERVAL:])
            avg_reward = np.mean(episode_rewards[-PRINT_INTERVAL:])
            avg_td_error = np.mean(td_errors) if td_errors else 0
            
            print(f"Episode {episode:4d} | "
                  f"Latency: {total_latency_ms:6.1f}ms | "
                  f"Avg Latency: {avg_latency:6.1f}ms | "
                  f"Best: {best_latency:6.1f}ms | "
                  f"Reward: {total_reward:7.1f} | "
                  f"Temp: {agent.temperature:.3f} | "
                  f"TD: {avg_td_error:6.2f} | "
                  f"Steps: {step_count}")
    
    # Final save
    agent.save()

    print("\n" + "=" * 80)

    print("Episode Overhead Times:")
    print(f"Mean overhead time per episode: {np.mean(episode_overhead_time)*1000:.4f}ms")
    print(f"Std overhead time per episode: {np.std(episode_overhead_time)*1000:.4f}ms")
    print(f"Mean overhead time per step: {np.mean(average_step_overhead_times)*1000:.4f}ms")

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best latency: {best_latency:.2f}ms at episode {best_episode}")
    print(f"Final temperature: {agent.temperature:.3f}")
    print("=" * 80)
    
    return agent, episode_latencies, episode_rewards

def evaluate_agent(agent, num_episodes=100):
    """
    Evaluate trained agent and collect assignment counts per (layer, node).
    
    Returns:
        latencies_ms: list of total latencies per episode
        rewards: list of total rewards per episode
        assignment_counts: dict mapping (layer, node) to {'edge': count, 'cloud': count}
    """
    print("\n" + "=" * 80)
    print("EVALUATION: Pure Latency Minimization")
    print("=" * 80)
    
    agent.is_test = True
    latencies_ms = []
    rewards = []
    
    # Count assignments: key = (layer, node) -> {'edge': count, 'cloud': count}
    assignment_counts = defaultdict(lambda: {'edge': 0, 'cloud': 0})
    
    for episode in range(num_episodes):
        agent.start_episode()
        state = create_initial_state(agent.simulator)
        done = False
        
        while not done:
            action, reward, latency_s, next_state, done = agent.step(state)
            
            # Record assignment for each node in this action
            layer = int(state[2])  # current layer index
            # action is a numpy array of shape (nodes, 2); column 1 is assignment (0=edge, 1=cloud)
            for node_idx in range(action.shape[0]):
                assign = action[node_idx, 1]
                key = (layer, node_idx)
                if assign == 0:
                    assignment_counts[key]['edge'] += 1
                else:
                    assignment_counts[key]['cloud'] += 1
            
            state = next_state
        
        latencies_ms.append(agent.current_episode_latency)
        rewards.append(agent.current_episode_reward)
    
    # Statistics (unchanged)
    mean_latency = np.mean(latencies_ms)
    std_latency = np.std(latencies_ms)
    min_latency = np.min(latencies_ms)
    max_latency = np.max(latencies_ms)
    p95_latency = np.percentile(latencies_ms, 95)
    
    print(f"Episodes: {num_episodes}")
    print(f"Mean latency: {mean_latency:.2f}ms")
    print(f"Std latency: {std_latency:.2f}ms")
    print(f"Min latency: {min_latency:.2f}ms")
    print(f"Max latency: {max_latency:.2f}ms")
    print(f"95th percentile: {p95_latency:.2f}ms")
    print(f"Mean reward: {np.mean(rewards):.2f}")
    print("=" * 80)
    
    return latencies_ms, rewards, assignment_counts

def aggregate_assignments_by_segment(assignment_counts):
    """
    Group nodes into model segments and compute edge/cloud percentages.
    
    Model segments:
    - YOLOS: layer 0, nodes 0,1
    - LLAMA (1-99): layers 1-99, node 0
    - LLAMA (100-299): layers 100-299, node 0
    - BART: layers 300-399, node 0
    
    Returns:
        dict: segment name -> {'edge_pct': float, 'cloud_pct': float}
    """
    segments = {
        'YOLOS': {'edge': 0, 'cloud': 0},
        'LLAMA (1-99)': {'edge': 0, 'cloud': 0},
        'LLAMA (100-299)': {'edge': 0, 'cloud': 0},
        'BART': {'edge': 0, 'cloud': 0}
    }
    
    for (layer, node), cnt in assignment_counts.items():
        total = cnt['edge'] + cnt['cloud']
        if total == 0:
            continue
        
        if layer == 0 and node in [0,1]:
            segments['YOLOS']['edge'] += cnt['edge']
            segments['YOLOS']['cloud'] += cnt['cloud']
        elif 1 <= layer <= 99:
            segments['LLAMA (1-99)']['edge'] += cnt['edge']
            segments['LLAMA (1-99)']['cloud'] += cnt['cloud']
        elif 100 <= layer <= 299:
            segments['LLAMA (100-299)']['edge'] += cnt['edge']
            segments['LLAMA (100-299)']['cloud'] += cnt['cloud']
        elif 300 <= layer <= 399:
            segments['BART']['edge'] += cnt['edge']
            segments['BART']['cloud'] += cnt['cloud']
        # else ignore (shouldn't happen)
    
    # Convert to percentages
    for seg in segments:
        total = segments[seg]['edge'] + segments[seg]['cloud']
        if total > 0:
            segments[seg]['edge_pct'] = 100 * segments[seg]['edge'] / total
            segments[seg]['cloud_pct'] = 100 * segments[seg]['cloud'] / total
        else:
            segments[seg]['edge_pct'] = 0
            segments[seg]['cloud_pct'] = 0
    
    return segments

def plot_assignment_percentages(segments):
    """Create a bar chart of edge vs cloud percentages per model segment."""
    labels = list(segments.keys())
    edge_pcts = [segments[seg]['edge_pct'] for seg in labels]
    cloud_pcts = [segments[seg]['cloud_pct'] for seg in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, edge_pcts, width, label='Edge', color='#4477AA')
    bars2 = ax.bar(x + width/2, cloud_pcts, width, label='Cloud', color='#66CCEE')
    
    ax.set_ylabel('Percentage of assignments (%)')
    ax.set_title('Assignment vector per model Segment')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
    # Load your profiling data. Adjust the import/path accordingly.
    # Example: from profiling.llm_profiling import get_LLM_profiling_data
    # profiling_data = get_LLM_profiling_data()
    # For now, we assume profiling_data is already available.
    # If not, you need to create it before calling train_a2c_agent.
    
    # Train
    # trained_agent, latencies, rewards = train_a2c_agent(profiling_data)
    
    # Evaluate and collect assignments
    # eval_latencies, eval_rewards, assignment_counts = evaluate_agent(trained_agent, num_episodes=500)
    
    # Save results (optional)
    # np.savez("pure_latency_results.npz",
    #          training_latencies=latencies,
    #          training_rewards=rewards,
    #          eval_latencies=eval_latencies,
    #          eval_rewards=eval_rewards)
    
    # Compute percentages per segment
    # segments = aggregate_assignments_by_segment(assignment_counts)
    
    # print("\nAssignment percentages per model segment:")
    # for seg, data in segments.items():
    #     print(f"{seg:20s}: Edge {data['edge_pct']:.1f}% | Cloud {data['cloud_pct']:.1f}%")
    
    # Plot the graph
    # plot_assignment_percentages(segments)
    
    # For demonstration, uncomment the lines above and provide your profiling_data.