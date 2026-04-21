import asyncio
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from Grouping_RL import GroupingRL
from profiling.profiling_class import ProfilingData
from models.doubleQ import TabularDoubleQLearningAgent   # adjust import as needed


def create_initial_state(simulator):
    """
    Create initial state for an episode.
    State: [bandwidth, cloud_contention, layer, previous_assignment]
    """
    bandwidth = simulator.get_current_bandwidth()
    cloud_contention = 0.0          # no pending cloud tasks at start
    layer = 0
    previous_assignment = None
    return (bandwidth, cloud_contention, layer, previous_assignment)


def train_double_q_agent(profiling_data: ProfilingData,
                         episodes=50000,
                         is_test=False,
                         verbose=True):
    """Main training loop for Double Q‑learning."""

    # Create agent
    agent = TabularDoubleQLearningAgent(
        profiling_data=profiling_data,
        is_test=is_test,
        alpha=0.1,
        gamma=0.95,
        reward_scale=10.0,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    grouping_RL_agent = GroupingRL()

    NUM_EPISODES = episodes
    PRINT_INTERVAL = 10

    # Tracking
    episode_latencies = []
    episode_rewards = []
    best_latency = float('inf')
    best_episode = 0

    print("=" * 80)
    print("TRAINING: Double Q‑Learning for Pure Latency Minimization")
    print("=" * 80)
    print(f"Total episodes: {NUM_EPISODES}")
    print(f"Learning rate: α = {agent.alpha}")
    print(f"Discount factor: γ = {agent.gamma}")
    print(f"Reward scale: {agent.reward_scale}")
    print(f"Epsilon decay: {agent.epsilon_decay} → min {agent.epsilon_min}")
    print("=" * 80)

    # Load previous models if they exist
    agent.load()
    grouping_RL_agent.load()

    episode_overhead_time = []
    average_step_overhead_times = []

    for episode in range(NUM_EPISODES):
        # Initial state
        bandwidth = agent.simulator.get_current_bandwidth()
        cloud_contention = 0.0
        state = (bandwidth, cloud_contention, 0, None)

        agent.start_episode()
        done = False
        step_count = 0
        td_errors = []
        step_overhead_times = []

        # Keep track of cloud contention for grouping agent
        last_pipeline_contention = [state[1]]

        # Decide number of groups for this episode
        number_of_groups = grouping_RL_agent.train(
            bandwidth,
            np.mean(last_pipeline_contention) if last_pipeline_contention else 0.0
        )

        while not done:
            start_step = time.perf_counter()
            action, reward, latency_s, next_state, done = agent.step(
                state, num_groups=number_of_groups
            )
            step_time = time.perf_counter() - start_step
            step_overhead_times.append(step_time)

            # Update tracking
            last_pipeline_contention.append(next_state[1])

            td_error = agent.update(state, action, reward, next_state, done)
            td_errors.append(td_error)

            state = next_state
            step_count += 1

        episode_overhead_time.append(np.sum(step_overhead_times))
        average_step_overhead_times.append(np.mean(step_overhead_times))

        # Update grouping agent with episode reward
        total_reward_ep = agent.current_episode_reward
        asyncio.run(grouping_RL_agent.get_reward(total_reward_ep))   # async handled inside
        if episode % 100 == 0:
            grouping_RL_agent.save()

        # End episode and get total latency
        total_latency_ms, total_reward = agent.end_episode()
        episode_latencies.append(total_latency_ms)
        episode_rewards.append(total_reward)

        if total_latency_ms < best_latency:
            best_latency = total_latency_ms
            best_episode = episode

        # Logging
        if episode % PRINT_INTERVAL == 0:
            avg_latency = np.mean(episode_latencies[-PRINT_INTERVAL:])
            avg_reward = np.mean(episode_rewards[-PRINT_INTERVAL:])
            avg_td = np.mean(td_errors) if td_errors else 0
            print(f"Episode {episode:4d} | "
                  f"Latency: {total_latency_ms:6.1f}ms | "
                  f"Avg Lat: {avg_latency:6.1f}ms | "
                  f"Best: {best_latency:6.1f}ms | "
                  f"Reward: {total_reward:7.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"TD: {avg_td:6.2f} | "
                  f"Steps: {step_count}")

    # Final save
    agent.save()

    print("\n" + "=" * 80)
    print("Overhead statistics:")
    print(f"Mean overhead per episode: {np.mean(episode_overhead_time)*1000:.4f} ms")
    print(f"Std overhead per episode: {np.std(episode_overhead_time)*1000:.4f} ms")
    print(f"Mean overhead per step: {np.mean(average_step_overhead_times)*1000:.4f} ms")
    print("=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best latency: {best_latency:.2f} ms at episode {best_episode}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print("=" * 80)

    return agent, episode_latencies, episode_rewards


def evaluate_double_q_agent(agent, num_episodes=100):
    """
    Evaluate trained Double Q‑learning agent and collect assignment counts.

    Returns:
        latencies_ms: list of total latencies per episode
        rewards: list of total rewards per episode
        assignment_counts: dict (layer, node) -> {'edge': count, 'cloud': count}
    """
    print("\n" + "=" * 80)
    print("EVALUATION: Double Q‑Learning (Pure Latency Minimization)")
    print("=" * 80)

    agent.is_test = True
    latencies_ms = []
    rewards = []
    assignment_counts = defaultdict(lambda: {'edge': 0, 'cloud': 0})

    for episode in range(num_episodes):
        agent.start_episode()
        state = create_initial_state(agent.simulator)
        done = False

        while not done:
            # Evaluation uses greedy actions; no grouping needed (or use None)
            action, reward, latency_s, next_state, done = agent.step(state, num_groups=None)

            # Record assignment for each node in the action
            layer = int(state[2])
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

    mean_lat = np.mean(latencies_ms)
    std_lat = np.std(latencies_ms)
    min_lat = np.min(latencies_ms)
    max_lat = np.max(latencies_ms)
    p95_lat = np.percentile(latencies_ms, 95)

    print(f"Episodes: {num_episodes}")
    print(f"Mean latency: {mean_lat:.2f} ms")
    print(f"Std latency: {std_lat:.2f} ms")
    print(f"Min latency: {min_lat:.2f} ms")
    print(f"Max latency: {max_lat:.2f} ms")
    print(f"95th percentile: {p95_lat:.2f} ms")
    print(f"Mean reward: {np.mean(rewards):.2f}")
    print("=" * 80)

    return latencies_ms, rewards, assignment_counts


def aggregate_assignments_by_segment(assignment_counts):
    """
    Group nodes into model segments and compute edge/cloud percentages.
    (Same as in A2C runner)
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
        if layer == 0 and node in (0, 1):
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

    for seg in segments:
        total = segments[seg]['edge'] + segments[seg]['cloud']
        if total > 0:
            segments[seg]['edge_pct'] = 100 * segments[seg]['edge'] / total
            segments[seg]['cloud_pct'] = 100 * segments[seg]['cloud'] / total
        else:
            segments[seg]['edge_pct'] = segments[seg]['cloud_pct'] = 0
    return segments


def plot_assignment_percentages(segments):
    """Bar chart of edge vs cloud percentages per segment."""
    labels = list(segments.keys())
    edge_pcts = [segments[seg]['edge_pct'] for seg in labels]
    cloud_pcts = [segments[seg]['cloud_pct'] for seg in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, edge_pcts, width, label='Edge', color='#4477AA')
    bars2 = ax.bar(x + width/2, cloud_pcts, width, label='Cloud', color='#66CCEE')

    ax.set_ylabel('Percentage of assignments (%)')
    ax.set_title('Assignment vector per model segment (Double Q‑Learning)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# ============================================================================
# Example usage (commented out)
# ============================================================================
# if __name__ == "__main__":
#     from profiling.llm_profiling import get_LLM_profiling_data
#     profiling_data = get_LLM_profiling_data()
#
#     # Train
#     trained_agent, train_latencies, train_rewards = train_double_q_agent(profiling_data, episodes=50000)
#
#     # Evaluate
#     eval_latencies, eval_rewards, assignment_counts = evaluate_double_q_agent(trained_agent, num_episodes=500)
#
#     # Save results
#     np.savez("double_q_results.npz",
#              train_latencies=train_latencies,
#              train_rewards=train_rewards,
#              eval_latencies=eval_latencies,
#              eval_rewards=eval_rewards)
#
#     # Segment analysis
#     segments = aggregate_assignments_by_segment(assignment_counts)
#     for seg, data in segments.items():
#         print(f"{seg:20s}: Edge {data['edge_pct']:.1f}% | Cloud {data['cloud_pct']:.1f}%")
#     plot_assignment_percentages(segments)