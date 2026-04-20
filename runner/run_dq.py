import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from profiling.profiling_class import ProfilingData
from models.doubleQ import TabularDoubleQLearningAgent   # adjust import to your actual file

def create_initial_state(simulator):
    """
    Create initial state for an episode.
    
    State format: [bandwidth, cloud_contention, layer, previous_assignment]
    """
    bandwidth = simulator.get_current_bandwidth()
    cloud_contention = 0.0          # No pending cloud tasks at start
    layer = 0                       # Start at first layer
    previous_assignment = None      # No previous layer
    
    return (bandwidth, cloud_contention, layer, previous_assignment)


def train_double_q_agent(profiling_data: ProfilingData, episodes=50000, is_test=False, verbose=True):
    """Main training loop for Double Q‑learning."""
    
    # Create agent
    agent = TabularDoubleQLearningAgent(
        profiling_data=profiling_data,
        is_test=is_test,
        alpha=0.1,           # Learning rate
        gamma=0.95,          # Discount factor
        reward_scale=10.0,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    )
    
    NUM_EPISODES = episodes
    PRINT_INTERVAL = 10
    
    # Tracking
    episode_latencies = []
    episode_rewards = []
    best_latency = float('inf')
    best_episode = 0
    
    print("=" * 80)
    print("TRAINING: Double Q-Learning for Pure Latency Minimization")
    print("=" * 80)
    print("Goal: Minimize total inference latency")
    print("No deadlines, no surplus, no negative counts")
    print("=" * 80)
    print(f"Total episodes: {NUM_EPISODES}")
    print(f"Learning rate α: {agent.alpha}")
    print(f"Discount factor γ: {agent.gamma}")
    print(f"Reward scale: {agent.reward_scale}")
    print(f"Initial ε: {agent.epsilon}, min ε: {agent.epsilon_min}, decay: {agent.epsilon_decay}")
    print("=" * 80)
    
    # Load existing tables if available
    agent.load()
    
    # Get initial environment values
    bandwidth = agent.simulator.get_current_bandwidth()
    cloud_contention = 0.0
    
    for episode in range(NUM_EPISODES):
        # Reset episode
        agent.start_episode()
        state = (bandwidth, cloud_contention, 0, None)   # start at layer 0
        done = False
        step_count = 0
        td_errors = []
        
        while not done:
            action, reward, latency_s, next_state, done = agent.step(state)
            td_error = agent.update(state, action, reward, next_state, done)
            td_errors.append(td_error)
            state = next_state
            bandwidth = state[0]
            cloud_contention = state[1]
            step_count += 1
        
        total_latency_ms, total_reward = agent.end_episode()
        episode_latencies.append(total_latency_ms)
        episode_rewards.append(total_reward)
        
        if total_latency_ms < best_latency:
            best_latency = total_latency_ms
            best_episode = episode
        
        if episode % PRINT_INTERVAL == 0:
            avg_latency = np.mean(episode_latencies[-PRINT_INTERVAL:])
            avg_reward = np.mean(episode_rewards[-PRINT_INTERVAL:])
            avg_td = np.mean(td_errors) if td_errors else 0
            
            print(f"Episode {episode:4d} | "
                  f"Latency: {total_latency_ms:6.1f}ms | "
                  f"Avg Latency: {avg_latency:6.1f}ms | "
                  f"Best: {best_latency:6.1f}ms | "
                  f"Reward: {total_reward:7.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"TD: {avg_td:6.2f} | "
                  f"Steps: {step_count}")
    
    agent.save()
    
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best latency: {best_latency:.2f}ms at episode {best_episode}")
    print(f"Final ε: {agent.epsilon:.3f}")
    print("=" * 80)
    
    return agent, episode_latencies, episode_rewards


def evaluate_agent(agent, num_episodes=100):
    """
    Evaluate trained agent and collect assignment counts per (layer, node).
    
    Returns:
        latencies_ms: list of total latencies per episode
        rewards: list of total rewards per episode
        assignment_counts: dict mapping (layer, node) -> {'edge': count, 'cloud': count}
    """
    print("\n" + "=" * 80)
    print("EVALUATION: Double Q-Learning Agent")
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
            action, reward, latency_s, next_state, done = agent.step(state)
            
            # Record assignment for each node in this action
            layer = int(state[2])   # current layer index
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
    ax.set_title('Double Q-Learning: Assignment per Model Segment')
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

