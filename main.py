"""
Main execution script for Pure Latency Minimization with Actor-Critic and Baseline Comparisons
"""

import numpy as np
import pandas as pd
import os
from profiling.cascade_profiling_data import cascade_profiling
from profiling.initialize_agx_profiling import get_LLM_profiling_data
from runner.run_a2c import aggregate_assignments_by_segment, evaluate_agent, plot_assignment_percentages, train_a2c_agent
from baselines.hurustic_baselines import run_scheduler 
import matplotlib.pyplot  as plt

from runner.run_dq import train_double_q_agent


def run_baseline_comparison(profiling_data, episodes=100):
    """
    Run baseline schedulers for comparison.
    
    Args:
        profiling_data: ProfilingData object
        episodes: Number of episodes to run for each baseline
    
    Returns:
        dict: Dictionary containing results for each scheduler
    """
    print("\n" + "=" * 80)
    print("RUNNING BASELINE SCHEDULERS")
    print("=" * 80)
    
    baseline_results = {}
    
    # Random scheduler
    print("\n📊 Random Scheduler")
    print("-" * 40)
    avg_latency, avg_reward, _ = run_scheduler(
        profiling_data=profiling_data,
        episodes=episodes,
        scheduler_type='random'
    )
    baseline_results['random'] = {
        'latency_ms': avg_latency,
        'reward': avg_reward
    }
    
    # All Edge scheduler
    print("\n📊 All Edge Scheduler")
    print("-" * 40)
    avg_latency, avg_reward, _ = run_scheduler(
        profiling_data=profiling_data,
        episodes=episodes,
        scheduler_type='all_edge'
    )
    baseline_results['all_edge'] = {
        'latency_ms': avg_latency,
        'reward': avg_reward
    }
    
    # All Cloud scheduler
    print("\n📊 All Cloud Scheduler")
    print("-" * 40)
    avg_latency, avg_reward, _ = run_scheduler(
        profiling_data=profiling_data,
        episodes=episodes,
        scheduler_type='all_cloud'
    )
    baseline_results['all_cloud'] = {
        'latency_ms': avg_latency,
        'reward': avg_reward
    }
    
    # Print summary comparison
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Scheduler':<15} {'Latency (ms)':<15} {'Reward':<15}")
    print("-" * 45)
    for scheduler, results in baseline_results.items():
        print(f"{scheduler:<15} {results['latency_ms']:<15.2f} {results['reward']:<15.2f}")
    print("=" * 80)
    
    return baseline_results


def compare_with_a2c(a2c_latency, a2c_reward, baseline_results):
    """
    Compare A2C results with baselines.
    
    Args:
        a2c_latency: Average latency from A2C
        a2c_reward: Average reward from A2C
        baseline_results: Dictionary with baseline results
    """
    print("\n" + "=" * 80)
    print("A2C vs BASELINES COMPARISON")
    print("=" * 80)
    print(f"{'Method':<15} {'Latency (ms)':<15} {'Reward':<15} {'Improvement':<20}")
    print("-" * 65)
    
    # Print A2C results
    print(f"{'A2C':<15} {a2c_latency:<15.2f} {a2c_reward:<15.2f} {'--':<20}")
    
    # Compare with each baseline
    for scheduler, results in baseline_results.items():
        latency_improvement = ((results['latency_ms'] - a2c_latency) / results['latency_ms']) * 100
        reward_improvement = ((a2c_reward - results['reward']) / abs(results['reward'])) * 100 if results['reward'] != 0 else float('inf')
        
        print(f"{scheduler:<15} {results['latency_ms']:<15.2f} {results['reward']:<15.2f} "
              f"Latency: {latency_improvement:+.1f}%, Reward: {reward_improvement:+.1f}%")
    
    print("=" * 80)

def moving_average(data, window):
    """Compute moving average with given window size."""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_convergence_curve(episode_rewards, window=100, title="", 
                           xlabel="Episode", ylabel="Reward", save_path="/Users/Manish/Desktop/LLM_RL_offloading/"):
    """
    Plot smoothed convergence curve of episode rewards (or latencies).
    
    Args:
        episode_rewards: list or array of values per episode
        window: moving average window size
        title, xlabel, ylabel: plot labels
        save_path: if provided, save figure to this path
    """
    plt.figure(figsize=(10, 6))
    
    # Plot raw values (light, semi-transparent)
    episodes = np.arange(len(episode_rewards))
    # plt.plot(episodes, episode_rewards, alpha=0.3, color='gray', label='Raw')
    
    # Plot smoothed values
    smoothed = moving_average(episode_rewards, window)
    smoothed_episodes = np.arange(window-1, len(episode_rewards))
    plt.plot(smoothed_episodes, smoothed, linewidth=2, color='blue', label=f'Moving avg (window={window})')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load profiling data
    print("=" * 80)
    print("LOADING PROFILING DATA")
    print("=" * 80)
    profiling_data = get_LLM_profiling_data()
    pipleline_overhead_time = []

    for n in range(1,10):
        try:
            os.remove("a2c_tables.pkl")
        except FileNotFoundError:
            pass   # Ignore if file doesn’t exist (like rm -f)
        try:
            os.remove("grouping_rl_tables.pkl")
        except FileNotFoundError:
            pass   # Ignore if file doesn’t exist (like rm -f)
        
        cascaded_profiling_data = cascade_profiling(profiling_data,n=n)  # Create 3 copies for pipelining
        print(f"✅ Loaded profiling data")
        print(f"   Number of layers: {len(profiling_data.layers)}")
        print(f"   Number of edge devices: {profiling_data.numberOfEdgeDevice}")
        print(f"   RTT: {profiling_data.rtt}ms")
        print(f"   Deadline (info only): {profiling_data.deadline}ms")
        print("=" * 80)
        
        # Define number of episodes for training and baselines
        TRAIN_EPISODES = 1000
        BASELINE_EPISODES = 1
        
        # # Run baseline schedulers
        # baseline_results = run_baseline_comparison(
        #     profiling_data=profiling_data,
        #     episodes=BASELINE_EPISODES
        # )
        
        # Train A2C agent
        print("\n" + "=" * 80)
        print("TRAINING A2C AGENT")
        print("=" * 80)
        
        agent, episode_latencies, episode_rewards, overhead_time = train_a2c_agent(
            profiling_data=cascaded_profiling_data,
            episodes=TRAIN_EPISODES,
            is_test=True,           # Training mode
            verbose=False,            # Print progress
            total_pipelines=n
        )
        pipleline_overhead_time.append(overhead_time)
        print(f"Pipeline overhead time for {n} pipelines: {overhead_time:.2f} ms")
    pd.DataFrame(pipleline_overhead_time).to_csv('pipeline_overhead.csv', index=False, header=False)

    # agent_dq, episode_latencies_dq, episode_rewards_dq = train_double_q_agent(
    #     profiling_data=profiling_data,
    #     episodes=TRAIN_EPISODES,
    #     is_test=True,           # Training mode
    #     verbose=False,            # Print progress)
    # )
    # latencies_ms, rewards, assignment_counts = evaluate_agent(agent, num_episodes=1000)

    # plot_convergence_curve(episode_rewards)

    # segments = aggregate_assignments_by_segment(assignment_counts)

    # plot_assignment_percentages(segments)




    
    # # Calculate average A2C performance (last 100 episodes for stable comparison)
    # a2c_avg_latency = np.mean(episode_latencies[-100:])
    # a2c_avg_reward = np.mean(episode_rewards[-100:])
    
    # print("\n" + "=" * 80)
    # print(f"A2C FINAL PERFORMANCE (last 100 episodes)")
    # print("=" * 80)
    # print(f"Average Latency: {a2c_avg_latency:.2f} ms")
    # print(f"Average Reward: {a2c_avg_reward:.2f}")
    # print("=" * 80)
    
    # # Compare A2C with baselines
    # compare_with_a2c(a2c_avg_latency, a2c_avg_reward, baseline_results)
    
    # # Optional: Plot comparison
    # try:
    #     import matplotlib.pyplot as plt
        
    #     plt.figure(figsize=(12, 5))
        
    #     # Latency comparison
    #     plt.subplot(1, 2, 1)
    #     schedulers = list(baseline_results.keys()) + ['A2C']
    #     latencies = [baseline_results[s]['latency_ms'] for s in baseline_results.keys()] + [a2c_avg_latency]
    #     colors = ['red', 'blue', 'green', 'purple']
        
    #     bars = plt.bar(schedulers, latencies, color=colors)
    #     plt.ylabel('Latency (ms)')
    #     plt.title('Latency Comparison: A2C vs Baselines')
    #     plt.xticks(rotation=45)
        
    #     # Add value labels on bars
    #     for bar, latency in zip(bars, latencies):
    #         plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
    #                 f'{latency:.1f}', ha='center', va='bottom')
        
    #     # Reward comparison
    #     plt.subplot(1, 2, 2)
    #     rewards = [baseline_results[s]['reward'] for s in baseline_results.keys()] + [a2c_avg_reward]
        
    #     bars = plt.bar(schedulers, rewards, color=colors)
    #     plt.ylabel('Reward')
    #     plt.title('Reward Comparison: A2C vs Baselines')
    #     plt.xticks(rotation=45)
        
    #     # Add value labels on bars
    #     for bar, reward in zip(bars, rewards):
    #         plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
    #                 f'{reward:.1f}', ha='center', va='bottom')
        
    #     plt.tight_layout()
    #     plt.savefig('a2c_vs_baselines_comparison.png', dpi=600)
    #     plt.show()
        
    # except ImportError:
    #     print("Matplotlib not available for plotting")
    
    # print("\n✅ Experiment completed!")
    
    # # Optional: Debug a single episode
    # # debug_single_episode(profiling_data)
