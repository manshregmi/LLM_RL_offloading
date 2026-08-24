import asyncio
import os
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
    bandwidth = 12
    cloud_contention = 0.0  # No pending cloud tasks at start
    previous_assignment = None  # No previous layer
    segment = simulator.get_segment_tuple(0)
    return (bandwidth, cloud_contention,segment, previous_assignment)


def train_a2c_agent(profiling_data: ProfilingData, episodes=50000, is_test=False, verbose=True, total_pipelines=1): 
    """Main training loop."""
    
    # Create agent
    agent = TabularActorCriticAgent(
        profiling_data=profiling_data,
        is_test=is_test,
        alpha_actor=0.25,
        alpha_critic=0.25,
        gamma=0.7,
        #average_reward_lr= 0.15,
        reward_scale=10.0,  # Scale reward magnitude
        total_pipelines=total_pipelines
    )
    agent.simulator.reset_layer_count()
    contention_trace_path = os.path.join("simulator","data","contention_log_20260823_152331.csv")
    agent.simulator.load_contention_trace(contention_trace_path)

    grouping_RL_agent = GroupingRL(total_pipelines=total_pipelines)
    
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
    print(f"Reward scale: {agent.reward_scale}")
    print("=" * 80)
    state = create_initial_state(agent.simulator)
    bandwidth = state[0]
    cloud_contention = state[1]
    segment = agent.simulator.get_segment_tuple(0)
    last_pipeline_contention = []
    average_last_pipeline_contention = 0.0
    state = (bandwidth, cloud_contention, 0, None)  # Start at segment 0 with no previous assignment
    agent.load()  # Load existing model if available
    grouping_RL_agent.load()  # Load grouping agent if it has a saved state
    episode_overhead_time = []
    average_step_overhead_times = []
    episode_number_of_groups = []
    episode_generated_tokens= []
    for episode in range(NUM_EPISODES):
        rewards_ep = 0
        state = (bandwidth, cloud_contention, segment, None)
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
        episode_number_of_groups.append(number_of_groups)
        # Run episode
        action_array = []
        count = 0
        while not done:
            action, reward, latency_s, next_state, done, overhead_time_per_step, cached_count = agent.step(state, num_groups= number_of_groups , count=count)
            # if (step_count < 2):
            #     print("first two actions are:", action)
            if(done):
                total_generated_tokens = agent.simulator.get_total_generated_tokens(done)
                episode_generated_tokens.append(total_generated_tokens)
            count = cached_count
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
        # print(f"there are {cached_count} cached actions for {number_of_groups} groups")
        # print("Assignment Vector",action_array)  
        episode_overhead_time.append(np.sum(step_overhead_time))
        average_step_overhead_times.append(np.mean(step_overhead_time))

        average_last_pipeline_contention = np.mean(last_pipeline_contention) if last_pipeline_contention else 0.0
        last_pipeline_contention = []
        
        # End episode
        total_latency_ms, total_reward = agent.end_episode()
        episode_latencies.append(total_latency_ms)
        episode_rewards.append(total_reward)
        
        grouping_RL_agent._update_tables(
            state_key=grouping_RL_agent.last_state_key,
            action_key=grouping_RL_agent.last_action_key,
            reward=total_reward,
            next_state_key=None,
            done=True
            )
        
        if (episode % 100)== 0:
            grouping_RL_agent.save()  # Save grouping agent state after each episode



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
    print(f"Mean overhead time per episode: {np.mean(episode_overhead_time[100:])*1000:.4f}ms")
    print(f"Std overhead time per episode: {np.std(episode_overhead_time[100:])*1000:.4f}ms")
    print(f"Mean overhead time per step: {np.mean(average_step_overhead_times)*1000:.4f}ms")


    print("\n" + "=" * 80)
    print("Average time per token:")
    print(f"Average time per token: {np.mean(episode_latencies)/np.mean(episode_generated_tokens):.4f}ms")
    print(f" minimum time per token: {np.min(episode_latencies)/np.max(episode_generated_tokens):.4f}ms")
    print(f" maximum time per token: {np.max(episode_latencies)/np.min(episode_generated_tokens):.4f}ms")

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best latency: {best_latency:.2f}ms at episode {best_episode}")
    print(f"Final temperature: {agent.temperature:.3f}")
    print("=" * 80)

    print("Episode Number of Groups:")
    print(f"Mean number of groups per episode: {np.mean(episode_number_of_groups):.4f}")
    print(f"Std number of groups per episode: {np.std(episode_number_of_groups):.4f}")

    # # plt.plot(episode_number_of_groups)
    # plt.plot(moving_average(episode_number_of_groups,300))
    # plt.xlabel("Episode")
    # plt.ylabel("Number of Groups")
    # plt.title("Number of Groups per Episode")
    # plt.savefig("number_of_groups_per_episode.png", dpi=600)
    # plt.show()


    return agent, episode_latencies, episode_rewards, np.mean(episode_overhead_time[100:])

def moving_average(data, window):
    """Compute moving average with given window size."""
    return np.convolve(data, np.ones(window)/window, mode='valid')

