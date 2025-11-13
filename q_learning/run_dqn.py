"""
Run DQN on 4x4 Maze
==================
"""

import numpy as np
from maze_env import MazeEnv
from dqn_learning import DQNLearner
from visualization import plot_training_progress, visualize_value_and_policy, \
                        draw_frame, frames_to_gif
import os


def train_agent(env: MazeEnv, agent: DQNLearner, num_episodes: int = 10000, max_steps: int = 50):
    """Train the DQN agent."""
    episode_rewards = []
    episode_lengths = []
    td_errors_per_episode = []
    
    print(f"Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Reset to random start position
        state = env.reset()
        state_features = env.state_to_features(state)
        
        agent.start_episode()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            # Choose action
            action = agent.choose_action(state_features, training=True)
            
            # Take step
            next_state, reward, done = env.step(state, action)
            next_state_features = env.state_to_features(next_state)
            
            # Store experience in replay buffer
            agent.update(state_features, action, reward, next_state_features, done)
            
            # Train the network
            agent.train_step()
            
            total_reward += reward
            steps += 1
            state = next_state
            state_features = next_state_features
        
        # End episode and get average TD error
        avg_td = agent.end_episode()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        td_errors_per_episode.append(avg_td)
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}, "
                  f"Avg Steps: {np.mean(episode_lengths[-100:]):.2f}")
    
    return episode_rewards, episode_lengths, td_errors_per_episode


def evaluate_all_starts(env: MazeEnv, agent: DQNLearner, max_steps: int = 50):
    """Evaluate agent from all possible start positions, record frames, and create GIFs."""
    traces = {}
    eval_dir = os.path.join("results", "dqn_evaluation_traces")
    os.makedirs(eval_dir, exist_ok=True)
    
    print("\nEvaluating from all possible start positions...")
    print(f"Creating frames and GIFs in {eval_dir}...")
    
    for idx, start_pos in enumerate(env.valid_starts):
        state = start_pos
        state_features = env.state_to_features(state)
        
        # Create directory for this start position
        case_name = f"eval_start_{start_pos[0]}_{start_pos[1]}"
        frames_dir = os.path.join(eval_dir, case_name)
        os.makedirs(frames_dir, exist_ok=True)
        
        trace = [state]  # Store the trajectory
        total_reward = 0
        steps = 0
        done = False
        
        # Greedy evaluation (no exploration)
        while not done and steps < max_steps:
            # Choose best action (greedy) BEFORE drawing frame
            action = agent.choose_action(state_features, training=False)
            
            # Draw frame with the action that will be taken
            draw_frame(env, state, steps, frames_dir, action=action)
            
            # Take step
            next_state, reward, done = env.step(state, action)
            next_state_features = env.state_to_features(next_state)
            
            trace.append(next_state)
            total_reward += reward
            steps += 1
            
            state = next_state
            state_features = next_state_features
        
        # Draw final frame if we reached goal or max steps
        if done or steps >= max_steps:
            # No action for final frame (episode ended)
            draw_frame(env, state, steps, frames_dir)
        
        # Create GIF from frames
        gif_path = os.path.join(frames_dir, f"{case_name}.gif")
        try:
            frames_to_gif(frames_dir, gif_path, fps=3)
        except Exception as e:
            print(f"[GIF creation skipped for {case_name}] {e}")
            gif_path = ""
        
        traces[start_pos] = {
            'trace': trace,
            'reward': total_reward,
            'steps': steps,
            'success': done and state == env.goal,
            'frames_dir': frames_dir,
            'gif_path': gif_path
        }
        
        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1}/{len(env.valid_starts)} start positions...")
    
    # Print summary
    successful = sum(1 for data in traces.values() if data['success'])
    print(f"\nEvaluation Summary:")
    print(f"  Total start positions: {len(traces)}")
    print(f"  Successful paths: {successful}")
    print(f"  Average steps: {np.mean([data['steps'] for data in traces.values()]):.2f}")
    print(f"  Average reward: {np.mean([data['reward'] for data in traces.values()]):.2f}")
    print(f"  Frames and GIFs saved to: {eval_dir}")


def main():
    """Main training and evaluation loop."""
    # Set random seed for reproducibility
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    
    # Create environment
    env = MazeEnv(size=4)
    
    # Create DQN agent
    agent = DQNLearner(
        input_size=2,  # (x, y) coordinates
        action_size=4,
        learning_rate=1e-3,
        gamma=0.95,
        epsilon=0.15,
        tau=0.005,
        batch_size=64,
        buffer_size=10000
    )
    
    # Train agent
    episode_rewards, episode_lengths, td_errors = train_agent(
        env, agent, num_episodes=10000, max_steps=50
    )
    
    # Plot training progress
    plot_training_progress(episode_rewards, episode_lengths, td_errors,
                          save_path='results/dqn_training_progress.png')
    
    # Visualize learned value function and policy
    visualize_value_and_policy(env, agent, save_path='results/dqn_value_and_policy.png')
    
    # Evaluate from all start positions
    evaluate_all_starts(env, agent, max_steps=50)
    
    print("\nTraining and evaluation complete!")
    print("All results saved to the 'results' directory.")


if __name__ == "__main__":
    main()

