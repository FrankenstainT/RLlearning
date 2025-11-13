"""
Run Cooperative DQN for Two-Agent Game
======================================
"""

import numpy as np
from typing import Tuple
from cooperative_env import CooperativeEnv
from cooperative_dqn import CooperativeDQN
from visualization import frames_to_gif
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def train_agent(env: CooperativeEnv, agent: CooperativeDQN, 
                num_episodes: int = 30000, max_steps: int = 50):
    """Train the cooperative DQN agent."""
    episode_rewards = []
    episode_lengths = []
    td_errors_per_episode = []
    
    print(f"Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Reset to random distinct positions
        agent1_pos, agent2_pos, visited_goals = env.reset()
        state = env.state_to_features(agent1_pos, agent2_pos, visited_goals)
        
        agent.start_episode()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            # Choose joint action
            action1, action2 = agent.choose_joint_action(state, training=True)
            joint_action_idx = env.joint_action_to_index(action1, action2)
            
            # Take step
            next_agent1_pos, next_agent2_pos, new_visited_goals, reward, done = \
                env.step(agent1_pos, action1, agent2_pos, action2, visited_goals)
            
            next_state = env.state_to_features(next_agent1_pos, next_agent2_pos, new_visited_goals)
            
            # Store experience
            agent.update(state, joint_action_idx, reward, next_state, done)
            
            # Train the network
            agent.train_step()
            
            total_reward += reward
            steps += 1
            
            agent1_pos = next_agent1_pos
            agent2_pos = next_agent2_pos
            visited_goals = new_visited_goals
            state = next_state
        
        # If episode ended due to max_steps, add final step with 0 reward
        if not done and steps >= max_steps:
            action1, action2 = agent.choose_joint_action(state, training=True)
            joint_action_idx = env.joint_action_to_index(action1, action2)
            
            final_reward = 0.0
            final_done = True
            
            agent.update(state, joint_action_idx, final_reward, state, final_done)
            agent.train_step()
            
            total_reward += final_reward
        
        # End episode and get average TD error
        avg_td = agent.end_episode()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        td_errors_per_episode.append(avg_td)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}, "
                  f"Avg Steps: {np.mean(episode_lengths[-100:]):.2f}, "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    return episode_rewards, episode_lengths, td_errors_per_episode


def plot_training_progress(episode_rewards: list, episode_lengths: list,
                          td_errors: list, window: int = 100,
                          save_path: str = 'cooperative_results/training_progress.png'):
    """Plot training progress."""
    from visualization import moving_average
    
    episodes = np.arange(1, len(episode_rewards) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Episode rewards
    axes[0].plot(episodes, episode_rewards, alpha=0.3, color='green', label='Raw')
    if len(episode_rewards) >= window:
        ma_rewards = moving_average(np.array(episode_rewards), window)
        ma_episodes = np.arange(window, len(episode_rewards) + 1)
        axes[0].plot(ma_episodes, ma_rewards, color='darkgreen', linewidth=2,
                    label=f'Rolling Avg (window={window})')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title('Episode Rewards Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # TD Errors
    axes[1].plot(episodes, td_errors, alpha=0.5, color='green')
    if len(td_errors) >= window:
        ma_td = moving_average(np.array(td_errors), window)
        ma_episodes = np.arange(window, len(td_errors) + 1)
        axes[1].plot(ma_episodes, ma_td, color='darkgreen', linewidth=2,
                    label=f'Rolling Avg (window={window})')
        axes[1].legend()
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Average TD Error')
    axes[1].set_title('TD Error Per Episode')
    axes[1].grid(True, alpha=0.3)
    
    # Steps per episode
    axes[2].plot(episodes, episode_lengths, alpha=0.3, color='purple', label='Raw')
    if len(episode_lengths) >= window:
        ma_steps = moving_average(np.array(episode_lengths), window)
        ma_episodes = np.arange(window, len(episode_lengths) + 1)
        axes[2].plot(ma_episodes, ma_steps, color='darkviolet', linewidth=2,
                    label=f'Rolling Avg (window={window})')
        axes[2].legend()
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Steps per Episode')
    axes[2].set_title('Episode Length Over Time')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training progress plots to {save_path}")


def visualize_policy(env: CooperativeEnv, agent: CooperativeDQN,
                    save_path: str = 'cooperative_results/policy.png'):
    """Visualize learned policy for different agent positions."""
    size = env.size
    
    # Show policy for agent1 at different positions, with agent2 fixed
    # We'll create a grid showing the best joint action
    fig, axes = plt.subplots(size, size, figsize=(16, 16))
    
    fixed_agent2 = (1, 1)  # Fix agent2 at center
    
    for agent1_y in range(size):
        for agent1_x in range(size):
            agent1_pos = (agent1_y, agent1_x)
            
            if agent1_pos == fixed_agent2:
                # Skip if same position
                axes[agent1_y, agent1_x].axis('off')
                continue
            
            ax = axes[agent1_y, agent1_x]
            # Show policy with no goals visited
            state = env.state_to_features(agent1_pos, fixed_agent2, visited_goals=set())
            action1, action2 = agent.get_policy(state)
            
            # Draw grid
            for y in range(size + 1):
                ax.plot([0, size], [y, y], 'k-', linewidth=0.5)
            for x in range(size + 1):
                ax.plot([x, x], [0, size], 'k-', linewidth=0.5)
            
            # Draw goals
            goal1_y, goal1_x = env.goal1
            goal2_y, goal2_x = env.goal2
            ax.add_patch(patches.Rectangle((goal1_x - 0.5, goal1_y - 0.5), 1, 1,
                                         linewidth=2, edgecolor='gold', facecolor='gold', alpha=0.7))
            ax.add_patch(patches.Rectangle((goal2_x - 0.5, goal2_y - 0.5), 1, 1,
                                         linewidth=2, edgecolor='gold', facecolor='gold', alpha=0.7))
            
            # Draw agents
            circle1 = plt.Circle((agent1_x, agent1_y), 0.2, color='blue', zorder=10)
            circle2 = plt.Circle((fixed_agent2[1], fixed_agent2[0]), 0.2, color='orange', zorder=10)
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            
            # Draw action arrows
            action_dirs = {
                0: (0, -0.3),  # Up
                1: (0, 0.3),    # Down
                2: (-0.3, 0),   # Left
                3: (0.3, 0)     # Right
            }
            
            if action1 in action_dirs:
                dx, dy = action_dirs[action1]
                ax.arrow(agent1_x, agent1_y, dx, dy, head_width=0.1, head_length=0.1,
                       fc='blue', ec='blue', linewidth=2, zorder=9)
            
            if action2 in action_dirs:
                dx, dy = action_dirs[action2]
                ax.arrow(fixed_agent2[1], fixed_agent2[0], dx, dy, head_width=0.1, head_length=0.1,
                       fc='orange', ec='orange', linewidth=2, zorder=9)
            
            ax.set_xlim(-0.5, size - 0.5)
            ax.set_ylim(-0.5, size - 0.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'A1@({agent1_x},{agent1_y})', fontsize=8)
    
    plt.suptitle('Learned Policy (Agent2 fixed at (1,1))', fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved policy visualization to {save_path}")


def draw_cooperative_frame(env: CooperativeEnv, agent1_pos: Tuple[int, int],
                          agent2_pos: Tuple[int, int], visited_goals: set,
                          step_idx: int, out_dir: str,
                          action1: int = None, action2: int = None):
    """Draw a single frame with both agents."""
    os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    size = env.size
    
    # Draw grid
    for y in range(size + 1):
        ax.plot([0, size], [y, y], 'k-', linewidth=2)
    for x in range(size + 1):
        ax.plot([x, x], [0, size], 'k-', linewidth=2)
    
    # Draw goals
    goal1_y, goal1_x = env.goal1
    goal2_y, goal2_x = env.goal2
    
    # Color goals based on whether they've been visited
    goal1_color = 'green' if env.goal1 in visited_goals else 'gold'
    goal2_color = 'green' if env.goal2 in visited_goals else 'gold'
    
    ax.add_patch(patches.Rectangle((goal1_x - 0.5, goal1_y - 0.5), 1, 1,
                                   linewidth=2, edgecolor=goal1_color, facecolor=goal1_color, alpha=0.8))
    ax.text(goal1_x, goal1_y, 'G1', ha='center', va='center',
           fontsize=14, fontweight='bold', color='black')
    
    ax.add_patch(patches.Rectangle((goal2_x - 0.5, goal2_y - 0.5), 1, 1,
                                   linewidth=2, edgecolor=goal2_color, facecolor=goal2_color, alpha=0.8))
    ax.text(goal2_x, goal2_y, 'G2', ha='center', va='center',
           fontsize=14, fontweight='bold', color='black')
    
    # Draw agents
    agent1_y, agent1_x = agent1_pos
    agent2_y, agent2_x = agent2_pos
    
    circle1 = plt.Circle((agent1_x, agent1_y), 0.3, color='blue', zorder=10)
    circle2 = plt.Circle((agent2_x, agent2_y), 0.3, color='orange', zorder=10)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.text(agent1_x, agent1_y, '1', ha='center', va='center',
           fontsize=14, fontweight='bold', color='white', zorder=11)
    ax.text(agent2_x, agent2_y, '2', ha='center', va='center',
           fontsize=14, fontweight='bold', color='white', zorder=11)
    
    # Draw action arrows
    action_dirs = {
        0: (0, -0.4),  # Up
        1: (0, 0.4),    # Down
        2: (-0.4, 0),   # Left
        3: (0.4, 0)     # Right
    }
    
    if action1 is not None and action1 in action_dirs:
        dx, dy = action_dirs[action1]
        ax.arrow(agent1_x, agent1_y, dx, dy, head_width=0.15, head_length=0.15,
               fc='darkblue', ec='darkblue', linewidth=3, zorder=9)
    
    if action2 is not None and action2 in action_dirs:
        dx, dy = action_dirs[action2]
        ax.arrow(agent2_x, agent2_y, dx, dy, head_width=0.15, head_length=0.15,
               fc='darkorange', ec='darkorange', linewidth=3, zorder=9)
    
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'Step {step_idx} | Goals visited: {len(visited_goals)}/2', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    fname = os.path.join(out_dir, f"frame_{step_idx:04d}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return fname


def evaluate_all_starts(env: CooperativeEnv, agent: CooperativeDQN, 
                        max_steps: int = 50, outdir: str = 'cooperative_results'):
    """Evaluate from all possible start position pairs, record frames, and create GIFs."""
    traces = {}
    eval_dir = os.path.join(outdir, "evaluation_traces")
    os.makedirs(eval_dir, exist_ok=True)
    
    start_pairs = env.get_all_start_pairs()
    print(f"\nEvaluating from {len(start_pairs)} possible start position pairs...")
    print(f"Creating frames and GIFs in {eval_dir}...")
    
    for idx, (start_agent1_pos, start_agent2_pos) in enumerate(start_pairs):
        agent1_pos = start_agent1_pos
        agent2_pos = start_agent2_pos
        visited_goals = set()
        state = env.state_to_features(agent1_pos, agent2_pos, visited_goals)
        
        # Create directory for this start pair
        case_name = f"eval_A1{start_agent1_pos[0]}_{start_agent1_pos[1]}_A2{start_agent2_pos[0]}_{start_agent2_pos[1]}"
        frames_dir = os.path.join(eval_dir, case_name)
        os.makedirs(frames_dir, exist_ok=True)
        
        trace = [(agent1_pos, agent2_pos, visited_goals.copy())]
        total_reward = 0
        steps = 0
        done = False
        
        # Greedy evaluation (no exploration)
        while not done and steps < max_steps:
            # Choose best joint action (greedy) BEFORE drawing frame
            action1, action2 = agent.choose_joint_action(state, training=False)
            
            # Draw frame with the actions that will be taken
            draw_cooperative_frame(env, agent1_pos, agent2_pos, visited_goals, steps, frames_dir,
                                 action1=action1, action2=action2)
            
            # Take step
            next_agent1_pos, next_agent2_pos, new_visited_goals, reward, done = \
                env.step(agent1_pos, action1, agent2_pos, action2, visited_goals)
            
            trace.append((next_agent1_pos, next_agent2_pos, new_visited_goals.copy()))
            total_reward += reward
            steps += 1
            
            agent1_pos = next_agent1_pos
            agent2_pos = next_agent2_pos
            visited_goals = new_visited_goals
            state = env.state_to_features(agent1_pos, agent2_pos, visited_goals)
        
        # Draw final frame
        if done or steps >= max_steps:
            draw_cooperative_frame(env, agent1_pos, agent2_pos, visited_goals, steps, frames_dir)
        
        # Create GIF from frames
        gif_path = os.path.join(frames_dir, f"{case_name}.gif")
        try:
            frames_to_gif(frames_dir, gif_path, fps=3)
        except Exception as e:
            print(f"[GIF creation skipped for {case_name}] {e}")
            gif_path = ""
        
        traces[(start_agent1_pos, start_agent2_pos)] = {
            'trace': trace,
            'reward': total_reward,
            'steps': steps,
            'success': done and len(visited_goals) == 2,
            'frames_dir': frames_dir,
            'gif_path': gif_path
        }
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(start_pairs)} start pairs...")
    
    # Print summary
    successful = sum(1 for data in traces.values() if data['success'])
    print(f"\nEvaluation Summary:")
    print(f"  Total start pairs: {len(traces)}")
    print(f"  Successful (both goals visited): {successful}")
    print(f"  Average steps: {np.mean([data['steps'] for data in traces.values()]):.2f}")
    print(f"  Average reward: {np.mean([data['reward'] for data in traces.values()]):.2f}")
    print(f"  Frames and GIFs saved to: {eval_dir}")


def main():
    """Main training and evaluation loop."""
    # Set random seed for reproducibility
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    
    # Create environment with two goals
    env = CooperativeEnv(size=4, goal1=(0, 0), goal2=(3, 3))
    
    # Create cooperative DQN agent
    agent = CooperativeDQN(
        input_size=6,  # [agent1_x, agent1_y, agent2_x, agent2_y, goal1_visited, goal2_visited]
        joint_action_size=16,  # 4 × 4 joint actions
        learning_rate=5e-4,  # Reduced for stability
        gamma=0.95,
        epsilon_start=1.0,  # Start with full exploration
        epsilon_end=0.01,  # End with minimal exploration
        epsilon_decay=0.995,  # Decay per episode
        tau=0.01,  # Faster target network update
        batch_size=64,
        buffer_size=50000  # Larger replay buffer
    )
    
    # Train agent
    episode_rewards, episode_lengths, td_errors = train_agent(
        env, agent, num_episodes=30000, max_steps=50
    )
    
    outdir = 'cooperative_results'
    
    # Plot training progress
    plot_training_progress(
        episode_rewards, episode_lengths, td_errors,
        save_path=os.path.join(outdir, 'training_progress.png')
    )
    
    # Visualize learned policy
    visualize_policy(env, agent, save_path=os.path.join(outdir, 'policy.png'))
    
    # Evaluate from all start position pairs
    evaluate_all_starts(env, agent, max_steps=50, outdir=outdir)
    
    print("\nTraining and evaluation complete!")
    print(f"All results saved to the '{outdir}' directory.")


if __name__ == "__main__":
    main()

