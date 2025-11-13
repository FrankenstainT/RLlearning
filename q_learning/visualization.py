"""
Visualization utilities for maze learning
==========================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List, Dict
import os
import imageio.v2 as imageio
from maze_env import MazeEnv


def moving_average(data: np.ndarray, window: int = 100) -> np.ndarray:
    """Compute moving average."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_training_progress(episode_rewards: List[float], episode_lengths: List[int], 
                          td_errors: List[float], window: int = 100, save_path: str = 'results/training_progress.png'):
    """Plot training progress."""
    episodes = np.arange(1, len(episode_rewards) + 1)
    
    # Episode rewards with rolling average
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Rewards
    axes[0].plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw')
    if len(episode_rewards) >= window:
        ma_rewards = moving_average(np.array(episode_rewards), window)
        ma_episodes = np.arange(window, len(episode_rewards) + 1)
        axes[0].plot(ma_episodes, ma_rewards, color='red', linewidth=2, 
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


def visualize_value_and_policy(env: MazeEnv, agent, save_path: str = 'results/value_and_policy.png'):
    """Visualize the learned value function and policy."""
    size = env.size
    
    # Extract value function and policy
    value_map = np.zeros((size, size))
    policy_map = np.zeros((size, size), dtype=int)
    
    for y in range(size):
        for x in range(size):
            state = (y, x)
            
            if state in env.obstacles:
                value_map[y, x] = np.nan
                policy_map[y, x] = -1
            elif state == env.goal:
                value_map[y, x] = 100.0  # Goal value
                policy_map[y, x] = -1
            else:
                # Check if agent uses state index (Q-learning) or features (DQN)
                if hasattr(agent, 'qtable'):
                    # Q-learning: use state index
                    state_idx = env.state_to_index(state)
                    value_map[y, x] = agent.get_value(state_idx)
                    policy_map[y, x] = agent.get_policy(state_idx)
                else:
                    # DQN: use feature vector
                    state_features = env.state_to_features(state)
                    value_map[y, x] = agent.get_value(state_features)
                    policy_map[y, x] = agent.get_policy(state_features)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Value function heatmap
    im1 = axes[0].imshow(value_map, cmap='viridis', origin='upper')
    axes[0].set_title('Learned Value Function', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])
    
    # Add text annotations for values
    for y in range(size):
        for x in range(size):
            if not np.isnan(value_map[y, x]):
                axes[0].text(x, y, f'{value_map[y, x]:.6f}', 
                           ha='center', va='center', color='white', fontweight='bold')
    
    # Policy arrow map
    axes[1].imshow(np.ones((size, size)), cmap='gray', alpha=0.3, origin='upper')
    axes[1].set_title('Learned Policy (Arrow Map)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    
    # Draw arrows for policy
    arrow_dict = {
        0: (0, -0.35, '↑'),  # Up
        1: (0, 0.35, '↓'),   # Down
        2: (-0.35, 0, '←'),  # Left
        3: (0.35, 0, '→')    # Right
    }
    
    for y in range(size):
        for x in range(size):
            if (y, x) in env.obstacles:
                axes[1].text(x, y, '█', ha='center', va='center', 
                           fontsize=20, color='red', fontweight='bold')
            elif (y, x) == env.goal:
                axes[1].text(x, y, '★', ha='center', va='center', 
                           fontsize=20, color='gold', fontweight='bold')
            else:
                action = policy_map[y, x]
                if action >= 0:
                    dx, dy, symbol = arrow_dict[action]
                    # Draw arrow
                    axes[1].arrow(x, y, dx, dy, head_width=0.2, head_length=0.2,
                               fc='darkblue', ec='darkblue', linewidth=2.5, alpha=0.8)
    
    # Set ticks
    for ax in axes:
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.grid(True, color='black', linewidth=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved value function and policy visualization to {save_path}")


def draw_frame(env: MazeEnv, state: Tuple[int, int], step_idx: int, 
               out_dir: str, action: int = None):
    """Draw a single frame of the maze state."""
    os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    size = env.size
    
    # Draw grid
    for y in range(size + 1):
        ax.plot([0, size], [y, y], 'k-', linewidth=2)
    for x in range(size + 1):
        ax.plot([x, x], [0, size], 'k-', linewidth=2)
    
    # Draw obstacles
    for obs_y, obs_x in env.obstacles:
        rect = patches.Rectangle((obs_x - 0.5, obs_y - 0.5), 1, 1,
                               linewidth=2, edgecolor='red', facecolor='red', alpha=0.7)
        ax.add_patch(rect)
        ax.text(obs_x, obs_y, 'OBST', ha='center', va='center',
               fontsize=14, fontweight='bold', color='white')
    
    # Draw goal
    goal_y, goal_x = env.goal
    rect = patches.Rectangle((goal_x - 0.5, goal_y - 0.5), 1, 1,
                           linewidth=2, edgecolor='gold', facecolor='gold', alpha=0.8)
    ax.add_patch(rect)
    ax.text(goal_x, goal_y, 'GOAL', ha='center', va='center',
           fontsize=16, fontweight='bold', color='black')
    
    # Draw agent
    agent_y, agent_x = state
    circle = plt.Circle((agent_x, agent_y), 0.3, color='blue', zorder=10)
    ax.add_patch(circle)
    ax.text(agent_x, agent_y, 'A', ha='center', va='center',
           fontsize=16, fontweight='bold', color='white', zorder=11)
    
    # Draw action arrow if provided
    if action is not None:
        action_dirs = {
            0: (0, -0.4),  # Up
            1: (0, 0.4),    # Down
            2: (-0.4, 0),   # Left
            3: (0.4, 0)     # Right
        }
        if action in action_dirs:
            dx, dy = action_dirs[action]
            ax.arrow(agent_x, agent_y, dx, dy, head_width=0.15, head_length=0.15,
                   fc='darkblue', ec='darkblue', linewidth=3, zorder=9)
    
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'Step {step_idx}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    fname = os.path.join(out_dir, f"frame_{step_idx:04d}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return fname


def frames_to_gif(frames_dir: str, out_gif: str, fps: int = 3):
    """Convert frames to GIF."""
    files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
                   if f.lower().endswith(".png")])
    if not files:
        print(f"No frames in {frames_dir}")
        return
    try:
        imgs = [imageio.imread(f) for f in files]
        imageio.mimsave(out_gif, imgs, duration=1.0 / fps, loop=0)
        print(f"Saved GIF: {out_gif}")
    except Exception as e:
        print(f"Error creating GIF {out_gif}: {e}")

