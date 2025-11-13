"""
4x4 Maze Environment with Q-Learning
=====================================

A custom 4x4 maze with obstacles at (1,1) and (2,2), goal at (3,3).
Agent learns to navigate using tabular Q-learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List, Dict
import os
import imageio.v2 as imageio

# Create output directory
os.makedirs("results", exist_ok=True)


class MazeEnv:
    """4x4 Maze Environment with obstacles and goal."""
    
    def __init__(self, size=4):
        self.size = size
        self.obstacles = [(1, 1), (2, 2)]  # Obstacles at (1,1) and (2,2)
        self.goal = (3, 3)  # Goal at (3,3)
        
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        self.action_names = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        
        # Get all valid start positions (excluding obstacles and goal)
        self.valid_starts = self._get_valid_starts()
        
    def _get_valid_starts(self) -> List[Tuple[int, int]]:
        """Get all valid starting positions (excluding obstacles and goal)."""
        valid = []
        for y in range(self.size):
            for x in range(self.size):
                pos = (y, x)
                if pos not in self.obstacles and pos != self.goal:
                    valid.append(pos)
        return valid
    
    def _is_valid_position(self, y: int, x: int) -> bool:
        """Check if position is within bounds and not an obstacle."""
        if not (0 <= y < self.size and 0 <= x < self.size):
            return False
        return (y, x) not in self.obstacles
    
    def reset(self, start_pos: Tuple[int, int] = None) -> Tuple[int, int]:
        """Reset environment to a random or specified start position."""
        if start_pos is None:
            start_pos = tuple(self.valid_starts[np.random.randint(len(self.valid_starts))])
        return start_pos
    
    def step(self, state: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take a step in the environment.
        Returns: (next_state, reward, done)
        """
        y, x = state
        dy, dx = self.actions[action]
        new_y, new_x = y + dy, x + dx
        
        # Check if move is valid
        if not self._is_valid_position(new_y, new_x):
            # Hit obstacle or out of bounds - stay in place, get penalty
            reward = -10.0
            next_state = state
        else:
            next_state = (new_y, new_x)
            reward = -1.0  # Step penalty
        
        # Check if goal reached
        if next_state == self.goal:
            reward = 100.0
            done = True
        else:
            done = False
        
        return next_state, reward, done
    
    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (y, x) state to linear index."""
        y, x = state
        return y * self.size + x
    
    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert linear index to (y, x) state."""
        y = index // self.size
        x = index % self.size
        return (y, x)


class QLearner:
    """Tabular Q-Learning agent."""
    
    def __init__(self, state_size: int, action_size: int, 
                 alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.15):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.qtable = np.zeros((state_size, action_size))
        
        # Track TD errors per episode
        self.episode_td_errors = []
    
    def choose_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            # Break ties randomly
            max_q = np.max(self.qtable[state, :])
            max_indices = np.where(self.qtable[state, :] == max_q)[0]
            return np.random.choice(max_indices)
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """Update Q-value using Q-learning rule."""
        current_q = self.qtable[state, action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.qtable[next_state, :])
        
        td_error = target - current_q
        self.qtable[state, action] += self.alpha * td_error
        
        # Track TD error for this episode
        self.episode_td_errors.append(abs(td_error))
    
    def start_episode(self):
        """Called at the start of each episode."""
        self.episode_td_errors = []
    
    def end_episode(self) -> float:
        """Called at the end of each episode. Returns average TD error."""
        if self.episode_td_errors:
            avg_td = np.mean(self.episode_td_errors)
        else:
            avg_td = 0.0
        self.episode_td_errors = []  # Reset for next episode
        return avg_td


def moving_average(data: np.ndarray, window: int = 100) -> np.ndarray:
    """Compute moving average."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def train_agent(env: MazeEnv, agent: QLearner, num_episodes: int = 10000, max_steps: int = 50):
    """Train the Q-learning agent."""
    episode_rewards = []
    episode_lengths = []
    td_errors_per_episode = []
    
    print(f"Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Reset to random start position
        state = env.reset()
        state_idx = env.state_to_index(state)
        
        agent.start_episode()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            # Choose action
            action = agent.choose_action(state_idx)
            
            # Take step
            next_state, reward, done = env.step(state, action)
            next_state_idx = env.state_to_index(next_state)
            
            # Update Q-table
            agent.update(state_idx, action, reward, next_state_idx, done)
            
            total_reward += reward
            steps += 1
            state = next_state
            state_idx = next_state_idx
        
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


def plot_training_progress(episode_rewards: List[float], episode_lengths: List[int], 
                          td_errors: List[float], window: int = 100):
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
    plt.savefig('results/training_progress.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved training progress plots to results/training_progress.png")


def visualize_value_and_policy(env: MazeEnv, agent: QLearner):
    """Visualize the learned value function and policy."""
    size = env.size
    
    # Extract value function and policy
    value_map = np.zeros((size, size))
    policy_map = np.zeros((size, size), dtype=int)
    
    for y in range(size):
        for x in range(size):
            state = (y, x)
            state_idx = env.state_to_index(state)
            
            if state in env.obstacles:
                value_map[y, x] = np.nan
                policy_map[y, x] = -1
            elif state == env.goal:
                value_map[y, x] = 100.0  # Goal value
                policy_map[y, x] = -1
            else:
                value_map[y, x] = np.max(agent.qtable[state_idx, :])
                policy_map[y, x] = np.argmax(agent.qtable[state_idx, :])
    
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
                axes[0].text(x, y, f'{value_map[y, x]:.1f}', 
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
    plt.savefig('results/value_and_policy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved value function and policy visualization to results/value_and_policy.png")


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


def evaluate_all_starts(env: MazeEnv, agent: QLearner, max_steps: int = 50):
    """Evaluate agent from all possible start positions, record frames, and create GIFs."""
    traces = {}
    eval_dir = os.path.join("results", "evaluation_traces")
    os.makedirs(eval_dir, exist_ok=True)
    
    print("\nEvaluating from all possible start positions...")
    print(f"Creating frames and GIFs in {eval_dir}...")
    
    for idx, start_pos in enumerate(env.valid_starts):
        state = start_pos
        state_idx = env.state_to_index(state)
        
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
            action = np.argmax(agent.qtable[state_idx, :])
            
            # Draw frame with the action that will be taken
            draw_frame(env, state, steps, frames_dir, action=action)
            
            # Take step
            next_state, reward, done = env.step(state, action)
            next_state_idx = env.state_to_index(next_state)
            
            trace.append(next_state)
            total_reward += reward
            steps += 1
            
            state = next_state
            state_idx = next_state_idx
        
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
    
    # Also create the overview plot
    _plot_all_traces_overview(env, traces)
    print("Saved evaluation traces overview to results/evaluation_traces.png")


def _plot_all_traces_overview(env: MazeEnv, traces: Dict):
    """Plot overview of all traces on a single figure."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw grid
    size = env.size
    for y in range(size + 1):
        ax.plot([0, size], [y, y], 'k-', linewidth=1)
    for x in range(size + 1):
        ax.plot([x, x], [0, size], 'k-', linewidth=1)
    
    # Draw obstacles
    for obs_y, obs_x in env.obstacles:
        rect = patches.Rectangle((obs_x - 0.5, obs_y - 0.5), 1, 1, 
                               linewidth=2, edgecolor='red', facecolor='red', alpha=0.5)
        ax.add_patch(rect)
        ax.text(obs_x, obs_y, 'OBST', ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white')
    
    # Draw goal
    goal_y, goal_x = env.goal
    rect = patches.Rectangle((goal_x - 0.5, goal_y - 0.5), 1, 1,
                           linewidth=2, edgecolor='gold', facecolor='gold', alpha=0.7)
    ax.add_patch(rect)
    ax.text(goal_x, goal_y, 'GOAL', ha='center', va='center',
           fontsize=12, fontweight='bold', color='black')
    
    # Plot traces
    colors = plt.cm.tab20(np.linspace(0, 1, len(traces)))
    for idx, (start_pos, data) in enumerate(traces.items()):
        trace = data['trace']
        color = colors[idx]
        
        # Plot trajectory
        if len(trace) > 1:
            xs = [pos[1] for pos in trace]
            ys = [pos[0] for pos in trace]
            ax.plot(xs, ys, 'o-', color=color, linewidth=2, markersize=6, 
                   alpha=0.7, label=f'Start{start_pos}')
            
            # Add arrow to show direction
            for i in range(len(trace) - 1):
                y1, x1 = trace[i]
                y2, x2 = trace[i + 1]
                dx = x2 - x1
                dy = y2 - y1
                if dx != 0 or dy != 0:
                    ax.arrow(x1, y1, dx * 0.3, dy * 0.3, 
                           head_width=0.1, head_length=0.1, 
                           fc=color, ec=color, alpha=0.7)
        
        # Mark start position
        start_y, start_x = start_pos
        ax.plot(start_x, start_y, 's', color=color, markersize=12, 
               markeredgecolor='black', markeredgewidth=2, label=f'Start{start_pos}')
    
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis to match matrix indexing
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Evaluation Traces from All Start Positions', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/evaluation_traces.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main training and evaluation loop."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create environment
    env = MazeEnv(size=4)
    
    # Create agent
    state_size = env.size * env.size
    action_size = 4
    agent = QLearner(state_size, action_size, alpha=0.1, gamma=0.95, epsilon=0.15)
    
    # Train agent
    episode_rewards, episode_lengths, td_errors = train_agent(
        env, agent, num_episodes=10000, max_steps=50
    )
    
    # Plot training progress
    plot_training_progress(episode_rewards, episode_lengths, td_errors)
    
    # Visualize learned value function and policy
    visualize_value_and_policy(env, agent)
    
    # Evaluate from all start positions
    evaluate_all_starts(env, agent, max_steps=50)
    
    print("\nTraining and evaluation complete!")
    print("All results saved to the 'results' directory.")


if __name__ == "__main__":
    main()

