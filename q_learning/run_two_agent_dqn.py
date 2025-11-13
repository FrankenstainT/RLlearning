"""
Run Two-Agent DQN for Pursuer-Evader Game
=========================================
"""

import numpy as np
from typing import Tuple
from pursuer_evader_env import PursuerEvaderEnv
from two_agent_dqn import TwoAgentDQN
from visualization import plot_training_progress, draw_frame, frames_to_gif
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def train_agents(env: PursuerEvaderEnv, agents: TwoAgentDQN, 
                 num_episodes: int = 30000, max_steps: int = 50):
    """Train the two-agent DQN system."""
    pursuer_rewards = []
    evader_rewards = []
    episode_lengths = []
    pursuer_td_errors = []
    evader_td_errors = []
    
    print(f"Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Reset to random distinct positions
        pursuer_pos, evader_pos = env.reset()
        state = env.state_to_features(pursuer_pos, evader_pos)
        
        agents.start_episode()
        pursuer_total_reward = 0
        evader_total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            # Both agents choose actions
            pursuer_action = agents.choose_action(state, 'pursuer', training=True)
            evader_action = agents.choose_action(state, 'evader', training=True)
            
            # Take step
            next_pursuer_pos, next_evader_pos, pursuer_reward, evader_reward, done = \
                env.step(pursuer_pos, pursuer_action, evader_pos, evader_action)
            
            next_state = env.state_to_features(next_pursuer_pos, next_evader_pos)
            
            # Store experiences in respective buffers
            agents.update(state, pursuer_action, pursuer_reward, next_state, done, 'pursuer')
            agents.update(state, evader_action, evader_reward, next_state, done, 'evader')
            
            # Train both networks
            agents.train_step('pursuer')
            agents.train_step('evader')
            
            pursuer_total_reward += pursuer_reward
            evader_total_reward += evader_reward
            steps += 1
            
            pursuer_pos = next_pursuer_pos
            evader_pos = next_evader_pos
            state = next_state
        
        # If episode ended due to max_steps (not caught), add final step with 0 reward
        if not done and steps >= max_steps:
            # Both agents choose actions (though they won't be used)
            pursuer_action = agents.choose_action(state, 'pursuer', training=True)
            evader_action = agents.choose_action(state, 'evader', training=True)
            
            # Final step with 0 reward for both agents
            final_pursuer_reward = 0.0
            final_evader_reward = 0.0
            final_done = True  # Episode is ending
            
            # Store final experiences
            agents.update(state, pursuer_action, final_pursuer_reward, state, final_done, 'pursuer')
            agents.update(state, evader_action, final_evader_reward, state, final_done, 'evader')
            
            # Train both networks on final step
            agents.train_step('pursuer')
            agents.train_step('evader')
            
            pursuer_total_reward += final_pursuer_reward
            evader_total_reward += final_evader_reward
        
        # End episode and get average TD errors
        pursuer_avg_td, evader_avg_td = agents.end_episode()
        
        pursuer_rewards.append(pursuer_total_reward)
        evader_rewards.append(evader_total_reward)
        episode_lengths.append(steps)
        pursuer_td_errors.append(pursuer_avg_td)
        evader_td_errors.append(evader_avg_td)
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Pursuer Avg Reward (last 100): {np.mean(pursuer_rewards[-100:]):.2f}, "
                  f"Evader Avg Reward (last 100): {np.mean(evader_rewards[-100:]):.2f}, "
                  f"Avg Steps: {np.mean(episode_lengths[-100:]):.2f}")
    
    return (pursuer_rewards, evader_rewards, episode_lengths, 
            pursuer_td_errors, evader_td_errors)


def plot_two_agent_training_progress(pursuer_rewards: list, evader_rewards: list,
                                    episode_lengths: list, pursuer_td_errors: list,
                                    evader_td_errors: list, window: int = 100,
                                    save_path: str = 'two_agent_results/two_agent_training_progress.png'):
    """Plot training progress for two agents."""
    from visualization import moving_average
    
    episodes = np.arange(1, len(pursuer_rewards) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Episode rewards - Pursuer
    axes[0, 0].plot(episodes, pursuer_rewards, alpha=0.3, color='red', label='Raw')
    if len(pursuer_rewards) >= window:
        ma_rewards = moving_average(np.array(pursuer_rewards), window)
        ma_episodes = np.arange(window, len(pursuer_rewards) + 1)
        axes[0, 0].plot(ma_episodes, ma_rewards, color='darkred', linewidth=2,
                       label=f'Rolling Avg (window={window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Episode Reward')
    axes[0, 0].set_title('Pursuer Episode Rewards Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode rewards - Evader
    axes[0, 1].plot(episodes, evader_rewards, alpha=0.3, color='blue', label='Raw')
    if len(evader_rewards) >= window:
        ma_rewards = moving_average(np.array(evader_rewards), window)
        ma_episodes = np.arange(window, len(evader_rewards) + 1)
        axes[0, 1].plot(ma_episodes, ma_rewards, color='darkblue', linewidth=2,
                       label=f'Rolling Avg (window={window})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Reward')
    axes[0, 1].set_title('Evader Episode Rewards Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Both rewards together
    axes[0, 2].plot(episodes, pursuer_rewards, alpha=0.3, color='red', label='Pursuer Raw')
    axes[0, 2].plot(episodes, evader_rewards, alpha=0.3, color='blue', label='Evader Raw')
    if len(pursuer_rewards) >= window:
        ma_pursuer = moving_average(np.array(pursuer_rewards), window)
        ma_evader = moving_average(np.array(evader_rewards), window)
        ma_episodes = np.arange(window, len(pursuer_rewards) + 1)
        axes[0, 2].plot(ma_episodes, ma_pursuer, color='darkred', linewidth=2,
                       label='Pursuer Avg')
        axes[0, 2].plot(ma_episodes, ma_evader, color='darkblue', linewidth=2,
                       label='Evader Avg')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Episode Reward')
    axes[0, 2].set_title('Both Agents Rewards Over Time')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # TD Errors - Pursuer
    axes[1, 0].plot(episodes, pursuer_td_errors, alpha=0.5, color='red')
    if len(pursuer_td_errors) >= window:
        ma_td = moving_average(np.array(pursuer_td_errors), window)
        ma_episodes = np.arange(window, len(pursuer_td_errors) + 1)
        axes[1, 0].plot(ma_episodes, ma_td, color='darkred', linewidth=2,
                       label=f'Rolling Avg (window={window})')
        axes[1, 0].legend()
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Average TD Error')
    axes[1, 0].set_title('Pursuer TD Error Per Episode')
    axes[1, 0].grid(True, alpha=0.3)
    
    # TD Errors - Evader
    axes[1, 1].plot(episodes, evader_td_errors, alpha=0.5, color='blue')
    if len(evader_td_errors) >= window:
        ma_td = moving_average(np.array(evader_td_errors), window)
        ma_episodes = np.arange(window, len(evader_td_errors) + 1)
        axes[1, 1].plot(ma_episodes, ma_td, color='darkblue', linewidth=2,
                       label=f'Rolling Avg (window={window})')
        axes[1, 1].legend()
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Average TD Error')
    axes[1, 1].set_title('Evader TD Error Per Episode')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Steps per episode
    axes[1, 2].plot(episodes, episode_lengths, alpha=0.3, color='purple', label='Raw')
    if len(episode_lengths) >= window:
        ma_steps = moving_average(np.array(episode_lengths), window)
        ma_episodes = np.arange(window, len(episode_lengths) + 1)
        axes[1, 2].plot(ma_episodes, ma_steps, color='darkviolet', linewidth=2,
                       label=f'Rolling Avg (window={window})')
        axes[1, 2].legend()
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Steps per Episode')
    axes[1, 2].set_title('Episode Length Over Time')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training progress plots to {save_path}")


def visualize_two_agent_policy(env: PursuerEvaderEnv, agents: TwoAgentDQN,
                               save_path: str = 'two_agent_results/two_agent_policies.png'):
    """Visualize learned policies for both agents."""
    size = env.size
    
    # For each agent, we need to show policy for each pursuer position
    # given a fixed evader position (or vice versa). Let's show pursuer policy
    # with evader at a fixed position, and evader policy with pursuer at a fixed position.
    
    # Pursuer policy: fix evader at (0,0), show pursuer policy at each position
    pursuer_value_map = np.zeros((size, size))
    pursuer_policy_map = np.zeros((size, size), dtype=int)
    fixed_evader = (0, 0)
    
    for y in range(size):
        for x in range(size):
            pursuer_pos = (y, x)
            if pursuer_pos != fixed_evader:
                state = env.state_to_features(pursuer_pos, fixed_evader)
                pursuer_value_map[y, x] = agents.get_value(state, 'pursuer')
                pursuer_policy_map[y, x] = agents.get_policy(state, 'pursuer')
            else:
                pursuer_value_map[y, x] = np.nan
                pursuer_policy_map[y, x] = -1
    
    # Evader policy: fix pursuer at (0,0), show evader policy at each position
    evader_value_map = np.zeros((size, size))
    evader_policy_map = np.zeros((size, size), dtype=int)
    fixed_pursuer = (0, 0)
    
    for y in range(size):
        for x in range(size):
            evader_pos = (y, x)
            if evader_pos != fixed_pursuer:
                state = env.state_to_features(fixed_pursuer, evader_pos)
                evader_value_map[y, x] = agents.get_value(state, 'evader')
                evader_policy_map[y, x] = agents.get_policy(state, 'evader')
            else:
                evader_value_map[y, x] = np.nan
                evader_policy_map[y, x] = -1
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    arrow_dict = {
        0: (0, -0.35, '↑'),  # Up
        1: (0, 0.35, '↓'),   # Down
        2: (-0.35, 0, '←'),  # Left
        3: (0.35, 0, '→')    # Right
    }
    
    # Pursuer value function
    im1 = axes[0, 0].imshow(pursuer_value_map, cmap='Reds', origin='upper')
    axes[0, 0].set_title('Pursuer Value Function\n(Evader fixed at (0,0))', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0])
    for y in range(size):
        for x in range(size):
            if not np.isnan(pursuer_value_map[y, x]):
                axes[0, 0].text(x, y, f'{pursuer_value_map[y, x]:.2f}',
                               ha='center', va='center', color='white', fontweight='bold', fontsize=8)
    
    # Pursuer policy
    axes[0, 1].imshow(np.ones((size, size)), cmap='gray', alpha=0.3, origin='upper')
    axes[0, 1].set_title('Pursuer Policy\n(Evader fixed at (0,0))', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    for y in range(size):
        for x in range(size):
            if (y, x) == fixed_evader:
                axes[0, 1].text(x, y, 'E', ha='center', va='center',
                               fontsize=16, color='blue', fontweight='bold')
            else:
                action = pursuer_policy_map[y, x]
                if action >= 0:
                    dx, dy, _ = arrow_dict[action]
                    axes[0, 1].arrow(x, y, dx, dy, head_width=0.2, head_length=0.2,
                                   fc='red', ec='red', linewidth=2.5, alpha=0.8)
    
    # Evader value function
    im2 = axes[1, 0].imshow(evader_value_map, cmap='Blues', origin='upper')
    axes[1, 0].set_title('Evader Value Function\n(Pursuer fixed at (0,0))',
                        fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1, 0])
    for y in range(size):
        for x in range(size):
            if not np.isnan(evader_value_map[y, x]):
                axes[1, 0].text(x, y, f'{evader_value_map[y, x]:.2f}',
                               ha='center', va='center', color='white', fontweight='bold', fontsize=8)
    
    # Evader policy
    axes[1, 1].imshow(np.ones((size, size)), cmap='gray', alpha=0.3, origin='upper')
    axes[1, 1].set_title('Evader Policy\n(Pursuer fixed at (0,0))',
                        fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    for y in range(size):
        for x in range(size):
            if (y, x) == fixed_pursuer:
                axes[1, 1].text(x, y, 'P', ha='center', va='center',
                               fontsize=16, color='red', fontweight='bold')
            else:
                action = evader_policy_map[y, x]
                if action >= 0:
                    dx, dy, _ = arrow_dict[action]
                    axes[1, 1].arrow(x, y, dx, dy, head_width=0.2, head_length=0.2,
                                   fc='blue', ec='blue', linewidth=2.5, alpha=0.8)
    
    # Set ticks
    for ax in axes.flat:
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.grid(True, color='black', linewidth=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved two-agent policy visualization to {save_path}")


def draw_two_agent_frame(env: PursuerEvaderEnv, pursuer_pos: Tuple[int, int],
                        evader_pos: Tuple[int, int], step_idx: int, out_dir: str,
                        pursuer_action: int = None, evader_action: int = None):
    """Draw a single frame with both agents."""
    os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    size = env.size
    
    # Draw grid
    for y in range(size + 1):
        ax.plot([0, size], [y, y], 'k-', linewidth=2)
    for x in range(size + 1):
        ax.plot([x, x], [0, size], 'k-', linewidth=2)
    
    # Draw pursuer
    pursuer_y, pursuer_x = pursuer_pos
    circle_p = plt.Circle((pursuer_x, pursuer_y), 0.3, color='red', zorder=10)
    ax.add_patch(circle_p)
    ax.text(pursuer_x, pursuer_y, 'P', ha='center', va='center',
           fontsize=16, fontweight='bold', color='white', zorder=11)
    
    # Draw evader
    evader_y, evader_x = evader_pos
    circle_e = plt.Circle((evader_x, evader_y), 0.3, color='blue', zorder=10)
    ax.add_patch(circle_e)
    ax.text(evader_x, evader_y, 'E', ha='center', va='center',
           fontsize=16, fontweight='bold', color='white', zorder=11)
    
    # Draw action arrows
    action_dirs = {
        0: (0, -0.4),  # Up
        1: (0, 0.4),    # Down
        2: (-0.4, 0),   # Left
        3: (0.4, 0)     # Right
    }
    
    if pursuer_action is not None and pursuer_action in action_dirs:
        dx, dy = action_dirs[pursuer_action]
        ax.arrow(pursuer_x, pursuer_y, dx, dy, head_width=0.15, head_length=0.15,
               fc='darkred', ec='darkred', linewidth=3, zorder=9)
    
    if evader_action is not None and evader_action in action_dirs:
        dx, dy = action_dirs[evader_action]
        ax.arrow(evader_x, evader_y, dx, dy, head_width=0.15, head_length=0.15,
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


def evaluate_all_starts(env: PursuerEvaderEnv, agents: TwoAgentDQN, max_steps: int = 50, outdir: str = 'two_agent_results'):
    """Evaluate from all possible start position pairs, record frames, and create GIFs."""
    traces = {}
    eval_dir = os.path.join(outdir, "two_agent_evaluation_traces")
    os.makedirs(eval_dir, exist_ok=True)
    
    start_pairs = env.get_all_start_pairs()
    print(f"\nEvaluating from {len(start_pairs)} possible start position pairs...")
    print(f"Creating frames and GIFs in {eval_dir}...")
    
    for idx, (start_pursuer_pos, start_evader_pos) in enumerate(start_pairs):
        pursuer_pos = start_pursuer_pos
        evader_pos = start_evader_pos
        state = env.state_to_features(pursuer_pos, evader_pos)
        
        # Create directory for this start pair
        case_name = f"eval_P{start_pursuer_pos[0]}_{start_pursuer_pos[1]}_E{start_evader_pos[0]}_{start_evader_pos[1]}"
        frames_dir = os.path.join(eval_dir, case_name)
        os.makedirs(frames_dir, exist_ok=True)
        
        trace = [(pursuer_pos, evader_pos)]
        pursuer_total_reward = 0
        evader_total_reward = 0
        steps = 0
        done = False
        
        # Greedy evaluation (no exploration)
        while not done and steps < max_steps:
            # Choose best actions (greedy) BEFORE drawing frame
            pursuer_action = agents.choose_action(state, 'pursuer', training=False)
            evader_action = agents.choose_action(state, 'evader', training=False)
            
            # Draw frame with the actions that will be taken
            draw_two_agent_frame(env, pursuer_pos, evader_pos, steps, frames_dir,
                               pursuer_action=pursuer_action, evader_action=evader_action)
            
            # Take step
            next_pursuer_pos, next_evader_pos, pursuer_reward, evader_reward, done = \
                env.step(pursuer_pos, pursuer_action, evader_pos, evader_action)
            
            trace.append((next_pursuer_pos, next_evader_pos))
            pursuer_total_reward += pursuer_reward
            evader_total_reward += evader_reward
            steps += 1
            
            pursuer_pos = next_pursuer_pos
            evader_pos = next_evader_pos
            state = env.state_to_features(pursuer_pos, evader_pos)
        
        # Draw final frame
        if done or steps >= max_steps:
            draw_two_agent_frame(env, pursuer_pos, evader_pos, steps, frames_dir)
        
        # Create GIF from frames
        gif_path = os.path.join(frames_dir, f"{case_name}.gif")
        try:
            frames_to_gif(frames_dir, gif_path, fps=3)
        except Exception as e:
            print(f"[GIF creation skipped for {case_name}] {e}")
            gif_path = ""
        
        traces[(start_pursuer_pos, start_evader_pos)] = {
            'trace': trace,
            'pursuer_reward': pursuer_total_reward,
            'evader_reward': evader_total_reward,
            'steps': steps,
            'caught': done and pursuer_pos == evader_pos,
            'frames_dir': frames_dir,
            'gif_path': gif_path
        }
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(start_pairs)} start pairs...")
    
    # Print summary
    caught_count = sum(1 for data in traces.values() if data['caught'])
    print(f"\nEvaluation Summary:")
    print(f"  Total start pairs: {len(traces)}")
    print(f"  Caught: {caught_count}")
    print(f"  Average steps: {np.mean([data['steps'] for data in traces.values()]):.2f}")
    print(f"  Average pursuer reward: {np.mean([data['pursuer_reward'] for data in traces.values()]):.2f}")
    print(f"  Average evader reward: {np.mean([data['evader_reward'] for data in traces.values()]):.2f}")
    print(f"  Frames and GIFs saved to: {eval_dir}")


def main():
    """Main training and evaluation loop."""
    # Set random seed for reproducibility
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    
    # Create environment
    env = PursuerEvaderEnv(size=4)
    
    # Create two-agent DQN system
    agents = TwoAgentDQN(
        input_size=4,  # [pursuer_x, pursuer_y, evader_x, evader_y]
        action_size=4,
        learning_rate=1e-3,
        gamma=0.95,
        epsilon=0.15,
        tau=0.005,
        batch_size=64,
        buffer_size=10000
    )
    
    # Train agents
    (pursuer_rewards, evader_rewards, episode_lengths,
     pursuer_td_errors, evader_td_errors) = train_agents(
        env, agents, num_episodes=30000, max_steps=50
    )
    outdir = 'two_agent_results'
    # Plot training progress
    plot_two_agent_training_progress(
        pursuer_rewards, evader_rewards, episode_lengths,
        pursuer_td_errors, evader_td_errors,
        save_path=os.path.join(outdir, 'two_agent_training_progress.png')
    )
    
    # Visualize learned policies
    visualize_two_agent_policy(env, agents, save_path=os.path.join(outdir, 'two_agent_policies.png'))
    
    # Evaluate from all start position pairs
    evaluate_all_starts(env, agents, max_steps=50, outdir=outdir)
    
    print("\nTraining and evaluation complete!")
    print(f"All results saved to the '{outdir}' directory.")


if __name__ == "__main__":
    main()

