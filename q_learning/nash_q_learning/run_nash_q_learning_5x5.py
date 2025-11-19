"""
Run Pure Nash Q-Learning for Competitive Pursuer-Evader Game (5x5 with Obstacles and Goal)
===========================================================================================
"""

import numpy as np
from typing import Tuple, List, Dict
import sys
import os
import time
import argparse
import json
import datetime

# Add parent directory to path to import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from competitive_env_5x5 import CompetitiveEnv5x5
from nash_q_learning import NashQLearning
from visualization import frames_to_gif
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def moving_average(data: np.ndarray, window: int = 100) -> np.ndarray:
    """Compute moving average."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_and_save(xs, ys, title, xlabel, ylabel, outpath):
    """Plot and save a figure."""
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def policy_entropy(p: np.ndarray) -> float:
    """Compute policy entropy."""
    p = np.asarray(p, float)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def train_agent(env: CompetitiveEnv5x5, agent: NashQLearning, 
                num_episodes: int = 30000, max_steps: int = 50,
                log_interval: int = 500,
                track_policy_drift: bool = True,
                print_interval: int | None = None):
    """Train the Nash Q-Learning agent."""
    pursuer_rewards = []
    evader_rewards = []
    episode_lengths = []
    td_errors_per_episode = []
    episode_winners = []
    policy_entropies = []
    value_diffs_max = []
    value_diffs_mean = []
    policy_l1_diffs = []
    
    # Get all possible states for drift tracking
    all_states = []
    for pursuer_pos in env.valid_positions:
        for evader_pos in env.valid_positions:
            if pursuer_pos != evader_pos and evader_pos != env.evader_goal:
                all_states.append((pursuer_pos, evader_pos))
    
    prev_snapshot = None
    if print_interval is None:
        print_interval = max(1, log_interval) if log_interval else 100
    
    print(f"Training for {num_episodes} episodes...")
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Reset to random distinct positions
        pursuer_pos, evader_pos = env.reset()
        
        agent.start_episode()
        pursuer_total_reward = 0
        evader_total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            # Choose joint action
            pursuer_action, evader_action = agent.choose_joint_action(
                pursuer_pos, evader_pos, training=True)
            
            # Take step
            next_pursuer_pos, next_evader_pos, pursuer_reward, evader_reward, done = \
                env.step(pursuer_pos, pursuer_action, evader_pos, evader_action)
            
            # Update Q-values
            agent.update(pursuer_pos, evader_pos, pursuer_action, evader_action,
                        pursuer_reward, evader_reward,
                        next_pursuer_pos, next_evader_pos, done)
            
            pursuer_total_reward += pursuer_reward
            evader_total_reward += evader_reward
            steps += 1
            
            pursuer_pos = next_pursuer_pos
            evader_pos = next_evader_pos
        
        # End episode and get average TD error
        avg_td = agent.end_episode()
        
        pursuer_rewards.append(pursuer_total_reward)
        evader_rewards.append(evader_total_reward)
        episode_lengths.append(steps)
        td_errors_per_episode.append(avg_td)
        
        # Track winners
        if done:
            # Check if evader reached goal or pursuer caught evader
            if evader_pos == env.evader_goal:
                episode_winners.append("Evader")
            else:
                episode_winners.append("Pursuer")
        else:
            episode_winners.append("Timeout")
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Track policy entropy and drift (less frequently to save time)
        if track_policy_drift and log_interval and (episode+1) % log_interval == 0:
            # Compute average policy entropy
            entropies = []
            for pursuer_pos_state, evader_pos_state in all_states:
                policy = agent.get_policy_distribution(pursuer_pos_state, evader_pos_state)
                entropies.append(policy_entropy(policy))
            policy_entropies.append(np.mean(entropies))
            
            # Track value and policy drift
            current_snapshot = agent.snapshot_policies_and_values(env, all_states)
            
            if prev_snapshot is not None:
                # Value differences
                value_diffs = []
                for state_key in current_snapshot['values']:
                    if state_key in prev_snapshot['values']:
                        diff = abs(current_snapshot['values'][state_key] - 
                                  prev_snapshot['values'][state_key])
                        value_diffs.append(diff)
                
                if value_diffs:
                    value_diffs_max.append(np.max(value_diffs))
                    value_diffs_mean.append(np.mean(value_diffs))
                
                # Policy L1 differences
                policy_l1_diffs_ep = []
                for state_key in current_snapshot['policies']:
                    if state_key in prev_snapshot['policies']:
                        l1 = np.sum(np.abs(current_snapshot['policies'][state_key] - 
                                          prev_snapshot['policies'][state_key]))
                        policy_l1_diffs_ep.append(l1)
                
                if policy_l1_diffs_ep:
                    policy_l1_diffs.append(np.mean(policy_l1_diffs_ep))
            else:
                value_diffs_max.append(0.0)
                value_diffs_mean.append(0.0)
                policy_l1_diffs.append(0.0)
            
            prev_snapshot = current_snapshot
        
        if (episode + 1) % print_interval == 0:
            pursuer_wins = sum(1 for w in episode_winners[-print_interval:] if w == "Pursuer")
            evader_wins = sum(1 for w in episode_winners[-print_interval:] if w == "Evader")
            elapsed_time = time.time() - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Get current learning rate
            current_lr = agent._get_learning_rate()
            
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Pursuer Avg Reward (last {print_interval}): {np.mean(pursuer_rewards[-print_interval:]):.2f}, "
                  f"Evader Avg Reward (last {print_interval}): {np.mean(evader_rewards[-print_interval:]):.2f}, "
                  f"Avg Steps: {np.mean(episode_lengths[-print_interval:]):.2f}, "
                  f"Pursuer Win Rate (last {print_interval}): {pursuer_wins/print_interval:.2%}, "
                  f"Evader Win Rate (last {print_interval}): {evader_wins/print_interval:.2%}, "
                  f"Epsilon: {agent.epsilon:.4f}, "
                  f"LR: {current_lr:.6f}, "
                  f"Time: {time_str}")
    
    return (pursuer_rewards, evader_rewards, episode_lengths, td_errors_per_episode,
            episode_winners, policy_entropies, value_diffs_max, value_diffs_mean, 
            policy_l1_diffs)


def plot_training_progress(pursuer_rewards: list, evader_rewards: list,
                          episode_lengths: list, td_errors: list,
                          episode_winners: list, policy_entropies: list,
                          value_diffs_max: list, value_diffs_mean: list,
                          policy_l1_diffs: list, outdir: str = 'nash_q_learning_results_5x5',
                          log_interval: int = 500):
    """Plot all training progress metrics."""
    os.makedirs(outdir, exist_ok=True)
    episodes = np.arange(1, len(pursuer_rewards) + 1)
    
    # Episode rewards - Pursuer
    plot_and_save(episodes, pursuer_rewards,
                 "Pursuer Episode Rewards", "Episode", "Reward",
                 os.path.join(outdir, "pursuer_rewards.png"))
    window = 100
    if len(pursuer_rewards) >= window:
        ma_pursuer = moving_average(np.array(pursuer_rewards), window)
        ma_episodes = np.arange(window, len(pursuer_rewards) + 1)
        plot_and_save(ma_episodes, ma_pursuer,
                     f"Pursuer Episode Rewards (MA={window})", "Episode", "Reward",
                     os.path.join(outdir, f"pursuer_rewards_ma{window}.png"))
    
    # Episode rewards - Evader
    plot_and_save(episodes, evader_rewards,
                 "Evader Episode Rewards", "Episode", "Reward",
                 os.path.join(outdir, "evader_rewards.png"))
    if len(evader_rewards) >= window:
        ma_evader = moving_average(np.array(evader_rewards), window)
        ma_episodes = np.arange(window, len(evader_rewards) + 1)
        plot_and_save(ma_episodes, ma_evader,
                     f"Evader Episode Rewards (MA={window})", "Episode", "Reward",
                     os.path.join(outdir, f"evader_rewards_ma{window}.png"))
    
    # TD Errors
    plot_and_save(episodes, td_errors,
                 "TD Error Per Episode", "Episode", "TD Error",
                 os.path.join(outdir, "td_error_per_episode.png"))
    if len(td_errors) >= window:
        ma_td = moving_average(np.array(td_errors), window)
        ma_episodes = np.arange(window, len(td_errors) + 1)
        plot_and_save(ma_episodes, ma_td,
                     f"TD Error Per Episode (MA={window})", "Episode", "TD Error",
                     os.path.join(outdir, f"td_error_ma{window}.png"))
    
    # Steps per episode
    plot_and_save(episodes, episode_lengths,
                 "Episode Lengths", "Episode", "Steps",
                 os.path.join(outdir, "episode_lengths.png"))
    
    # Win rate
    pursuer_wins = np.array([1 if w == "Pursuer" else 0 for w in episode_winners], float)
    evader_wins = np.array([1 if w == "Evader" else 0 for w in episode_winners], float)
    if len(pursuer_wins) >= window:
        ma_pursuer_wins = moving_average(pursuer_wins, window)
        ma_evader_wins = moving_average(evader_wins, window)
        ma_episodes = np.arange(window, len(pursuer_wins) + 1)
        plot_and_save(ma_episodes, ma_pursuer_wins,
                     f"Pursuer Win Rate (MA={window})", "Episode", "Win Rate",
                     os.path.join(outdir, "pursuer_win_rate.png"))
        plot_and_save(ma_episodes, ma_evader_wins,
                     f"Evader Win Rate (MA={window})", "Episode", "Win Rate",
                     os.path.join(outdir, "evader_win_rate.png"))
    
    # Policy entropy (tracked every log_interval episodes)
    if policy_entropies:
        entropy_episodes = np.arange(0, len(policy_entropies)) * log_interval
        plot_and_save(entropy_episodes, policy_entropies,
                     "Average Policy Entropy", "Episode", "Entropy",
                     os.path.join(outdir, "policy_entropy.png"))
    
    # Value differences (tracked every log_interval episodes)
    if value_diffs_max:
        diff_episodes = np.arange(0, len(value_diffs_max)) * log_interval
        plot_and_save(diff_episodes, value_diffs_max,
                     "Max Value Difference Between Steps", "Episode", "Max |ΔV|",
                     os.path.join(outdir, "value_diff_max.png"))
        plot_and_save(diff_episodes, value_diffs_mean,
                     "Mean Value Difference Between Steps", "Episode", "Mean |ΔV|",
                     os.path.join(outdir, "value_diff_mean.png"))
    
    # Policy L1 differences (tracked every log_interval episodes)
    if policy_l1_diffs:
        diff_episodes = np.arange(0, len(policy_l1_diffs)) * log_interval
        plot_and_save(diff_episodes, policy_l1_diffs,
                     "Mean L1 Norm of Policy Differences", "Episode", "Mean L1(π)",
                     os.path.join(outdir, "policy_l1_diff.png"))


def visualize_policies(env: CompetitiveEnv5x5, agent: NashQLearning,
                      outdir: str = 'nash_q_learning_results_5x5'):
    """Visualize learned policies for pursuer and evader."""
    os.makedirs(outdir, exist_ok=True)
    size = env.size
    
    # Create policy maps for pursuer and evader
    # Fix one agent position and show policy for the other
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    arrow_dict = {
        0: (0, -0.3, '↑'),  # Up
        1: (0, 0.3, '↓'),   # Down
        2: (-0.3, 0, '←'),  # Left
        3: (0.3, 0, '→')    # Right
    }
    
    # Pursuer policy: fix evader at (2,2) if valid, else use first valid position
    fixed_evader = (2, 2) if (2, 2) not in env.obstacles and (2, 2) != env.evader_goal else env.valid_positions[0]
    pursuer_value_map = np.zeros((size, size))
    pursuer_policy_map = np.zeros((size, size), dtype=int)
    
    for pursuer_y in range(size):
        for pursuer_x in range(size):
            pursuer_pos = (pursuer_y, pursuer_x)
            if pursuer_pos not in env.obstacles and pursuer_pos != fixed_evader:
                pursuer_value_map[pursuer_y, pursuer_x] = agent.get_value(pursuer_pos, fixed_evader)
                pursuer_action, _ = agent.get_policy(pursuer_pos, fixed_evader)
                pursuer_policy_map[pursuer_y, pursuer_x] = pursuer_action
            else:
                pursuer_value_map[pursuer_y, pursuer_x] = np.nan
                pursuer_policy_map[pursuer_y, pursuer_x] = -1
    
    # Evader policy: fix pursuer at (0,0) for consistent visualization
    # This matches the user's observation about states (0,0,2,4) and (0,0,3,4)
    fixed_pursuer = (0, 0)
    evader_value_map = np.zeros((size, size))
    evader_policy_map = np.zeros((size, size), dtype=int)
    
    for evader_y in range(size):
        for evader_x in range(size):
            evader_pos = (evader_y, evader_x)
            if evader_pos not in env.obstacles and evader_pos != fixed_pursuer and evader_pos != env.evader_goal:
                evader_value_map[evader_y, evader_x] = agent.get_value(fixed_pursuer, evader_pos)
                _, evader_action = agent.get_policy(fixed_pursuer, evader_pos)
                evader_policy_map[evader_y, evader_x] = evader_action
            else:
                evader_value_map[evader_y, evader_x] = np.nan
                evader_policy_map[evader_y, evader_x] = -1
    
    # Plot pursuer value
    im1 = axes[0, 0].imshow(pursuer_value_map, cmap='Reds', origin='upper')
    axes[0, 0].set_title(f'Pursuer Value Function\n(Evader fixed at {fixed_evader})', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot pursuer policy
    axes[0, 1].imshow(np.ones((size, size)), cmap='gray', alpha=0.3, origin='upper')
    axes[0, 1].set_title(f'Pursuer Policy\n(Evader fixed at {fixed_evader})', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    for y in range(size):
        for x in range(size):
            if (y, x) in env.obstacles:
                axes[0, 1].text(x, y, '█', ha='center', va='center',
                               fontsize=16, color='black', fontweight='bold')
            elif (y, x) == fixed_evader:
                axes[0, 1].text(x, y, 'E', ha='center', va='center',
                               fontsize=16, color='blue', fontweight='bold')
            elif (y, x) == env.evader_goal:
                axes[0, 1].text(x, y, 'G', ha='center', va='center',
                               fontsize=16, color='green', fontweight='bold')
            else:
                action = pursuer_policy_map[y, x]
                if action >= 0:
                    dx, dy, _ = arrow_dict[action]
                    axes[0, 1].arrow(x, y, dx, dy, head_width=0.15, head_length=0.15,
                                   fc='red', ec='red', linewidth=2, alpha=0.8)
    
    # Plot evader value
    im2 = axes[1, 0].imshow(evader_value_map, cmap='Blues', origin='upper')
    axes[1, 0].set_title(f'Evader Value Function\n(Pursuer fixed at {fixed_pursuer})',
                        fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Plot evader policy
    axes[1, 1].imshow(np.ones((size, size)), cmap='gray', alpha=0.3, origin='upper')
    axes[1, 1].set_title(f'Evader Policy\n(Pursuer fixed at {fixed_pursuer})',
                        fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    for y in range(size):
        for x in range(size):
            if (y, x) in env.obstacles:
                axes[1, 1].text(x, y, '█', ha='center', va='center',
                               fontsize=16, color='black', fontweight='bold')
            elif (y, x) == fixed_pursuer:
                axes[1, 1].text(x, y, 'P', ha='center', va='center',
                               fontsize=16, color='red', fontweight='bold')
            elif (y, x) == env.evader_goal:
                axes[1, 1].text(x, y, 'G', ha='center', va='center',
                               fontsize=16, color='green', fontweight='bold')
            else:
                action = evader_policy_map[y, x]
                if action >= 0:
                    dx, dy, _ = arrow_dict[action]
                    axes[1, 1].arrow(x, y, dx, dy, head_width=0.15, head_length=0.15,
                                   fc='blue', ec='blue', linewidth=2, alpha=0.8)
    
    # Set ticks
    for ax in axes.flat:
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.grid(True, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'policies.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved policy visualization to {os.path.join(outdir, 'policies.png')}")


def draw_frame(env: CompetitiveEnv5x5, pursuer_pos: Tuple[int, int],
              evader_pos: Tuple[int, int], step_idx: int, out_dir: str,
              pursuer_action: int = None, evader_action: int = None):
    """Draw a single frame with both agents, obstacles, and goal."""
    os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    size = env.size
    
    # Draw grid
    for y in range(size + 1):
        ax.plot([0, size], [y, y], 'k-', linewidth=2)
    for x in range(size + 1):
        ax.plot([x, x], [0, size], 'k-', linewidth=2)
    
    # Draw obstacles
    for obs_y, obs_x in env.obstacles:
        rect = patches.Rectangle((obs_x - 0.5, obs_y - 0.5), 1, 1,
                               linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
        ax.add_patch(rect)
        ax.text(obs_x, obs_y, '█', ha='center', va='center',
               fontsize=20, color='black', fontweight='bold')
    
    # Draw evader goal
    goal_y, goal_x = env.evader_goal
    rect = patches.Rectangle((goal_x - 0.5, goal_y - 0.5), 1, 1,
                           linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.8)
    ax.add_patch(rect)
    ax.text(goal_x, goal_y, 'G', ha='center', va='center',
           fontsize=18, color='darkgreen', fontweight='bold')
    
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


def evaluate_all_starts(env: CompetitiveEnv5x5, agent: NashQLearning, 
                        max_steps: int = 50, outdir: str = 'nash_q_learning_results_5x5'):
    """Evaluate from all possible start position pairs, record frames, and create GIFs."""
    traces = {}
    eval_dir = os.path.join(outdir, "evaluation_traces")
    os.makedirs(eval_dir, exist_ok=True)
    
    start_pairs = env.get_all_start_pairs()
    print(f"\nEvaluating from {len(start_pairs)} possible start position pairs...")
    print(f"Creating frames and GIFs in {eval_dir}...")
    
    pursuer_rewards_eval = []
    evader_rewards_eval = []
    
    for idx, (start_pursuer_pos, start_evader_pos) in enumerate(start_pairs):
        pursuer_pos = start_pursuer_pos
        evader_pos = start_evader_pos
        
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
            # Choose best joint action (greedy) BEFORE drawing frame
            pursuer_action, evader_action = agent.choose_joint_action(
                pursuer_pos, evader_pos, training=False)
            
            # Draw frame with the actions that will be taken
            draw_frame(env, pursuer_pos, evader_pos, steps, frames_dir,
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
        
        # Draw final frame
        if done or steps >= max_steps:
            draw_frame(env, pursuer_pos, evader_pos, steps, frames_dir)
        
        # Create GIF from frames
        gif_path = os.path.join(frames_dir, f"{case_name}.gif")
        try:
            frames_to_gif(frames_dir, gif_path, fps=3)
        except Exception as e:
            print(f"[GIF creation skipped for {case_name}] {e}")
            gif_path = ""
        
        # Determine outcome
        caught = done and pursuer_pos == evader_pos
        evader_won = done and evader_pos == env.evader_goal
        
        traces[(start_pursuer_pos, start_evader_pos)] = {
            'trace': trace,
            'pursuer_reward': pursuer_total_reward,
            'evader_reward': evader_total_reward,
            'steps': steps,
            'caught': caught,
            'evader_won': evader_won,
            'frames_dir': frames_dir,
            'gif_path': gif_path
        }
        
        pursuer_rewards_eval.append(pursuer_total_reward)
        evader_rewards_eval.append(evader_total_reward)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(start_pairs)} start pairs...")
    
    # Plot distribution of final rewards
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(pursuer_rewards_eval, bins=30, alpha=0.7, color='red', edgecolor='black')
    plt.title('Distribution of Pursuer Final Rewards')
    plt.xlabel('Final Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(evader_rewards_eval, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution of Evader Final Rewards')
    plt.xlabel('Final Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'reward_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    caught_count = sum(1 for data in traces.values() if data['caught'])
    evader_won_count = sum(1 for data in traces.values() if data['evader_won'])
    print(f"\nEvaluation Summary:")
    print(f"  Total start pairs: {len(traces)}")
    print(f"  Pursuer wins (caught): {caught_count}")
    print(f"  Evader wins (reached goal): {evader_won_count}")
    print(f"  Timeouts: {len(traces) - caught_count - evader_won_count}")
    print(f"  Average steps: {np.mean([data['steps'] for data in traces.values()]):.2f}")
    print(f"  Average pursuer reward: {np.mean(pursuer_rewards_eval):.2f}")
    print(f"  Average evader reward: {np.mean(evader_rewards_eval):.2f}")
    print(f"  Frames and GIFs saved to: {eval_dir}")
    print(f"  Reward distribution plot saved to: {os.path.join(outdir, 'reward_distribution.png')}")


def plot_visit_frequencies(agent: NashQLearning, env: CompetitiveEnv5x5, outdir: str):
    """Plot visualization of state visit frequencies."""
    os.makedirs(outdir, exist_ok=True)
    
    # Collect visit frequencies for all states
    visit_data = []
    for pursuer_pos in env.valid_positions:
        for evader_pos in env.valid_positions:
            if pursuer_pos != evader_pos and evader_pos != env.evader_goal:
                # Create state key manually (same format as in NashQLearning)
                pursuer_y, pursuer_x = pursuer_pos
                evader_y, evader_x = evader_pos
                state_key = (pursuer_y, pursuer_x, evader_y, evader_x)
                visit_count = agent.visit_counts.get(state_key, 0)
                visit_data.append(visit_count)
    
    visit_data = np.array(visit_data)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram of visit frequencies
    axes[0, 0].hist(visit_data, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Visit Count')
    axes[0, 0].set_ylabel('Number of States')
    axes[0, 0].set_title('Distribution of State Visit Frequencies')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Log-scale histogram (to see distribution better)
    non_zero_visits = visit_data[visit_data > 0]
    if len(non_zero_visits) > 0:
        axes[0, 1].hist(non_zero_visits, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Visit Count (log scale)')
        axes[0, 1].set_ylabel('Number of States')
        axes[0, 1].set_title('Distribution of Visited States (Non-zero)')
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No visited states', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Distribution of Visited States (Non-zero)')
    
    # 3. Cumulative distribution
    sorted_visits = np.sort(visit_data)[::-1]  # Sort descending
    cumulative = np.cumsum(sorted_visits)
    cumulative_pct = cumulative / cumulative[-1] * 100 if cumulative[-1] > 0 else cumulative
    axes[1, 0].plot(range(len(sorted_visits)), cumulative_pct, linewidth=2)
    axes[1, 0].set_xlabel('State Rank (by visit count)')
    axes[1, 0].set_ylabel('Cumulative % of Total Visits')
    axes[1, 0].set_title('Cumulative Visit Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Statistics text
    stats_text = f"""Visit Frequency Statistics:
    
Total States: {len(visit_data)}
Visited States: {np.sum(visit_data > 0)}
Unvisited States: {np.sum(visit_data == 0)}

Total Visits: {np.sum(visit_data):,}
Mean Visits: {np.mean(visit_data):.2f}
Median Visits: {np.median(visit_data):.2f}
Min Visits: {np.min(visit_data)}
Max Visits: {np.max(visit_data):,}
Std Dev: {np.std(visit_data):.2f}

Coverage: {np.sum(visit_data > 0) / len(visit_data) * 100:.2f}%
"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, 
                   verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Statistics Summary')
    
    plt.tight_layout()
    visit_plot_path = os.path.join(outdir, 'visit_frequencies.png')
    plt.savefig(visit_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visit frequency visualization to {visit_plot_path}")


def save_parameters(params: Dict, outdir: str):
    """Save all parameters to a JSON file."""
    params_file = os.path.join(outdir, 'parameters.json')
    # Add timestamp
    params['run_info'] = {
        'timestamp': datetime.datetime.now().isoformat(),
        'timestamp_readable': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Parameters saved to {params_file}")


def main():
    """Main training and evaluation loop."""
    parser = argparse.ArgumentParser(description='Train or evaluate Pure Nash Q-Learning agent (5x5 with obstacles and goal)')
    parser.add_argument('--outdir', type=str, default='nash_q_learning_results_5x5',
                        help='Output directory for results')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training (only evaluate)')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    
    # Environment parameters
    env_size = 5
    obstacles = [(1, 1), (2, 2), (3, 3)]
    evader_goal = (0, 4)
    
    # Agent hyperparameters
    state_size = 25  # 5x5 grid
    joint_action_size = 16  # 4 × 4 joint actions
    learning_rate = 0.1  # Initial learning rate (will decrease)
    gamma = 0.95
    epsilon_start = 0.2
    epsilon_end = 0.05
    epsilon_decay = 0.9999
    
    # Improved learning parameters
    use_decreasing_lr = False  # Use decreasing learning rate for convergence
    # Optimistic initialization: use small value to encourage exploration without biasing Nash equilibrium
    # Too high (e.g., 1.0) causes constant Q-value matrices, leading to uniform Nash policies
    # Value should be small relative to reward scale (catch reward = 10.0)
    optimistic_init = 0.1  # Small optimistic value to encourage exploration
    
    # Training parameters
    num_episodes = 60000
    max_steps = 50
    
    # Create environment
    env = CompetitiveEnv5x5(size=env_size, obstacles=obstacles, evader_goal=evader_goal)
    
    # Create Nash Q-Learning agent with improved settings
    agent = NashQLearning(
        state_size=state_size,
        joint_action_size=joint_action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        use_decreasing_lr=use_decreasing_lr,
        optimistic_init=optimistic_init
    )
    
    # Collect all parameters
    parameters = {
        'random_seed': random_seed,
        'environment': {
            'size': env_size,
            'obstacles': obstacles,
            'evader_goal': evader_goal,
        },
        'agent': {
            'state_size': state_size,
            'joint_action_size': joint_action_size,
            'learning_rate': learning_rate,
            'use_decreasing_lr': use_decreasing_lr,
            'optimistic_init': optimistic_init,
            'gamma': gamma,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay,
        },
        'training': {
            'num_episodes': num_episodes,
            'max_steps': max_steps,
        },
        'algorithm': {
            'type': 'Pure Nash Q-Learning',
            'nash_solver': 'Linear Programming (scipy.optimize.linprog)',
        }
    }
    
    # Save parameters
    save_parameters(parameters, outdir)
    
    log_interval = 100
    
    # Train agent (skip if skip-training is set)
    if args.skip_training:
        print("Skipping training (--skip-training set)")
        pursuer_rewards, evader_rewards, episode_lengths, td_errors = [], [], [], []
        episode_winners, policy_entropies, value_diffs_max, value_diffs_mean, policy_l1_diffs = [], [], [], [], []
    else:
        # Train agent
        (pursuer_rewards, evader_rewards, episode_lengths, td_errors,
         episode_winners, policy_entropies, value_diffs_max, value_diffs_mean,
         policy_l1_diffs) = train_agent(
            env,
            agent,
            num_episodes=num_episodes,
            max_steps=max_steps,
            log_interval=log_interval,
            track_policy_drift=True,
            print_interval=100
        )
        
        # Plot training progress
        if pursuer_rewards:  # Only plot if we have training data
            plot_training_progress(pursuer_rewards, evader_rewards, episode_lengths, td_errors,
                                  episode_winners, policy_entropies, value_diffs_max, 
                                  value_diffs_mean, policy_l1_diffs, outdir=outdir,
                                  log_interval=log_interval)
        
        # Save final Q-table
        q_table_path = os.path.join(outdir, 'final_q_table.json')
        agent.save_q_table(q_table_path)
        
        # Save visit frequencies
        visit_freq_path = os.path.join(outdir, 'visit_frequencies.json')
        agent.save_visit_frequencies(visit_freq_path)
        
        # Save final policies and values
        policies_values_path = os.path.join(outdir, 'final_policies_and_values.json')
        agent.save_policies_and_values(env, policies_values_path)
        
        # Analyze state coverage
        coverage_info = agent.analyze_state_coverage(env)
        print(f"\nQ-table Coverage Analysis:")
        print(f"  Total states: {coverage_info['total_states']}")
        print(f"  Visited states: {coverage_info['visited_states']}")
        print(f"  Unvisited states: {coverage_info['unvisited_states']}")
        print(f"  Coverage: {coverage_info['coverage']:.2%}")
        if coverage_info['unvisited_states'] > 0:
            print(f"  Sample unvisited states: {coverage_info['unvisited_state_list']}")
        
        # Print visit statistics
        visit_stats = coverage_info.get('visit_statistics', {})
        if visit_stats:
            print(f"\nVisit Frequency Statistics:")
            print(f"  Total visits: {visit_stats.get('total_visits', 0):,}")
            print(f"  Mean visits per state: {visit_stats.get('mean_visits', 0):.2f}")
            print(f"  Median visits per state: {visit_stats.get('median_visits', 0):.2f}")
            print(f"  Min visits: {visit_stats.get('min_visits', 0)}")
            print(f"  Max visits: {visit_stats.get('max_visits', 0):,}")
            print(f"  Std dev: {visit_stats.get('std_visits', 0):.2f}")
        
        # Print state-action coverage
        sa_coverage = coverage_info.get('state_action_coverage', {})
        if sa_coverage:
            print(f"\nState-Action Coverage:")
            print(f"  Total state-action pairs: {sa_coverage.get('total_state_actions', 0):,}")
            print(f"  Visited state-action pairs: {sa_coverage.get('visited_state_actions', 0):,}")
            print(f"  Coverage: {sa_coverage.get('coverage', 0):.2%}")
            print(f"  Mean visits per state-action: {sa_coverage.get('mean_visits_per_sa', 0):.2f}")
            print(f"  Median visits per state-action: {sa_coverage.get('median_visits_per_sa', 0):.2f}")
            print(f"  Min visits: {sa_coverage.get('min_visits_per_sa', 0)}")
            print(f"  Max visits: {sa_coverage.get('max_visits_per_sa', 0):,}")
        
        # Print learning rate info
        if agent.use_decreasing_lr:
            current_lr = agent._get_learning_rate()
            print(f"\nLearning Rate:")
            print(f"  Initial learning rate: {agent.learning_rate_initial:.6f}")
            print(f"  Current learning rate: {current_lr:.6f}")
            print(f"  Total updates: {agent.total_updates:,}")
            print(f"  Learning rate decay: α_t = α₀ / (1 + t/10000)")
        
        # Save coverage info
        coverage_path = os.path.join(outdir, 'state_coverage.json')
        import json
        coverage_serializable = {
            'total_states': coverage_info['total_states'],
            'visited_states': coverage_info['visited_states'],
            'unvisited_states': coverage_info['unvisited_states'],
            'coverage': coverage_info['coverage'],
            'unvisited_state_list': [f"P{pos[0]}_{pos[1]}_E{pos[2]}_{pos[3]}" 
                                    for pos in coverage_info['unvisited_state_list']],
            'visit_statistics': visit_stats,
            'state_action_coverage': sa_coverage,
            'learning_rate_info': {
                'use_decreasing_lr': agent.use_decreasing_lr,
                'initial_lr': agent.learning_rate_initial,
                'current_lr': agent._get_learning_rate() if agent.use_decreasing_lr else agent.learning_rate_initial,
                'total_updates': agent.total_updates
            } if hasattr(agent, 'use_decreasing_lr') else {}
        }
        with open(coverage_path, 'w') as f:
            json.dump(coverage_serializable, f, indent=2)
        print(f"  Saved coverage analysis to {coverage_path}")
        
        # Create visualization of visit frequencies
        plot_visit_frequencies(agent, env, outdir)
    
    # Visualize and evaluate
    visualize_policies(env, agent, outdir=outdir)
    evaluate_all_starts(env, agent, max_steps=50, outdir=outdir)
    
    print("\nTraining and evaluation complete!")
    print(f"All results saved to the '{outdir}' directory.")


if __name__ == "__main__":
    main()

