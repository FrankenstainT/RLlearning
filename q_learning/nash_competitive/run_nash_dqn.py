"""
Run Nash DQN for Competitive Pursuer-Evader Game
=================================================
"""

import numpy as np
from typing import Tuple, List, Dict
import sys
import os
import time
# Add parent directory to path to import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from competitive_env import CompetitiveEnv
from nash_dqn import NashDQN, IS_WINDOWS, get_has_multiprocessing, get_cuda_available
from visualization import frames_to_gif
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import json


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


def train_agent(env: CompetitiveEnv, agent: NashDQN, 
                num_episodes: int = 30000, max_steps: int = 50,
                log_interval: int = 500,
                track_policy_drift: bool = True,
                print_interval: int | None = None):
    """Train the Nash DQN agent."""
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
            if pursuer_pos != evader_pos:
                state = env.state_to_features(pursuer_pos, evader_pos)
                all_states.append(state)
    
    prev_snapshot = None
    if print_interval is None:
        print_interval = max(1, log_interval) if log_interval else 100
    
    print(f"Training for {num_episodes} episodes...")
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Reset to random distinct positions
        pursuer_pos, evader_pos = env.reset()
        state = env.state_to_features(pursuer_pos, evader_pos)
        
        agent.start_episode()
        pursuer_total_reward = 0
        evader_total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            # Choose joint action
            pursuer_action, evader_action = agent.choose_joint_action(state, training=True)
            joint_action_idx = env.joint_action_to_index(pursuer_action, evader_action)
            
            # Take step
            next_pursuer_pos, next_evader_pos, pursuer_reward, evader_reward, done = \
                env.step(pursuer_pos, pursuer_action, evader_pos, evader_action)
            
            next_state = env.state_to_features(next_pursuer_pos, next_evader_pos)
            
            # Store experience (use pursuer reward for Q-learning, but track both)
            agent.update(state, joint_action_idx, pursuer_reward, next_state, done)
            
            # Train the network
            agent.train_step()
            
            pursuer_total_reward += pursuer_reward
            evader_total_reward += evader_reward
            steps += 1
            
            pursuer_pos = next_pursuer_pos
            evader_pos = next_evader_pos
            state = next_state
        
        # If episode ended due to max_steps, add final step with 0 reward
        # if not done and steps >= max_steps:
        #     pursuer_action, evader_action = agent.choose_joint_action(state, training=True)
        #     joint_action_idx = env.joint_action_to_index(pursuer_action, evader_action)
            
        #     final_reward = 0.0
        #     final_done = True
            
        #     agent.update(state, joint_action_idx, final_reward, state, final_done)
        #     agent.train_step()
            
        #     pursuer_total_reward += final_reward
        #     evader_total_reward += final_reward
        
        # End episode and get average TD error
        avg_td = agent.end_episode()
        
        pursuer_rewards.append(pursuer_total_reward)
        evader_rewards.append(evader_total_reward)
        episode_lengths.append(steps)
        td_errors_per_episode.append(avg_td)
        
        # Track winners
        if done:
            episode_winners.append("Pursuer")
        else:
            episode_winners.append("Timeout")
        
        # Decay epsilon
        agent.decay_epsilon()
        # Track policy entropy and drift (less frequently to save time)
        if track_policy_drift and log_interval and (episode+1) % log_interval == 0:
            # Compute average policy entropy
            entropies = []
            for state in all_states:  # Reduced sample size for efficiency
                policy = agent.get_policy_distribution(state)
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
            elapsed_time = time.time() - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Pursuer Avg Reward (last {print_interval}): {np.mean(pursuer_rewards[-print_interval:]):.2f}, "
                  f"Evader Avg Reward (last {print_interval}): {np.mean(evader_rewards[-print_interval:]):.2f}, "
                  f"Avg Steps: {np.mean(episode_lengths[-print_interval:]):.2f}, "
                  f"Pursuer Win Rate (last {print_interval}): {pursuer_wins/print_interval:.2%}, "
                  f"Epsilon: {agent.epsilon:.4f}, "
                  f"Time: {time_str}")
    
    return (pursuer_rewards, evader_rewards, episode_lengths, td_errors_per_episode,
            episode_winners, policy_entropies, value_diffs_max, value_diffs_mean, 
            policy_l1_diffs)


def plot_training_progress(pursuer_rewards: list, evader_rewards: list,
                          episode_lengths: list, td_errors: list,
                          episode_winners: list, policy_entropies: list,
                          value_diffs_max: list, value_diffs_mean: list,
                          policy_l1_diffs: list, outdir: str = 'nash_results'):
    """Plot all training progress metrics."""
    os.makedirs(outdir, exist_ok=True)
    episodes = np.arange(1, len(pursuer_rewards) + 1)
    
    # Episode rewards - Pursuer
    plot_and_save(episodes, pursuer_rewards,
                 "Pursuer Episode Rewards", "Episode", "Reward",
                 os.path.join(outdir, "pursuer_rewards.png"))
    window = 20
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
    if len(pursuer_wins) >= window:
        ma_wins = moving_average(pursuer_wins, window)
        ma_episodes = np.arange(window, len(pursuer_wins) + 1)
        plot_and_save(ma_episodes, ma_wins,
                     f"Pursuer Win Rate (MA={window})", "Episode", "Win Rate",
                     os.path.join(outdir, "win_rate.png"))
    
    # Policy entropy (tracked every 500 episodes)
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


def visualize_policies(env: CompetitiveEnv, agent: NashDQN,
                      outdir: str = 'nash_results'):
    """Visualize learned policies for pursuer and evader."""
    os.makedirs(outdir, exist_ok=True)
    size = env.size
    
    # Create policy maps for pursuer and evader
    # Fix one agent position and show policy for the other
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    arrow_dict = {
        0: (0, -0.35, '↑'),  # Up
        1: (0, 0.35, '↓'),   # Down
        2: (-0.35, 0, '←'),  # Left
        3: (0.35, 0, '→')    # Right
    }
    
    # Pursuer policy: fix evader at (1,1)
    fixed_evader = (1, 1)
    pursuer_value_map = np.zeros((size, size))
    pursuer_policy_map = np.zeros((size, size), dtype=int)
    
    for pursuer_y in range(size):
        for pursuer_x in range(size):
            pursuer_pos = (pursuer_y, pursuer_x)
            if pursuer_pos != fixed_evader:
                state = env.state_to_features(pursuer_pos, fixed_evader)
                pursuer_value_map[pursuer_y, pursuer_x] = agent.get_value(state)
                pursuer_action, _ = agent.get_policy(state)
                pursuer_policy_map[pursuer_y, pursuer_x] = pursuer_action
            else:
                pursuer_value_map[pursuer_y, pursuer_x] = np.nan
                pursuer_policy_map[pursuer_y, pursuer_x] = -1
    
    # Evader policy: fix pursuer at (1,1)
    fixed_pursuer = (1, 1)
    evader_value_map = np.zeros((size, size))
    evader_policy_map = np.zeros((size, size), dtype=int)
    
    for evader_y in range(size):
        for evader_x in range(size):
            evader_pos = (evader_y, evader_x)
            if evader_pos != fixed_pursuer:
                state = env.state_to_features(fixed_pursuer, evader_pos)
                evader_value_map[evader_y, evader_x] = agent.get_value(state)
                _, evader_action = agent.get_policy(state)
                evader_policy_map[evader_y, evader_x] = evader_action
            else:
                evader_value_map[evader_y, evader_x] = np.nan
                evader_policy_map[evader_y, evader_x] = -1
    
    # Plot pursuer value
    im1 = axes[0, 0].imshow(pursuer_value_map, cmap='Reds', origin='upper')
    axes[0, 0].set_title('Pursuer Value Function\n(Evader fixed at (1,1))', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot pursuer policy
    axes[0, 1].imshow(np.ones((size, size)), cmap='gray', alpha=0.3, origin='upper')
    axes[0, 1].set_title('Pursuer Policy\n(Evader fixed at (1,1))', 
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
    
    # Plot evader value
    im2 = axes[1, 0].imshow(evader_value_map, cmap='Blues', origin='upper')
    axes[1, 0].set_title('Evader Value Function\n(Pursuer fixed at (1,1))',
                        fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Plot evader policy
    axes[1, 1].imshow(np.ones((size, size)), cmap='gray', alpha=0.3, origin='upper')
    axes[1, 1].set_title('Evader Policy\n(Pursuer fixed at (1,1))',
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
    plt.savefig(os.path.join(outdir, 'policies.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved policy visualization to {os.path.join(outdir, 'policies.png')}")


def draw_frame(env: CompetitiveEnv, pursuer_pos: Tuple[int, int],
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


def evaluate_all_starts(env: CompetitiveEnv, agent: NashDQN, 
                        max_steps: int = 50, outdir: str = 'nash_results'):
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
            # Choose best joint action (greedy) BEFORE drawing frame
            pursuer_action, evader_action = agent.choose_joint_action(state, training=False)
            
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
            state = env.state_to_features(pursuer_pos, evader_pos)
        
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
        
        traces[(start_pursuer_pos, start_evader_pos)] = {
            'trace': trace,
            'pursuer_reward': pursuer_total_reward,
            'evader_reward': evader_total_reward,
            'steps': steps,
            'caught': done and pursuer_pos == evader_pos,
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
    print(f"\nEvaluation Summary:")
    print(f"  Total start pairs: {len(traces)}")
    print(f"  Caught: {caught_count}")
    print(f"  Average steps: {np.mean([data['steps'] for data in traces.values()]):.2f}")
    print(f"  Average pursuer reward: {np.mean(pursuer_rewards_eval):.2f}")
    print(f"  Average evader reward: {np.mean(evader_rewards_eval):.2f}")
    print(f"  Frames and GIFs saved to: {eval_dir}")
    print(f"  Reward distribution plot saved to: {os.path.join(outdir, 'reward_distribution.png')}")


def save_parameters(params: Dict, outdir: str):
    """Save all parameters to a JSON file."""
    params_file = os.path.join(outdir, 'parameters.json')
    # Add timestamp
    import datetime
    params['run_info'] = {
        'timestamp': datetime.datetime.now().isoformat(),
        'timestamp_readable': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Parameters saved to {params_file}")


def print_parallelization_status(agent, env):
    """Print parallelization and optimization features status."""
    import torch
    
    print("\n" + "=" * 80)
    print("Parallelization and Optimization Status")
    print("=" * 80)
    
    # 1. Batch GPU computation for all Q-values
    gpu_available = torch.cuda.is_available()
    batch_gpu = gpu_available
    print(f"1. Batch GPU computation for all Q-values: {'✓ ENABLED' if batch_gpu else '✗ DISABLED (CPU only)'}")
    if batch_gpu:
        print(f"   Device: {torch.cuda.get_device_name(0)}")
    
    # 2. Multiprocessing for batch linear programming
    has_multiprocessing = get_has_multiprocessing()
    cuda_available = get_cuda_available()
    multiprocessing_enabled = (
        agent.enable_multiprocessing and
        has_multiprocessing and 
        agent.num_workers > 1 and 
        hasattr(env, 'valid_positions')
    )
    if multiprocessing_enabled:
        all_states_count = len([(p, e) for p in env.valid_positions 
                                for e in env.valid_positions if p != e])
        multiprocessing_enabled = all_states_count > 10
    
    print(f"2. Multiprocessing for batch linear programming: {'✓ ENABLED' if multiprocessing_enabled else '✗ DISABLED'}")
    if multiprocessing_enabled:
        print(f"   Workers: {agent.num_workers}")
        if cuda_available and not IS_WINDOWS:
            print(f"   Start method: spawn (required for CUDA)")
        elif IS_WINDOWS:
            print(f"   Start method: spawn (Windows default)")
        else:
            print(f"   Start method: fork (Linux default)")
    else:
        reasons = []
        if not agent.enable_multiprocessing:
            reasons.append("manually disabled")
        if not has_multiprocessing:
            reasons.append("not supported")
        if agent.num_workers <= 1:
            reasons.append(f"num_workers={agent.num_workers}")
        if cuda_available and IS_WINDOWS:
            reasons.append("Windows with CUDA (spawn overhead)")
        if not reasons:
            reasons.append("small batch size (< 10 states)")
        print(f"   Reason: {', '.join(reasons)}")
    
    # 3. Pre-solve all linear programming after update
    cache_update = agent.update_cache_after_training
    cache_freq = agent.cache_update_frequency
    print(f"3. Pre-solve all linear programming after update: {'✓ ENABLED' if cache_update else '✗ DISABLED'}")
    if cache_update:
        if hasattr(agent, '_all_states_for_cache') and agent._all_states_for_cache:
            num_states = len(agent._all_states_for_cache)
            print(f"   States to cache: {num_states}")
            print(f"   Cache update frequency: Every {cache_freq} training steps")
            print(f"   NOTE: Updating every step is too expensive with exact LP!")
        else:
            print(f"   WARNING: All states not set for cache (call agent.set_all_states_for_cache())")
    else:
        print(f"   Nash equilibria will be solved on-demand during action selection")
    
    # 4. Fast Nash algorithm
    fast_nash = agent.use_fast_nash
    print(f"4. Fast Nash algorithm: {'✓ ENABLED' if fast_nash else '✗ DISABLED (using exact LP)'}")
    if fast_nash:
        print(f"   Using iterative best response (approximate, ~10-100x faster)")
    else:
        print(f"   Using exact linear programming solver")
    
    print("=" * 80 + "\n")


def main():
    """Main training and evaluation loop."""
    # Set random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    import torch
    torch.manual_seed(random_seed)
    
    outdir = 'nash_results_light'
    os.makedirs(outdir, exist_ok=True)
    
    # Environment parameters
    env_size = 4
    
    # Agent hyperparameters
    input_size = 4  # [pursuer_x, pursuer_y, evader_x, evader_y]
    joint_action_size = 16  # 4 × 4 joint actions
    hidden_size = 128
    learning_rate = 5e-4
    gamma = 0.95
    epsilon_start = 0.5
    epsilon_end = 0.001
    epsilon_decay = 0.995
    tau = 0.01
    batch_size = 256  # Increased for better GPU utilization
    buffer_size = 50000
    
    # Training parameters
    num_episodes = 1000
    max_steps = 50
    
    # Create environment
    env = CompetitiveEnv(size=env_size)
    
    # Create Nash DQN agent
    # Optionally enable cache updates periodically for faster action selection
    update_cache = False  # Set to True to pre-compute Nash for all states periodically
    # Since train_step is already batched (256 samples), we can update cache more frequently
    # cache_update_frequency=1 means every training step (which processes 256 samples)
    cache_update_frequency = 1  # Update cache every N training steps (1 = every step, which is batched)
    enable_multiprocessing = False  # Set True to allow worker pool usage for snapshots/cache
    target_cache_refresh_interval = 25  # Clear target Nash cache every N soft updates
    agent = NashDQN(
        input_size=input_size,
        joint_action_size=joint_action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        tau=tau,
        batch_size=batch_size,
        buffer_size=buffer_size,
        use_fast_nash=True, # Use fast approximate Nash solver
        update_cache_after_training=update_cache,
        cache_update_frequency=cache_update_frequency,
        enable_multiprocessing=enable_multiprocessing,
        target_cache_refresh_interval=target_cache_refresh_interval
    )
    
    # Set all states for cache (if cache updates are enabled)
    if update_cache:
        all_states_for_cache = []
        for pursuer_pos in env.valid_positions:
            for evader_pos in env.valid_positions:
                if pursuer_pos != evader_pos:
                    state = env.state_to_features(pursuer_pos, evader_pos)
                    all_states_for_cache.append(state)
        agent.set_all_states_for_cache(all_states_for_cache)
        print(f"Cache update enabled: Will pre-compute Nash for {len(all_states_for_cache)} states every {cache_update_frequency} training steps")
    
    # Get device information
    import torch
    device_info = {
        'device': str(agent.device),
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device_info['cuda_device'] = torch.cuda.get_device_name(0)
        device_info['cuda_version'] = torch.version.cuda
        device_info['gpu_count'] = torch.cuda.device_count()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("WARNING: CUDA not available, using CPU (will be very slow)")
    
    # Collect all parameters
    parameters = {
        'random_seed': random_seed,
        'environment': {
            'size': env_size,
        },
        'agent': {
            'input_size': input_size,
            'joint_action_size': joint_action_size,
            'hidden_size': hidden_size,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay,
            'tau': tau,
            'batch_size': batch_size,
            'buffer_size': buffer_size,
            'enable_multiprocessing': enable_multiprocessing,
            'target_cache_refresh_interval': target_cache_refresh_interval,
        },
        'training': {
            'num_episodes': num_episodes,
            'max_steps': max_steps,
        },
        'network_architecture': {
            'type': 'NashDQNNetwork',
            'layers': [
                {'type': 'Linear', 'input': input_size, 'output': hidden_size},
                {'type': 'ReLU'},
                {'type': 'Linear', 'input': hidden_size, 'output': hidden_size},
                {'type': 'ReLU'},
                {'type': 'Linear', 'input': hidden_size, 'output': joint_action_size},
            ]
        },
        'nash_equilibrium': {
            'method': 'linear_programming',
            'solver': 'scipy.optimize.linprog',
        },
        'device': device_info
    }
    
    # Save parameters
    save_parameters(parameters, outdir)
    
    # Print parallelization status
    print_parallelization_status(agent, env)
    
    # Train agent
    (pursuer_rewards, evader_rewards, episode_lengths, td_errors,
     episode_winners, policy_entropies, value_diffs_max, value_diffs_mean,
     policy_l1_diffs) = train_agent(
        env,
        agent,
        num_episodes=num_episodes,
        max_steps=max_steps,
        log_interval=500,  # Disable expensive drift tracking
        track_policy_drift=True,
        print_interval=20
    )
    
    # Plot training progress
    plot_training_progress(pursuer_rewards, evader_rewards, episode_lengths, td_errors,
                          episode_winners, policy_entropies, value_diffs_max, 
                          value_diffs_mean, policy_l1_diffs, outdir=outdir)
    
    # Visualize and evaluate using exact LP solver
    with agent.exact_nash_mode():
        visualize_policies(env, agent, outdir=outdir)
        evaluate_all_starts(env, agent, max_steps=50, outdir=outdir)
    
    # Cleanup resources
    agent.cleanup()
    
    print("\nTraining and evaluation complete!")
    print(f"All results saved to the '{outdir}' directory.")


if __name__ == "__main__":
    main()

