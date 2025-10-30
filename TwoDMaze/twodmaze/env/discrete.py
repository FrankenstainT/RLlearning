import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pettingzoo import ParallelEnv
from gymnasium import spaces
import imageio.v2 as imageio  # use v2 to avoid deprecation warning
import pickle


# =========================
#  Environment Definition
# =========================
class PursuitEvasionParallelEnv(ParallelEnv):
    """
    Two-player pursuit/evasion on a grid with obstacles and one safe zone.

    Conventions
    ----------
    - Grid coordinates are (y, x) with (0,0) at top-left.
    - Actions: 0=up, 1=down, 2=left, 3=right.
    - Capture: same cell or a swap -> pursuer wins.
    - Safe-zone tie: if both stand in the safe-zone together, that's "same cell" so pursuer wins (as specified).
    - timeout_mode: "evader_win" or "tie" when max_steps reached.
    """

    metadata = {"render_modes": ["human"], "name": "pursuit_evasion_v0"}

    def __init__(self, grid_size=5, max_steps=50, render_mode="human", timeout_mode="tie", obstacles=None):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.timeout_mode = timeout_mode

        self.agents = ["pursuer", "evader"]
        self.pos = {}
        self.step_count = 0

        # Per-episode RNG (set in reset with the provided seed)
        self.rng = np.random.default_rng()

        # Visualization frames (written via render)
        self.frames = []
        # Safe zone (bottom-left)
        self.safe_zone = (self.grid_size - 1, 0)

        # Reward shaping / terminal reward
        self.terminal_reward = 30
        self.safe_zone_distance_factor = 1.0

        # Obstacles
        self.obstacles = set(obstacles) if obstacles is not None else set()

        # Discrete actions
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.agents}

        # Observation = flattened grid + own (y,x) position (normalized)
        # Grid encoding in observation (all in [0,1]):
        #   empty=0.0, obstacle=0.5, safe_zone=0.3, pursuer=1.0, evader=0.8
        obs_dim = grid_size * grid_size + 2
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        # Episode-specific RNG (do NOT touch global RNG)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # You can randomize starts if you like; here we keep yours:
        self.pos = {"pursuer": [0, 3], "evader": [0, 4]}

        self.step_count = 0
        self.frames = []
        obs = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def step(self, actions):
        self.step_count += 1

        cur_p = tuple(self.pos["pursuer"])
        cur_e = tuple(self.pos["evader"])

        # apply moves
        for agent, action in actions.items():
            self._move(agent, action)

        rewards = {}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {agent: {} for agent in self.agents}
        winner = None

        # --- terminal checks ---
        swapping = (tuple(self.pos["pursuer"]) == cur_e and tuple(self.pos["evader"]) == cur_p)
        same_cell = (tuple(self.pos["pursuer"]) == tuple(self.pos["evader"]))

        terminal_reward = self.terminal_reward

        # Capture first (includes same-time arrival in same cell)
        if same_cell or swapping:
            rewards["pursuer"] = terminal_reward
            rewards["evader"] = -terminal_reward
            terminations = {a: True for a in self.agents}
            winner = "pursuer"

        # Evader makes it to safe zone alone
        elif tuple(self.pos["evader"]) == self.safe_zone:
            rewards["pursuer"] = -terminal_reward
            rewards["evader"] = terminal_reward
            terminations = {a: True for a in self.agents}
            winner = "evader"

        # Timeout
        elif self.step_count >= self.max_steps:
            if self.timeout_mode == "evader_win":
                rewards["pursuer"] = -terminal_reward
                rewards["evader"] = terminal_reward
                winner = "evader"
            elif self.timeout_mode == "tie":
                rewards["pursuer"] = 0.0
                rewards["evader"] = 0.0
                winner = "tie"
            truncations = {a: True for a in self.agents}

        # Shaping
        else:
            py, px = self.pos["pursuer"]
            ey, ex = self.pos["evader"]
            # pursuer-evader distance delta (negative if farther, positive if closer)
            dist_before = abs(cur_e[0] - cur_p[0]) + abs(cur_e[1] - cur_p[1])
            dist_after = abs(py - ey) + abs(px - ex)
            distance_delta = dist_before - dist_after

            # evader-to-safe-zone distance delta (positive if evader moved closer)
            ev_before = abs(cur_e[0] - self.safe_zone[0]) + abs(cur_e[1] - self.safe_zone[1])
            ev_after = abs(ey - self.safe_zone[0]) + abs(ex - self.safe_zone[1])
            ev_distance_delta = ev_before - ev_after

            # pursuer likes getting closer to evader; dislikes evader getting closer to safe zone
            shaping = distance_delta - self.safe_zone_distance_factor * ev_distance_delta
            rewards["pursuer"] = shaping
            rewards["evader"] = -shaping
            winner = None

        for agent in self.agents:
            infos[agent]["winner"] = winner

        obs = self._get_obs()
        return obs, rewards, terminations, truncations, infos

    def render(self, save_dir="renders", episode=0, step=0, live=True):
        os.makedirs(save_dir, exist_ok=True)
        grid = np.zeros((self.grid_size, self.grid_size))
        py, px = self.pos["pursuer"]
        ey, ex = self.pos["evader"]
        sy, sx = self.safe_zone

        # labels for visualization (not used in observations)
        grid[py, px] = 1  # pursuer
        grid[ey, ex] = 2  # evader
        grid[sy, sx] = 3  # safe zone
        for oy, ox in self.obstacles:
            grid[oy, ox] = 4  # obstacle

        cmap = colors.ListedColormap(["white", "red", "blue", "green", "black"])
        plt.figure()
        plt.imshow(grid, cmap=cmap, vmin=0, vmax=4)
        plt.xticks(range(self.grid_size))
        plt.yticks(range(self.grid_size))
        plt.grid(True, which="both", color="gray", linewidth=0.5)
        fname = os.path.join(save_dir, f"ep{episode:05d}_step{step:02d}.png")
        plt.savefig(fname)
        if live:
            plt.show(block=False)
            plt.pause(0.2)
            plt.clf()
        else:
            plt.close()

    def close(self):
        plt.close()

    # --- helpers ---
    def _move(self, agent, action):
        y, x = self.pos[agent]
        new_y, new_x = y, x
        if action == 0 and y > 0:  # up
            new_y -= 1
        elif action == 1 and y < self.grid_size - 1:  # down
            new_y += 1
        elif action == 2 and x > 0:  # left
            new_x -= 1
        elif action == 3 and x < self.grid_size - 1:  # right
            new_x += 1

        # obstacle collision -> stay
        if (new_y, new_x) in self.obstacles:
            self.pos[agent] = [y, x]
        else:
            self.pos[agent] = [new_y, new_x]

    def _get_obs(self):
        """
        Encode a single-channel grid into [0,1] plus own (y,x)/grid_size:
            empty   = 0.0
            obstacle= 0.5
            safe    = 0.3
            pursuer = 1.0
            evader  = 0.8
        """
        obs = {}
        for agent in self.agents:
            grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            py, px = self.pos["pursuer"]
            ey, ex = self.pos["evader"]
            sy, sx = self.safe_zone
            # mark map
            for oy, ox in self.obstacles:
                grid[oy, ox] = 0.5
            grid[sy, sx] = 0.3
            grid[py, px] = 1.0
            grid[ey, ex] = 0.8

            flat = grid.flatten()
            pos_vec = np.array(self.pos[agent], dtype=np.float32) / float(self.grid_size)
            obs[agent] = np.concatenate([flat, pos_vec]).astype(np.float32)
        return obs

    def state_key(self):
        # Canonical, agent-agnostic world state key
        return f"P{tuple(self.pos['pursuer'])}|E{tuple(self.pos['evader'])}|S{self.safe_zone}|O{sorted(self.obstacles)}"


def save_episode_animation(episode, save_dir="renders", fps=2):
    """Combine saved frames of one episode into both GIF and MP4."""
    import glob
    pattern = os.path.join(save_dir, f"ep{episode:05d}_step*.png")
    frame_files = sorted(glob.glob(pattern))
    if not frame_files:
        print(f"[warn] No frames found for episode {episode} in {save_dir}")
        return
    images = [imageio.imread(f) for f in frame_files]

    gif_path = os.path.join(save_dir, f"episode_{episode:05d}.gif")
    imageio.mimsave(gif_path, images, duration=1 / fps, loop=0, subrectangles=False)

    mp4_path = os.path.join(save_dir, f"episode_{episode:05d}.mp4")
    imageio.mimsave(mp4_path, images, fps=fps, macro_block_size=None)
    print(f"Saved animation: {gif_path}")
    print(f"Saved animation: {mp4_path}")


# =========================
# Minimax-Q glue utilities
# =========================
sys.path.append(os.path.abspath("../../../minimax_q_learning"))
from minimax_q_learner import MiniMaxQLearner  # noqa: E402


class EpsGreedyPolicy:
    """
    Epsilon-greedy *around* a base mixed strategy pi_probs.

    Uses its own RNG (set per episode) so episodes are reproducible and independent.
    """

    def __init__(self, epsilon=0.1, rng=None):
        self.epsilon = float(epsilon)
        self.rng = rng if rng is not None else np.random.default_rng()

    def set_rng(self, rng):
        self.rng = rng

    def _normalize(self, pi_probs):
        p = np.asarray(pi_probs, dtype=float).copy()
        p = np.clip(p, 0.0, None)
        s = p.sum()
        if s <= 0 or np.isnan(s):
            p = np.ones_like(p, dtype=float) / len(p)
        else:
            p = p / s
        return p

    def select_action(self, pi_probs):
        """Training selection: epsilon-uniform else sample from pi."""
        p = self._normalize(pi_probs)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(low=0, high=len(p)))
        return int(self.rng.choice(len(p), p=p))

    def select_greedy_action(self, pi_probs, deterministic=False):
        """Evaluation: argmax if deterministic, else sample from pi."""
        p = self._normalize(pi_probs)
        if deterministic:
            return int(np.argmax(p))
        else:
            return int(self.rng.choice(len(p), p=p))


def qtensor_by_state(learner, actions):
    """Return dict: state_key -> (nA x nA) matrix with Q[action_pursuer, action_evader]."""
    nA = len(actions)
    out = {}
    for s, qdict in learner.q.items():
        M = np.zeros((nA, nA), dtype=float)
        for a1 in actions:
            for a2 in actions:
                M[a1, a2] = qdict.get((a1, a2), 0.0)
        out[s] = M
    return out


def save_q_and_report(pursuer_learner, evader_learner, actions, outdir="stat", tol=1e-5):
    os.makedirs(outdir, exist_ok=True)
    pQ = qtensor_by_state(pursuer_learner, actions)
    eQ = qtensor_by_state(evader_learner, actions)

    # Save raw Q-tables (pickle)
    with open(os.path.join(outdir, "q_tables.pkl"), "wb") as f:
        pickle.dump({"pursuer": pQ, "evader": eQ}, f)

    # Compare: for states present in either learner
    all_states = sorted(set(pQ.keys()) | set(eQ.keys()))
    diffs = []
    missing_in_p, missing_in_e = [], []
    for s in all_states:
        if s not in pQ:
            missing_in_p.append(s)
            continue
        if s not in eQ:
            missing_in_e.append(s)
            continue
        D = pQ[s] + eQ[s]  # should be ~ zero matrix
        diffs.append((s, float(np.max(np.abs(D))), float(np.mean(np.abs(D)))))

    diffs.sort(key=lambda t: t[1], reverse=True)
    report_path = os.path.join(outdir, "q_antisymmetry_report.txt")
    with open(report_path, "w") as f:
        f.write("Q antisymmetry check: expecting Q_pursuer(s,a1,a2) = - Q_evader(s,a1,a2)\n")
        f.write(f"Total comparable states: {len(diffs)}\n")
        if diffs:
            worst = diffs[0]
            f.write(f"Worst state max|Qp+Qe| = {worst[1]:.6g}, mean|Qp+Qe| = {worst[2]:.6g}, state = {worst[0]}\n")
            num_ok = sum(d[1] <= tol for d in diffs)
            f.write(f"States within tol ({tol}): {num_ok}/{len(diffs)}\n")
            f.write("\nTop 10 worst states:\n")
            for s, mx, mn in diffs[:10]:
                f.write(f"  max={mx:.6g}, mean={mn:.6g}, state={s}\n")
        if missing_in_p:
            f.write(f"\nStates missing in pursuer table: {len(missing_in_p)}\n")
        if missing_in_e:
            f.write(f"\nStates missing in evader table: {len(missing_in_e)}\n")
    print(f"Saved Q tables and antisymmetry report to {report_path}")


# =========================
#          Main
# =========================
if __name__ == "__main__":
    # --- environment ---
    env = PursuitEvasionParallelEnv(grid_size=5, max_steps=10, obstacles=[(0, 2), (1, 1)])

    # --- create Minimax-Q learners ---
    actions = list(range(4))  # 0=up,1=down,2=left,3=right

    # --- training loop ---
    num_episodes = 30000
    init_eps = 0.3
    init_alpha = 0.2
    gamma = 1.0

    os.makedirs("stat", exist_ok=True)
    with open(os.path.join("stat", "setting.txt"), "w") as f:
        f.write(
            f"init_epsilon {init_eps}\ninit_alpha {init_alpha}\n"
            f"terminal_reward {env.terminal_reward}\n"
            f"gamma {gamma}\nsafezone_distance_factor {env.safe_zone_distance_factor}\n"
        )

    returns = {"pursuer": [], "evader": []}
    avg_returns = {"pursuer": [], "evader": []}
    episode_lengths, pursuer_rewards, evader_rewards, winners = [], [], [], []
    pursuer_rewards_per_step, evader_rewards_per_step = [], []

    # Policies with per-episode RNG
    pursuer_policy = EpsGreedyPolicy(epsilon=init_eps)
    evader_policy = EpsGreedyPolicy(epsilon=init_eps)

    pursuer_learner = MiniMaxQLearner(aid="pursuer", alpha=init_alpha, policy=pursuer_policy, gamma=gamma, actions=actions)
    evader_learner = MiniMaxQLearner(aid="evader",  alpha=init_alpha, policy=evader_policy,  actions=actions)
    learners = {"pursuer": pursuer_learner, "evader": evader_learner}

    # Log initial (nonstate) mixed strategy
    print(f"ep0 pi pursuer {pursuer_learner.pi['nonstate']}, evader {evader_learner.pi['nonstate']}")
    log_path = os.path.join("stat", "pi_log.txt")
    log_file = open(log_path, "w")

    for ep in range(num_episodes):
        # Create distinct RNGs per episode for env + each agent (no global RNG usage)
        env_seed = 17_071 * (ep + 1) + 3  # arbitrary linear congruential-ish
        pursuer_seed = 97_561 * (ep + 1) + 11
        evader_seed  = 73_421 * (ep + 1) + 29

        # Set RNGs
        obs, infos = env.reset(seed=env_seed)
        pursuer_policy.set_rng(np.random.default_rng(pursuer_seed))
        evader_policy.set_rng(np.random.default_rng(evader_seed))

        total_rewards = {"pursuer": 0.0, "evader": 0.0}
        step = 0

        if (ep + 1) % 100 == 0:
            state_str = env.state_key()
            # ensure state exists
            pursuer_learner.check_new_state(state_str)
            evader_learner.check_new_state(state_str)
            env.render(episode=ep, step=step, live=False)
            log_file.write(
                f"ep{ep} pi pursuer {pursuer_learner.pi[state_str]}, "
                f"evader {evader_learner.pi[state_str]}\n"
            )
            print(
                f"ep{ep} pi pursuer {pursuer_learner.pi[state_str]}, "
                f"evader {evader_learner.pi[state_str]}"
            )

        while True:
            actions_dict = {}
            for agent in env.agents:
                state_str = env.state_key()
                learners[agent].state = state_str
                learners[agent].check_new_state(state_str)
                actions_dict[agent] = learners[agent].act(training=True)

            next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)

            total_rewards["pursuer"] += rewards["pursuer"]
            total_rewards["evader"] += rewards["evader"]

            # learners update
            for agent in env.agents:
                next_state_str = env.state_key()
                learners[agent].observe(
                    state=next_state_str,
                    reward=rewards[agent],
                    opponent_action=actions_dict["evader" if agent == "pursuer" else "pursuer"],
                    is_learn=True
                )

            obs = next_obs
            if (ep + 1) % 100 == 0:
                env.render(episode=ep, step=step + 1, live=False)

            step += 1

            if any(terminations.values()) or any(truncations.values()):
                winners.append(infos["pursuer"]["winner"])
                break

        # store total return
        for agent in total_rewards:
            returns[agent].append(total_rewards[agent])

        # end-episode hooks for learners
        pursuer_learner.end_episode()
        evader_learner.end_episode()

        # per-step averages for plots
        avg_p = total_rewards["pursuer"] / step
        avg_e = total_rewards["evader"] / step
        pursuer_rewards.append(total_rewards["pursuer"])
        evader_rewards.append(total_rewards["evader"])
        pursuer_rewards_per_step.append(avg_p)
        evader_rewards_per_step.append(avg_e)
        episode_lengths.append(step)

        # rolling stats every 100 episodes
        if (ep + 1) % 100 == 0:
            for agent in returns:
                avg_returns[agent].append(float(np.mean(returns[agent][-100:])))
            print(f"Episode {ep}, Avg(100) pursuer={avg_returns['pursuer'][-1]:.3f}, "
                  f"evader={avg_returns['evader'][-1]:.3f}")
            save_episode_animation(ep)

        print(f"Episode {ep} finished in {step} steps. Avg rewards: "
              f"Pursuer {avg_p:.4g}; Evader {avg_e:.4g}")

    log_file.close()
    save_q_and_report(pursuer_learner, evader_learner, actions, outdir="stat", tol=1e-5)

    # =========================
    #         Plots
    # =========================
    plt.figure()
    plt.plot(episode_lengths)
    plt.title("Episode Lengths")
    plt.savefig(os.path.join("stat", "episode_lengths.png"))

    def moving_average(x, window=100):
        if len(x) < window:
            return np.array([])
        return np.convolve(x, np.ones(window) / window, mode="valid")

    plt.figure()
    ma_pr = moving_average(pursuer_rewards, 100)
    ma_er = moving_average(evader_rewards, 100)
    if ma_pr.size:
        plt.plot(ma_pr, label="Pursuer")
        plt.plot(ma_er, label="Evader")
        plt.legend()
        plt.title("Smoothed Cumulative Rewards per Episode (100-episode MA)")
        plt.savefig(os.path.join("stat", "rewards.png"))

    plt.figure()
    labels, counts = np.unique(winners, return_counts=True)
    plt.bar(labels, counts)
    plt.title("Winners Distribution")
    plt.savefig(os.path.join("stat", "winners.png"))

    plt.figure()
    x_axis = np.arange(1, len(avg_returns["pursuer"]) + 1) * 100
    plt.plot(x_axis, avg_returns["pursuer"], label="Pursuer")
    plt.plot(x_axis, avg_returns["evader"], label="Evader")
    plt.xlabel("Episodes")
    plt.ylabel("Average return (last 100)")
    plt.legend()
    plt.title("Learning Progress (Minimax-Q)")
    plt.savefig(os.path.join("stat", "return100.png"))

    plt.figure()
    ma_ps = moving_average(pursuer_rewards_per_step, 100)
    ma_es = moving_average(evader_rewards_per_step, 100)
    if ma_ps.size:
        plt.plot(ma_ps, label="Pursuer (per step)")
        plt.plot(ma_es, label="Evader (per step)")
        plt.xlabel("Episodes")
        plt.ylabel("Average Rewards per Step (100-episode MA)")
        plt.legend()
        plt.title("Average Rewards per Step per Episode")
        plt.savefig(os.path.join("stat", "rewards_per_step.png"))

    plt.figure()
    plt.plot(pursuer_learner.episode_deltas, label="Pursuer")
    plt.plot(evader_learner.episode_deltas, label="Evader")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Max Q-update Δ")
    plt.title("Q-table Convergence")
    plt.savefig(os.path.join("stat", "q_convergence.png"))
    plt.close()

    # ===== Final Evaluation (greedy play) =====
    print("\n=== Running Final Evaluation ===")
    eval_episodes = 1
    for ep in range(eval_episodes):
        # Different RNGs from training
        env_seed = 2_000_003 + ep
        obs, infos = env.reset(seed=env_seed)

        # evaluation uses deterministic greedy; still set RNGs so sampling path is stable if used
        pursuer_policy.set_rng(np.random.default_rng(3_000_001 + ep))
        evader_policy.set_rng(np.random.default_rng(3_100_001 + ep))

        step = 0
        total_rewards = {"pursuer": 0.0, "evader": 0.0}
        env.render(episode=num_episodes + ep, step=step, live=False)

        while True:
            actions_dict = {}
            for agent in env.agents:
                state_str = env.state_key()
                learners[agent].state = state_str
                if state_str not in learners[agent].pi:
                    learners[agent].pi[state_str] = np.repeat(1.0 / len(actions), len(actions))
                actions_dict[agent] = learners[agent].policy.select_greedy_action(
                    learners[agent].pi[state_str], deterministic=True
                )

            next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)
            total_rewards["pursuer"] += rewards["pursuer"]
            total_rewards["evader"] += rewards["evader"]

            env.render(episode=num_episodes + ep, step=step + 1, live=False)
            obs = next_obs
            step += 1

            if any(terminations.values()) or any(truncations.values()):
                winner = infos["pursuer"]["winner"]
                print(f"Eval Episode {ep}: winner={winner}, steps={step}, "
                      f"R_p={total_rewards['pursuer']:.2f}, R_e={total_rewards['evader']:.2f}")
                break

        save_episode_animation(num_episodes + ep, save_dir="renders", fps=2)

    env.close()
