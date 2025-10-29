import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pettingzoo import ParallelEnv
from gymnasium import spaces
import imageio.v2 as imageio  # use v2 to avoid deprecation warning
import pickle


class PursuitEvasionParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "pursuit_evasion_v0"}

    def __init__(self, grid_size=5, max_steps=50, render_mode="human", timeout_mode="tie", obstacles=None):
        """
          timeout_mode: str
              - "evader_win": evader wins if time runs out
              - "tie": game ends in a draw if time runs out
          """
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.timeout_mode = timeout_mode  # <-- new flag

        self.agents = ["pursuer", "evader"]
        self.pos = {}
        self.step_count = 0
        self.frames = []  # store frames for animation
        # define safe zone (bottom-left corner)
        self.safe_zone = (self.grid_size - 1, 0)
        self.terminal_reward = 30
        self.safe_zone_distance_factor = 1
        # obstacles (list of (y,x) tuples)
        if obstacles is None:
            self.obstacles = set()
        else:
            self.obstacles = set(obstacles)
        # 4 actions: up, down, left, right
        self.action_spaces = {
            agent: spaces.Discrete(4) for agent in self.agents
        }
        # observation = flattened grid + own pos
        obs_dim = grid_size * grid_size + 2
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
            for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        # initialize RNG
        if seed is not None:
            np.random.seed(seed)
        # positions: pursuer (top-left), evader (bottom-right)
        self.pos = {"pursuer": [0, 3], "evader": [0, 4]}
        self.step_count = 0
        self.frames = []  # reset frame buffer
        obs = self._get_obs()
        infos = {agent: {} for agent in self.agents}

        return obs, infos

    def step(self, actions):
        self.step_count += 1

        # current positions
        cur_p = tuple(self.pos["pursuer"])
        cur_e = tuple(self.pos["evader"])
        # apply moves
        for agent, action in actions.items():
            self._move(agent, action)

        # rewards
        rewards = {}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {agent: {} for agent in self.agents}
        winner = None
        # ---- 1) Capture checks (highest priority) ----
        # direct collision into same cell OR swapping -> pursuer catches evader
        swapping = (tuple(self.pos["pursuer"]) == cur_e and tuple(self.pos["evader"]) == cur_p)
        same_cell = (tuple(self.pos["pursuer"]) == tuple(self.pos["evader"]))
        # terminal reward
        terminal_reward = self.terminal_reward
        if same_cell or swapping:
            rewards["pursuer"] = terminal_reward
            rewards["evader"] = -terminal_reward
            terminations = {a: True for a in self.agents}
            winner = "pursuer"
        elif tuple(self.pos["evader"]) == self.safe_zone:
            # evader reaches safe zone first → evader wins
            rewards["pursuer"] = -terminal_reward
            rewards["evader"] = terminal_reward
            winner = "evader"
            terminations = {a: True for a in self.agents}
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
        else:
            # calculate distance between pursuer and evader
            py, px = self.pos["pursuer"]
            ey, ex = self.pos["evader"]
            # distance shaping: normalized to [0, 0.1] per step
            distance_delta = (abs(cur_e[0] - cur_p[0]) + abs(cur_e[1] - cur_p[1])) - (abs(py - ey) + abs(px - ex))
            ev_distance_delta = abs(cur_e[0] - self.safe_zone[0]) + abs(cur_e[1] - self.safe_zone[1]) - (
                    abs(ey - self.safe_zone[0]) + abs(ex - self.safe_zone[1]))
            shaping = distance_delta - self.safe_zone_distance_factor * ev_distance_delta
            # print(f"shaping: {1 - distance / max_distance} {.1 * ev_distance / max_distance}")
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
        grid[py, px] = 1  # pursuer
        grid[ey, ex] = 2  # evader
        sy, sx = self.safe_zone
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
        # if next cell is obstacle → stay still
        if (new_y, new_x) in self.obstacles:
            self.pos[agent] = [y, x]
        else:
            self.pos[agent] = [new_y, new_x]

    def _get_obs(self):
        obs = {}
        for agent in self.agents:
            grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            py, px = self.pos["pursuer"]
            ey, ex = self.pos["evader"]
            grid[py, px] = 1.0
            grid[ey, ex] = 2
            for oy, ox in self.obstacles:
                grid[oy, ox] = 0.5  # mark obstacles
            flat = grid.flatten()
            pos_vec = np.array(self.pos[agent]) / self.grid_size
            obs[agent] = np.concatenate([flat, pos_vec])
        return obs

    def state_key(self):
        # Canonical, agent-agnostic world state key
        return f"P{tuple(self.pos['pursuer'])}|E{tuple(self.pos['evader'])}|S{self.safe_zone}|O{sorted(self.obstacles)}"


def save_episode_animation(episode, save_dir="renders", fps=2):
    """Combine saved frames of one episode into a GIF"""
    import glob
    pattern = os.path.join(save_dir, f"ep{episode:05d}_step*.png")
    frame_files = sorted(glob.glob(pattern))
    images = [imageio.imread(f) for f in frame_files]
    gif_path = os.path.join(save_dir, f"episode_{episode:05d}.gif")
    # print(f"episode {episode} len(images) {len(images)}")
    # save GIF: duration per frame = 1/fps
    duration_per_frame = 1 / fps
    imageio.mimsave(gif_path, images, duration=duration_per_frame, loop=0, subrectangles=False)
    video_path = os.path.join(save_dir, f"episode_{episode:05d}.mp4")
    # write video
    imageio.mimsave(video_path, images, fps=fps, macro_block_size=None)
    print(f"Saved animation: {gif_path}")


sys.path.append(os.path.abspath("../../../minimax_q_learning"))

from minimax_q_learner import MiniMaxQLearner


class EpsGreedyPolicy:
    """
    Epsilon-greedy *around* a base mixed strategy pi_probs.

    - select_action(pi_probs):
        With probability epsilon choose a uniform random action (exploration).
        Otherwise sample an action from the pi_probs distribution (preserves mixed strategy).
    - select_greedy_action(pi_probs, deterministic=True):
        If deterministic=True -> return argmax(pi_probs) (pure).
        If deterministic=False -> sample from pi_probs (stochastic evaluation).
    """

    def __init__(self, epsilon=0.1):
        self.epsilon = float(epsilon)

    def _normalize(self, pi_probs):
        p = np.asarray(pi_probs, dtype=float).copy()
        # clip negative numerical garbage, then normalize to sum=1
        p = np.clip(p, 0.0, None)
        s = p.sum()
        if s <= 0 or np.isnan(s):
            # fallback to uniform
            p = np.ones_like(p, dtype=float) / len(p)
        else:
            p = p / s
        return p

    def select_action(self, pi_probs):
        """
        Training action selection:
        - with prob epsilon: uniform random
        - else: sample according to pi_probs
        Returns: integer action index
        """
        p = self._normalize(pi_probs)
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(len(p)))
        return int(np.random.choice(len(p), p=p))

    def select_greedy_action(self, pi_probs, deterministic=False):
        """
        Evaluation action selection.
        - deterministic=True  => pure argmax (reproducible)
        - deterministic=False => sample according to pi_probs (keeps mixed play)
        """
        p = self._normalize(pi_probs)
        if deterministic:
            return int(np.argmax(p))
        else:
            return int(np.random.choice(len(p), p=p))


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
            f.write(f"States missing in evader table: {len(missing_in_e)}\n")
    print(f"Saved Q tables and antisymmetry report to {report_path}")


if __name__ == "__main__":
    # --- environment ---
    env = PursuitEvasionParallelEnv(grid_size=5, max_steps=10, obstacles=[(0, 2), (1, 1)])

    # --- create Minimax-Q learners ---
    actions = list(range(4))  # up, down, left, right

    # --- training loop ---
    num_episodes = 30000
    #num_episodes = 1
    init_eps = 0.3
    init_alpha = 0.2
    gamma = 1
    os.makedirs("stat", exist_ok=True)
    setting_path = os.path.join("stat", "setting.txt")
    with open(setting_path, "w") as f:
        f.write(
            f"init_epsilon {init_eps}\n init_alpha {init_alpha}\nterminal_reward {env.terminal_reward}\n"
            f"gamma {gamma}\nsafezone_distance_factor {env.safe_zone_distance_factor}"
        )
    returns = {"pursuer": [], "evader": []}
    avg_returns = {"pursuer": [], "evader": []}
    episode_lengths, pursuer_rewards, evader_rewards, winners = [], [], [], []
    pursuer_rewards_per_step, evader_rewards_per_step = [], []
    pursuer_policy = EpsGreedyPolicy(epsilon=init_eps)
    evader_policy = EpsGreedyPolicy(epsilon=init_eps)

    pursuer_learner = MiniMaxQLearner(aid="pursuer", alpha=init_alpha, policy=pursuer_policy, gamma=gamma,
                                      actions=actions)
    evader_learner = MiniMaxQLearner(aid="evader", alpha=init_alpha, policy=evader_policy, actions=actions)

    learners = {"pursuer": pursuer_learner, "evader": evader_learner}
    print(f"ep0 pi pursuer {pursuer_learner.pi['nonstate']}, evader {evader_learner.pi['nonstate']}")
    log_path = os.path.join("stat", "pi_log.txt")
    log_file = open(log_path, "w")
    for ep in range(num_episodes):
        # pursuer_learner.policy.epsilon = max(0.05, init_eps * (0.999 ** ep))
        # evader_learner.policy.epsilon = max(0.05, init_eps * (0.999 ** ep))
        #
        # pursuer_learner.alpha = max(0.01, init_alpha * (0.999 ** ep))
        # evader_learner.alpha = max(0.01, init_alpha * (0.999 ** ep))
        obs, infos = env.reset(seed=ep)
        total_rewards = {"pursuer": 0, "evader": 0}
        avg_rewards = {"pursuer": 0, "evader": 0}
        step = 0
        if (ep + 1) % 100 == 0:
            # use a canonical, agent-agnostic state key
            state_str = env.state_key()  # <- add env.state_key() as shown earlier

            # make sure both learners have this state initialized
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
        while env.agents:
            # get actions from learners
            actions_dict = {}
            for agent in env.agents:
                state_str = env.state_key()
                learners[agent].state = state_str
                learners[agent].check_new_state(state_str)
                actions_dict[agent] = learners[agent].act(training=True)

            # take a step in the environment
            next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)
            # rewards["pursuer"] /= step + 1
            # rewards["evader"] /= step+1
            total_rewards["pursuer"] += rewards["pursuer"]
            total_rewards["evader"] += rewards["evader"]

            # let learners observe and learn
            for agent in env.agents:
                next_state_str = env.state_key()
                learners[agent].observe(
                    state=next_state_str,
                    reward=rewards[agent],
                    opponent_action=actions_dict[["evader", "pursuer"][agent == "evader"]],
                    is_learn=True
                )

            obs = next_obs
            if (ep + 1) % 100 == 0:
                env.render(episode=ep, step=step + 1, live=False)
            step += 1
            if any(terminations.values()) or any(truncations.values()):
                winners.append(infos["pursuer"]["winner"])
                break
        # store total return for this episode
        for agent in total_rewards:
            returns[agent].append(total_rewards[agent])
        pursuer_learner.end_episode()
        evader_learner.end_episode()
        # reward per step
        pursuer_rewards.append(total_rewards["pursuer"])
        evader_rewards.append(total_rewards["evader"])
        avg_rewards["pursuer"] = total_rewards["pursuer"] / step
        avg_rewards["evader"] = total_rewards["evader"] / step
        pursuer_rewards_per_step.append(avg_rewards["pursuer"])
        evader_rewards_per_step.append(avg_rewards["evader"])
        episode_lengths.append(step)
        # every 100 episodes: compute moving average
        if (ep + 1) % 100 == 0:
            for agent in returns:
                avg = np.mean(returns[agent][-100:])
                avg_returns[agent].append(avg)
            print(f"Episode {ep}, Avg(100) pursuer={avg_returns['pursuer'][-1]:.3f}, "
                  f"evader={avg_returns['evader'][-1]:.3f}")
            # save episode animation
            save_episode_animation(ep)
        episode_lengths.append(step)
        print(
            f"Episode {ep} finished in {step} steps. Avg rewards: Pursurer {avg_rewards['pursuer']:.4g}; Evader {avg_rewards['evader']:.4g}")
    log_file.close()
    save_q_and_report(pursuer_learner, evader_learner, actions, outdir="stat", tol=1e-5)

    plt.figure()
    plt.plot(episode_lengths)
    plt.title("Episode Lengths")
    plt.savefig(os.path.join("stat", "episode_lengths.png"))


    def moving_average(x, window=100):
        return np.convolve(x, np.ones(window) / window, mode="valid")


    plt.figure()
    plt.plot(moving_average(pursuer_rewards, 100), label="Pursuer")
    plt.plot(moving_average(evader_rewards, 100), label="Evader")
    plt.legend()
    plt.title("Smoothed Cumulative Rewards per Episode (100-episode MA)")
    plt.savefig(os.path.join("stat", "rewards.png"))

    plt.figure()
    labels, counts = np.unique(winners, return_counts=True)
    plt.bar(labels, counts)
    plt.title("Winners Distribution")
    plt.savefig(os.path.join("stat", "winners.png"))

    plt.figure()
    # --- plot learning curve ---
    plt.plot(np.arange(1, len(avg_returns["pursuer"]) + 1) * 100, avg_returns["pursuer"], label="Pursuer")
    plt.plot(np.arange(1, len(avg_returns["evader"]) + 1) * 100, avg_returns["evader"], label="Evader")
    plt.xlabel("Episodes")
    plt.ylabel("Average return (last 100)")
    plt.legend()
    plt.title("Learning Progress (Minimax-Q)")
    plt.savefig(os.path.join("stat", "return100.png"))

    plt.figure()
    plt.plot(moving_average(pursuer_rewards_per_step, 100), label="Pursuer (per step)")
    plt.plot(moving_average(evader_rewards_per_step, 100), label="Evader (per step)")
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
    eval_episodes = 1  # number of demo runs
    for ep in range(eval_episodes):
        obs, infos = env.reset(seed=999 + ep)
        step = 0
        total_rewards = {"pursuer": 0, "evader": 0}
        env.render(episode=num_episodes + ep, step=step, live=False)
        while True:
            actions_dict = {}
            for agent in env.agents:
                state_str = env.state_key()
                learners[agent].state = state_str
                # make sure the state exists in learned policy
                if state_str not in learners[agent].pi:
                    learners[agent].pi[state_str] = np.repeat(1.0 / len(actions), len(actions))
                # deterministic greedy policy
                actions_dict[agent] = learners[agent].policy.select_greedy_action(learners[agent].pi[state_str])

            # environment step
            next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)
            total_rewards["pursuer"] += rewards["pursuer"]
            total_rewards["evader"] += rewards["evader"]

            # render to file (no live display)
            env.render(episode=num_episodes + ep, step=step + 1, live=False)
            obs = next_obs
            step += 1

            if any(terminations.values()) or any(truncations.values()):
                winner = infos["pursuer"]["winner"]
                print(f"Eval Episode {ep}: winner={winner}, steps={step}, "
                      f"R_p={total_rewards['pursuer']:.2f}, R_e={total_rewards['evader']:.2f}")
                break

        # make a video for this evaluation run
        save_episode_animation(num_episodes + ep, save_dir="renders", fps=2)

    env.close()
