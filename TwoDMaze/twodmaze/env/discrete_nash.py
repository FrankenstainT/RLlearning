import os
import sys
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pettingzoo import ParallelEnv
from gymnasium import spaces
import imageio.v2 as imageio  # use v2 to avoid deprecation warning
from scipy.optimize import linprog


# ============================================================
#                PURSUIT–EVASION PARALLEL ENV
# ============================================================
class PursuitEvasionParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "pursuit_evasion_v0"}

    def __init__(self, grid_size=5, max_steps=50, render_mode="human", timeout_mode="tie", obstacles=None):
        """
        timeout_mode:
            - "evader_win": evader wins if time runs out
            - "tie": game ends in a draw if time runs out
        """
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.timeout_mode = timeout_mode

        self.agents = ["pursuer", "evader"]
        self.pos = {}
        self.step_count = 0
        self.frames = []  # store frames for animation

        # define safe zone (bottom-left corner)
        self.safe_zone = (self.grid_size - 1, 0)
        self.terminal_reward = 10  # was 30
        self.safe_zone_distance_factor = 0.1  # was 1
        self.distance_factor = .1
        # obstacles (list of (y,x) tuples)
        self.obstacles = set(obstacles) if obstacles is not None else set()

        # 4 actions: up, down, left, right
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.agents}

        # observation = flattened grid + own pos (not used for keys anymore, but kept for compatibility)
        obs_dim = grid_size * grid_size + 2
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
            for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        # positions (customize if you want)
        self.pos = {"pursuer": [0, 3], "evader": [0, 4]}
        self.step_count = 0
        self.frames = []
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

        rewards = {}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {agent: {} for agent in self.agents}
        winner = None

        # ---- 1) Capture checks (highest priority) ----
        swapping = (tuple(self.pos["pursuer"]) == cur_e and tuple(self.pos["evader"]) == cur_p)
        same_cell = (tuple(self.pos["pursuer"]) == tuple(self.pos["evader"]))
        terminal_reward = self.terminal_reward

        if same_cell or swapping:
            rewards["pursuer"] = terminal_reward
            rewards["evader"] = -terminal_reward
            terminations = {a: True for a in self.agents}
            winner = "pursuer"
        elif tuple(self.pos["evader"]) == self.safe_zone:
            rewards["pursuer"] = -terminal_reward
            rewards["evader"] = terminal_reward
            terminations = {a: True for a in self.agents}
            winner = "evader"
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
            # step shaping (antisymmetric)
            py, px = self.pos["pursuer"]
            ey, ex = self.pos["evader"]
            dist_before = abs(cur_e[0] - cur_p[0]) + abs(cur_e[1] - cur_p[1])
            dist_after = abs(py - ey) + abs(px - ex)
            distance_delta = dist_before - dist_after

            ev_before = abs(cur_e[0] - self.safe_zone[0]) + abs(cur_e[1] - self.safe_zone[1])
            ev_after = abs(ey - self.safe_zone[0]) + abs(ex - self.safe_zone[1])
            ev_distance_delta = ev_before - ev_after

            shaping = self.distance_factor * distance_delta - self.safe_zone_distance_factor * ev_distance_delta
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
        # obstacle block -> stay
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
            grid[ey, ex] = 2.0
            for oy, ox in self.obstacles:
                grid[oy, ox] = 0.5
            flat = grid.flatten()
            pos_vec = np.array(self.pos[agent]) / self.grid_size
            obs[agent] = np.concatenate([flat, pos_vec])
        return obs

    def state_key(self):
        # Canonical, agent-agnostic world state key
        return f"P{tuple(self.pos['pursuer'])}|E{tuple(self.pos['evader'])}|S{self.safe_zone}|O{sorted(self.obstacles)}"


# ============================================================
#                UTIL: SAVE EPISODE ANIMATION
# ============================================================
def save_episode_animation(episode, save_dir="renders", fps=2):
    pattern = os.path.join(save_dir, f"ep{episode:05d}_step*.png")
    frame_files = sorted(glob.glob(pattern))
    if not frame_files:
        print(f"No frames found for episode {episode} in {save_dir}")
        return
    images = [imageio.imread(f) for f in frame_files]
    gif_path = os.path.join(save_dir, f"episode_{episode:05d}.gif")
    duration_per_frame = 1 / fps
    imageio.mimsave(gif_path, images, duration=duration_per_frame, loop=0, subrectangles=False)
    video_path = os.path.join(save_dir, f"episode_{episode:05d}.mp4")
    imageio.mimsave(video_path, images, fps=fps, macro_block_size=None)
    print(f"Saved animation: {gif_path}")


# ============================================================
#                EPSILON-GREEDY AROUND MIXED POLICY
# ============================================================
class EpsGreedyPolicy:
    """
    Epsilon-greedy around a given mixed strategy pi_probs.
    """

    def __init__(self, epsilon=0.1):
        self.epsilon = float(epsilon)

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
        p = self._normalize(pi_probs)
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(len(p)))
        return int(np.random.choice(len(p), p=p))

    def select_greedy_action(self, pi_probs, deterministic=False):
        p = self._normalize(pi_probs)
        if deterministic:
            return int(np.argmax(p))
        else:
            return int(np.random.choice(len(p), p=p))


# ============================================================
#                ONE-LP SOLVER FOR BOTH POLICIES
# ============================================================
def solve_both_policies_one_lp(M):
    """
    Solve both players' optimal mixed strategies from a single LP:
        maximize v
        s.t.   v <= x^T M[:,j]   for all j
               M[i,:] y <= v      for all i
               1^T x = 1, x >= 0
               1^T y = 1, y >= 0
    We minimize -v via SciPy HiGHS.
    Returns x, y, v.
    """
    M = np.asarray(M, dtype=float)
    m, n = M.shape

    # Decision vector z = [x(0..m-1), y(0..n-1), v]
    num_vars = m + n + 1
    x_slice = slice(0, m)
    y_slice = slice(m, m + n)
    v_idx = m + n

    # Objective: minimize -v (equivalently, maximize v)
    c = np.zeros(num_vars)
    c[v_idx] = -1.0

    # Inequalities: A_ub z <= b_ub
    A_ub, b_ub = [], []

    # 1) v <= x^T M[:,j] -> -M[:,j]^T x + v <= 0
    for j in range(n):
        row = np.zeros(num_vars)
        row[x_slice] = -M[:, j]
        row[v_idx] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # 2) M[i,:] y <= v -> M[i,:] y - v <= 0
    for i in range(m):
        row = np.zeros(num_vars)
        row[y_slice] = M[i, :]
        row[v_idx] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    A_ub = np.vstack(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None

    # Equalities: 1^T x = 1, 1^T y = 1
    A_eq = np.zeros((2, num_vars))
    b_eq = np.array([1.0, 1.0])
    A_eq[0, x_slice] = 1.0
    A_eq[1, y_slice] = 1.0

    # Bounds: x>=0, y>=0, v free
    bounds = [(0, None)] * m + [(0, None)] * n + [(None, None)]

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs"
    )
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")

    z = res.x
    x = z[x_slice].clip(min=0)
    y = z[y_slice].clip(min=0)
    v = z[v_idx]

    # Normalize (robust to numerical drift)
    sx, sy = x.sum(), y.sum()
    if sx > 0:
        x /= sx
    else:
        x = np.ones(m) / m
    if sy > 0:
        y /= sy
    else:
        y = np.ones(n) / n

    return x, y, v


# ============================================================
#                NASH-Q LEARNER (ONE Q TABLE)
# ============================================================
class NashQLearner:
    def __init__(self, actions, alpha=0.2, gamma=0.99,
                 eps_policy_pursuer=None, eps_policy_evader=None,
                 alpha_power=0.6):
        self.actions = actions
        self.nA = len(actions)
        self.gamma = float(gamma)
        self.alpha0 = float(alpha)      # initial scale; real α is visitation based
        self.alpha_power = float(alpha_power)

        self.Q = {}         # state -> {(ap,ae): q}
        self.V = {}         # state -> scalar
        self.pi_x = {}      # state -> pursuer policy
        self.pi_y = {}      # state -> evader  policy
        self.visits = {}    # state -> {(ap,ae): count}
        self.policy_dirty = set()  # states needing LP resolve

        self.state = None
        self.last_ap = None
        self.last_ae = None
        self.eps_p = eps_policy_pursuer
        self.eps_e = eps_policy_evader

        self.episode_max_delta = 0.0
        self.episode_deltas = []

    def _ensure_state(self, s):
        if s not in self.Q:
            self.Q[s] = {(ap, ae): 0.0 for ap in self.actions for ae in self.actions}
            self.V[s] = 0.0
            self.pi_x[s] = np.ones(self.nA) / self.nA
            self.pi_y[s] = np.ones(self.nA) / self.nA
            self.visits[s] = {(ap, ae): 0 for ap in self.actions for ae in self.actions}
            self.policy_dirty.add(s)

    def _matrix_from_Q(self, s):
        self._ensure_state(s)
        M = np.zeros((self.nA, self.nA))
        for ap in self.actions:
            for ae in self.actions:
                M[ap, ae] = self.Q[s][(ap, ae)]
        return M

    def _solve_policies(self, s, force=False):
        self._ensure_state(s)
        if (s in self.policy_dirty) or force:
            M = self._matrix_from_Q(s)
            x, y, v = solve_both_policies_one_lp(M)
            self.pi_x[s], self.pi_y[s], self.V[s] = x, y, v
            self.policy_dirty.discard(s)
        return self.pi_x[s], self.pi_y[s], self.V[s]

    def start_state(self, s):
        self.state = s
        self._ensure_state(s)
        self._solve_policies(s, force=True)

    def act(self):
        s = self.state
        x, y, _ = self._solve_policies(s)
        ap = self.eps_p.select_action(x) if self.eps_p else int(np.random.choice(self.nA, p=x))
        ae = self.eps_e.select_action(y) if self.eps_e else int(np.random.choice(self.nA, p=y))
        self.last_ap, self.last_ae = ap, ae
        return {"pursuer": ap, "evader": ae}

    def _alpha(self, s, ap, ae):
        # α = α0 / (1 + visits)^{alpha_power}
        n = self.visits[s][(ap, ae)]
        return self.alpha0 / ((1.0 + n) ** self.alpha_power)

    def observe(self, next_state, r_pursuer, done):
        s = self.state
        ap, ae = self.last_ap, self.last_ae
        self._ensure_state(s)
        self._ensure_state(next_state)

        # ensure next state's value is consistent with its current Q
        self._solve_policies(next_state)

        # target and visitation-based step
        target = r_pursuer + (0.0 if done else self.gamma * self.V[next_state])
        old_q = self.Q[s][(ap, ae)]
        a = self._alpha(s, ap, ae)
        new_q = old_q + a * (target - old_q)
        self.Q[s][(ap, ae)] = new_q
        self.visits[s][(ap, ae)] += 1

        # policy at s is now stale (Q changed) -> mark dirty
        self.policy_dirty.add(s)

        self.episode_max_delta = max(self.episode_max_delta, abs(new_q - old_q))
        self.state = next_state

    def end_episode(self):
        self.episode_deltas.append(self.episode_max_delta)
        self.episode_max_delta = 0.0



# ============================================================
#                SAVE NASH-Q TABLES (STATE -> MATRIX)
# ============================================================
def save_nash_q(nash_learner, actions, outdir="stat"):
    os.makedirs(outdir, exist_ok=True)
    mats = {}
    for s in nash_learner.Q.keys():
        M = np.zeros((len(actions), len(actions)), float)
        for ap in actions:
            for ae in actions:
                M[ap, ae] = nash_learner.Q[s][(ap, ae)]
        mats[s] = M
    with open(os.path.join(outdir, "nash_q_tables.pkl"), "wb") as f:
        pickle.dump(mats, f)
    print(f"Saved Nash-Q tables to {os.path.join(outdir, 'nash_q_tables.pkl')}")


def state_exploitability(M, x, y, v):
    """
    Return (eps, row_gain, col_gain):
      row_gain = max_i M[i,:] y - v
      col_gain = v - min_j x^T M[:,j]
      eps = max(row_gain, col_gain)
    """
    M = np.asarray(M, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    row_vals = M @ y  # length m
    col_vals = M.T @ x  # length n
    row_gain = float(np.max(row_vals) - v)
    col_gain = float(v - np.min(col_vals))
    eps = max(row_gain, col_gain)
    return eps, row_gain, col_gain


def nash_report(nash_learner, actions, tol=1e-4, outdir="stat"):
    """
    For every state in the learned Q table:
      1) build M from Q
      2) solve one-LP to get (x,y,v)
      3) compute exploitability eps
    Writes a text report and returns a dict with summary stats.
    """
    os.makedirs(outdir, exist_ok=True)
    worst = (-1.0, None, None, None)  # (eps, state, row_gain, col_gain)
    eps_list = []
    lines = []
    for s in nash_learner.Q.keys():
        # build M
        nA = len(actions)
        M = np.zeros((nA, nA), float)
        for ap in actions:
            for ae in actions:
                M[ap, ae] = nash_learner.Q[s][(ap, ae)]
        # policies & value implied by current Q
        x, y, v = solve_both_policies_one_lp(M)
        eps, row_gain, col_gain = state_exploitability(M, x, y, v)
        eps_list.append(eps)
        if eps > worst[0]:
            worst = (eps, s, row_gain, col_gain)
        lines.append(
            f"state={s}\n  v={v:.6g}  eps={eps:.3e}  row_gain={row_gain:.3e}  col_gain={col_gain:.3e}\n"
            f"  x={np.round(x, 6)}\n  y={np.round(y, 6)}\n"
        )

    # Aggregate
    if eps_list:
        eps_arr = np.array(eps_list, float)
        summary = {
            "num_states": len(eps_list),
            "eps_max": float(np.max(eps_arr)),
            "eps_mean": float(np.mean(eps_arr)),
            "eps_median": float(np.median(eps_arr)),
            "within_tol": int(np.sum(eps_arr <= tol)),
            "tol": tol,
            "worst_state": worst[1],
            "worst_eps": worst[0],
            "worst_row_gain": worst[2],
            "worst_col_gain": worst[3],
        }
    else:
        summary = {
            "num_states": 0, "eps_max": None, "eps_mean": None,
            "eps_median": None, "within_tol": 0, "tol": tol,
            "worst_state": None, "worst_eps": None,
            "worst_row_gain": None, "worst_col_gain": None,
        }

    # Write report
    report_path = os.path.join(outdir, "nash_exploitability_report.txt")
    with open(report_path, "w") as f:
        f.write("Per-state ε-Nash (exploitability) report\n")
        f.write(f"Tolerance: {tol}\n\n")
        for ln in lines:
            f.write(ln + "\n")
        f.write("\n=== Summary ===\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved Nash exploitability report to {report_path}")
    return summary


# ============================================================
#                           MAIN
# ============================================================
if __name__ == "__main__":
    # --- environment ---
    env = PursuitEvasionParallelEnv(grid_size=5, max_steps=50, obstacles=[(0, 2), (1, 1)])

    # --- actions (rows= pursuer, cols = evader) ---
    actions = list(range(4))  # up, down, left, right

    # --- training loop params ---
    num_episodes = 50000
    # num_episodes = 101
    init_eps = 0.3
    final_eps = 0.02
    init_alpha = 0.2
    gamma = .9
    eps_decay = (final_eps / init_eps) ** (1.0 / num_episodes)  # exponential

    os.makedirs("stat", exist_ok=True)
    with open(os.path.join("stat", "setting.txt"), "w") as f:
        f.write(
            f"init_epsilon {init_eps}\ninit_alpha {init_alpha}\nterminal_reward {env.terminal_reward}\n"
            f"gamma {gamma}\nsafezone_distance_factor {env.safe_zone_distance_factor}\n"
        )

    # trackers
    returns = {"pursuer": [], "evader": []}
    avg_returns = {"pursuer": [], "evader": []}
    episode_lengths, pursuer_rewards, evader_rewards, winners = [], [], [], []
    pursuer_rewards_per_step, evader_rewards_per_step = [], []

    # policies used for exploration/eval
    pursuer_policy = EpsGreedyPolicy(epsilon=init_eps)
    evader_policy = EpsGreedyPolicy(epsilon=init_eps)

    nash_learner = NashQLearner(
        actions=actions,
        alpha=0.5,  # α0 (pre-visitation): slightly larger, will shrink fast
        gamma=gamma,
        eps_policy_pursuer=EpsGreedyPolicy(epsilon=init_eps),
        eps_policy_evader=EpsGreedyPolicy(epsilon=init_eps),
        alpha_power=0.6,  # 0.6 ~ Robbins–Monro style
    )
    print("ep0 Nash-Q initialized: uniform policies at unseen states")
    log_path = os.path.join("stat", "pi_log.txt")
    log_file = open(log_path, "w")

    for ep in range(num_episodes):
        obs, infos = env.reset(seed=ep)
        total_rewards = {"pursuer": 0.0, "evader": 0.0}
        avg_rewards = {"pursuer": 0.0, "evader": 0.0}
        step = 0

        # start state
        state_str = env.state_key()
        nash_learner.start_state(state_str)
        nash_learner.eps_p.epsilon *= eps_decay
        nash_learner.eps_e.epsilon *= eps_decay
        # (optional) clamp to final_eps
        nash_learner.eps_p.epsilon = max(nash_learner.eps_p.epsilon, final_eps)
        nash_learner.eps_e.epsilon = max(nash_learner.eps_e.epsilon, final_eps)
        # occasional logging/frames
        if (ep + 1) % 100 == 0:
            x, y, v = nash_learner._solve_policies(state_str)
            env.render(episode=ep, step=step, live=False)
            log_file.write(f"ep{ep} Nash pi_x {x}, pi_y {y}, v {v:.4f}\n")
            print(f"ep{ep} Nash pi_x {np.round(x, 4)}, pi_y {np.round(y, 4)}, v {v:.4f}")

        while True:
            # act using current Nash policies (+ epsilon exploration)
            joint = nash_learner.act()
            actions_dict = {"pursuer": joint["pursuer"], "evader": joint["evader"]}

            # env step
            next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)
            done = any(terminations.values()) or any(truncations.values())

            # accumulate returns
            total_rewards["pursuer"] += rewards["pursuer"]
            total_rewards["evader"] += rewards["evader"]

            # learn from pursuer payoff only
            next_state_str = env.state_key()
            nash_learner.observe(
                next_state=next_state_str,
                r_pursuer=rewards["pursuer"],
                done=done
            )

            if (ep + 1) % 100 == 0:
                env.render(episode=ep, step=step + 1, live=False)

            step += 1
            if done:
                winners.append(infos["pursuer"]["winner"])
                break

        # bookkeeping for plots
        for agent in total_rewards:
            returns[agent].append(total_rewards[agent])
        nash_learner.end_episode()

        pursuer_rewards.append(total_rewards["pursuer"])
        evader_rewards.append(total_rewards["evader"])
        avg_rewards["pursuer"] = total_rewards["pursuer"] / step
        avg_rewards["evader"] = total_rewards["evader"] / step
        pursuer_rewards_per_step.append(avg_rewards["pursuer"])
        evader_rewards_per_step.append(avg_rewards["evader"])
        episode_lengths.append(step)

        # every 100 episodes: moving average over last 100
        if (ep + 1) % 100 == 0:
            for agent in returns:
                avg = np.mean(returns[agent][-100:])
                avg_returns[agent].append(avg)
            print(f"Episode {ep}, Avg(100) pursuer={avg_returns['pursuer'][-1]:.3f}, "
                  f"evader={avg_returns['evader'][-1]:.3f}")
            save_episode_animation(ep)

        print(f"Episode {ep} finished in {step} steps. "
              f"Avg rewards: Pursuer {avg_rewards['pursuer']:.4g}; Evader {avg_rewards['evader']:.4g}")

    log_file.close()
    summary = nash_report(nash_learner, actions, tol=1e-4, outdir="stat")
    print("Exploitability summary:", summary)

    # ----- save Q -----
    save_nash_q(nash_learner, actions, outdir="stat")


    # ============================================================
    #                      PLOTTING
    # ============================================================
    def moving_average(x, window=100):
        x = np.asarray(x, dtype=float)
        if len(x) < window:
            return x.copy()
        return np.convolve(x, np.ones(window) / window, mode="valid")


    # Episode lengths
    plt.figure()
    plt.plot(episode_lengths)
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.savefig(os.path.join("stat", "episode_lengths.png"))

    # Smoothed cumulative rewards per episode
    plt.figure()
    plt.plot(moving_average(pursuer_rewards, 100), label="Pursuer")
    plt.plot(moving_average(evader_rewards, 100), label="Evader")
    plt.legend()
    plt.title("Smoothed Cumulative Rewards per Episode (100-episode MA)")
    plt.xlabel("Episode (MA index)")
    plt.ylabel("Return")
    plt.savefig(os.path.join("stat", "rewards.png"))

    # Winners distribution
    if len(winners) > 0:
        plt.figure()
        labels, counts = np.unique(winners, return_counts=True)
        plt.bar(labels, counts)
        plt.title("Winners Distribution")
        plt.xlabel("Winner")
        plt.ylabel("Count")
        plt.savefig(os.path.join("stat", "winners.png"))

    # Learning progress (Avg return over last 100)
    plt.figure()
    xs = np.arange(1, len(avg_returns["pursuer"]) + 1) * 100
    plt.plot(xs, avg_returns["pursuer"], label="Pursuer")
    plt.plot(xs, avg_returns["evader"], label="Evader")
    plt.xlabel("Episodes")
    plt.ylabel("Average return (last 100)")
    plt.legend()
    plt.title("Learning Progress (Nash-Q)")
    plt.savefig(os.path.join("stat", "return100.png"))

    # Average rewards per step per episode (smoothed)
    plt.figure()
    plt.plot(moving_average(pursuer_rewards_per_step, 100), label="Pursuer (per step)")
    plt.plot(moving_average(evader_rewards_per_step, 100), label="Evader (per step)")
    plt.xlabel("Episode (MA index)")
    plt.ylabel("Average Rewards per Step (100-episode MA)")
    plt.legend()
    plt.title("Average Rewards per Step per Episode")
    plt.savefig(os.path.join("stat", "rewards_per_step.png"))

    # Q-table convergence (per-episode max Δ)
    plt.figure()
    plt.plot(nash_learner.episode_deltas, label="Nash-Q")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Max Q-update Δ")
    plt.title("Q-table Convergence (Nash-Q)")
    plt.savefig(os.path.join("stat", "q_convergence.png"))
    plt.close()

    # ============================================================
    #                FINAL EVALUATION (GREEDY)
    # ============================================================
    print("\n=== Running Final Evaluation (Nash-Q) ===")
    eval_episodes = 1
    for ep in range(eval_episodes):
        obs, infos = env.reset(seed=999 + ep)
        step = 0
        total_rewards = {"pursuer": 0.0, "evader": 0.0}

        s = env.state_key()
        nash_learner.start_state(s)  # ensures policies ready

        env.render(episode=num_episodes + ep, step=step, live=False)
        while True:
            if nash_learner.state not in nash_learner.Q:
                print("Eval reached unseen state:", nash_learner.state)
            nash_learner._ensure_state(nash_learner.state)  # <-- add this

            # greedy (deterministic=True) play from current Nash policies
            x, y, _ = nash_learner._solve_policies(nash_learner.state)
            ap = nash_learner.eps_p.select_greedy_action(x, deterministic=True)
            ae = nash_learner.eps_e.select_greedy_action(y, deterministic=True)
            actions_dict = {"pursuer": ap, "evader": ae}

            next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)
            total_rewards["pursuer"] += rewards["pursuer"]
            total_rewards["evader"] += rewards["evader"]

            env.render(episode=num_episodes + ep, step=step + 1, live=False)

            next_s = env.state_key()
            nash_learner._ensure_state(next_s)
            # advance internal pointer (no learning during eval)
            nash_learner.state = next_s

            step += 1
            if any(terminations.values()) or any(truncations.values()):
                winner = infos["pursuer"]["winner"]
                print(f"Eval Episode {ep}: winner={winner}, steps={step}, "
                      f"R_p={total_rewards['pursuer']:.2f}, R_e={total_rewards['evader']:.2f}")
                break

        save_episode_animation(num_episodes + ep, save_dir="renders", fps=2)

    env.close()
