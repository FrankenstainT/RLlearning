import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

os.makedirs("results", exist_ok=True)
sns.set_theme()
ACTION_VEC = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # (dy, dx) in grid (row, col)
ARROW_STYLE = dict(angles='xy', scale_units='xy', scale=3, width=0.006)

AGENT_COLORS = {1: "black", 2: "darkorange"}
AGENT_OFFSETS = {1: (-0.15, -0.15), 2: (0.15, 0.15)}  # (dx, dy) visual offset so arrows won’t overlap


# ============================================================
#                CUSTOM TWO-AGENT FROZEN LAKE ENV
# ============================================================
class TwoAgentFrozenLake:
    """
    Custom two-agent FrozenLake environment.
    Both agents move on a shared 4x4 grid with holes, frozen tiles, and one goal.
    The episode ends when both reach the goal.
    """

    def __init__(self, map_size=4, seed=123):
        self.n = map_size
        self.rng = np.random.default_rng(seed)
        self.desc = np.array([
            list("SFFF"),
            list("FHFH"),
            list("FFFH"),
            list("HFFG"),
        ])
        self.n_states = self.n * self.n
        self.n_actions = 4
        self.goal = (self.n - 1, self.n - 1)
        self.current_state = None
        self.step_penalty = -0.08

    # ----- Coordinate helpers -----
    def encode_state(self, s1, s2):
        return s1 * self.n_states + s2

    def decode_state(self, joint_idx):
        return divmod(joint_idx, self.n_states)

    def move(self, y, x, action):
        """Move one agent according to the action, bounded by edges."""
        if action == 0:  # LEFT
            x = max(x - 1, 0)
        elif action == 1:  # DOWN
            y = min(y + 1, self.n - 1)
        elif action == 2:  # RIGHT
            x = min(x + 1, self.n - 1)
        elif action == 3:  # UP
            y = max(y - 1, 0)
        return y, x

    # ----- Environment core -----
    def reset(self):
        s1 = (0, 0)
        s2 = (0, 0)
        self.current_state = (s1, s2)
        return self.encode_state(self.pos_to_idx(s1), self.pos_to_idx(s2))

    def pos_to_idx(self, pos):
        y, x = pos
        return y * self.n + x

    def idx_to_pos(self, idx):
        return divmod(idx, self.n)

    def step(self, a1, a2):
        (y1, x1), (y2, x2) = self.current_state
        if (y1, x1) == self.goal:
            ny1, nx1 = y1, x1
        else:
            ny1, nx1 = self.move(y1, x1, a1)
        if (y2, x2) == self.goal:
            ny2, nx2 = y2, x2
        else:
            ny2, nx2 = self.move(y2, x2, a2)
        tile1, tile2 = self.desc[ny1, nx1], self.desc[ny2, nx2]
        r1 = 0
        r2 = 0
        if tile1 == "H" or tile2 == "H":
            self.current_state = ((ny1, nx1), (ny2, nx2))
            ns = self.encode_state(self.pos_to_idx((ny1, nx1)), self.pos_to_idx((ny2, nx2)))
            if tile1 == "H":
                r1 = -1
            if tile2 == 'H':
                r2 = -1
            return ns, r1, r2, True
        r1 = 1.0 if (y1, x1) != self.goal and (ny1, nx1) == self.goal else 0.0
        r2 = 1.0 if (y2, x2) != self.goal and (ny2, nx2) == self.goal else 0.0

        r1 += self.step_penalty  # base step penalty
        r2 += self.step_penalty
        gy, gx = self.goal
        r1 += 0.1 * (abs(gy - y1) + abs(gx - x1)-(abs(gy - ny1) + abs(gx - nx1)))
        r2 += .1 * (abs(gy - y2) + abs(gx - x2)-(abs(gy - ny2) + abs(gx - nx2)))
        done = (tile1 == "G" and tile2 == "G")

        self.current_state = ((ny1, nx1), (ny2, nx2))
        next_state = self.encode_state(self.pos_to_idx((ny1, nx1)), self.pos_to_idx((ny2, nx2)))
        return next_state, r1, r2, done

    def step_with_info(self, a1, a2):
        (y1, x1), (y2, x2) = self.current_state

        # respect waiting at goal (same as step)
        ny1, nx1 = (y1, x1) if (y1, x1) == self.goal else self.move(y1, x1, a1)
        ny2, nx2 = (y2, x2) if (y2, x2) == self.goal else self.move(y2, x2, a2)

        tile1, tile2 = self.desc[ny1, nx1], self.desc[ny2, nx2]
        r1 = 0
        r2 = 0
        # If any agent falls into a hole: terminal with -1.0 (match step)
        if tile1 == "H" or tile2 == "H":
            if tile1 == 'H':
                r1 = -1
            if tile2 =='H':
                r2 = -1
            self.current_state = ((ny1, nx1), (ny2, nx2))
            next_state = self.encode_state(self.pos_to_idx((ny1, nx1)), self.pos_to_idx((ny2, nx2)))
            info = {
                "s1": self.pos_to_idx((y1, x1)), "s2": self.pos_to_idx((y2, x2)),
                "ns1": self.pos_to_idx((ny1, nx1)), "ns2": self.pos_to_idx((ny2, nx2)),
                "s1_yx": (y1, x1), "s2_yx": (y2, x2),
                "ns1_yx": (ny1, nx1), "ns2_yx": (ny2, nx2),
                "a1": a1, "a2": a2,
                "r_env_1": 0.0, "r_env_2": 0.0,
                "base_step_penalty": 0.0,
                "proximity_gain": 0.0,
                "hole_penalty": -1.0,
                "dist_before": None, "dist_after": None,
                "reward_r1": r1,
                "reward_r2":r2,
                "done": True,
                "tile1": tile1, "tile2": tile2,
            }
            return next_state, -1.0, True, info

        # normal step (same shaping as step())
        r1 = 1.0 if (y1, x1) != self.goal and (ny1, nx1) == self.goal else 0.0
        r2 = 1.0 if (y2, x2) != self.goal and (ny2, nx2) == self.goal else 0.0
        r1+= self.step_penalty
        r2+= self.step_penalty

        gy, gx = self.goal
        dist_before = abs(gy - y1) + abs(gx - x1) + abs(gy - y2) + abs(gx - x2)
        dist_after = abs(gy - ny1) + abs(gx - nx1) + abs(gy - ny2) + abs(gx - nx2)
        proximity_gain = 0.1 * (dist_before - dist_after)
        r1 += .1*(abs(gy - y1) + abs(gx - x1)-(abs(gy - ny1) + abs(gx - nx1)))
        r2+= .1*(abs(gy - y2) + abs(gx - x2)-(abs(gy - ny2) + abs(gx - nx2)))

        done = (tile1 == "G" and tile2 == "G")

        self.current_state = ((ny1, nx1), (ny2, nx2))
        next_state = self.encode_state(self.pos_to_idx((ny1, nx1)), self.pos_to_idx((ny2, nx2)))

        info = {
            "s1": self.pos_to_idx((y1, x1)), "s2": self.pos_to_idx((y2, x2)),
            "ns1": self.pos_to_idx((ny1, nx1)), "ns2": self.pos_to_idx((ny2, nx2)),
            "s1_yx": (y1, x1), "s2_yx": (y2, x2),
            "ns1_yx": (ny1, nx1), "ns2_yx": (ny2, nx2),
            "a1": a1, "a2": a2,
            "r_env_1": r1, "r_env_2": r2,
            "base_step_penalty": - 0.08,
            "proximity_gain": proximity_gain,
            "hole_penalty": 0.0,
            "dist_before": dist_before, "dist_after": dist_after,
            "reward_r1": r1,
            "reward_r2": r2,
            "done": done,
            "tile1": tile1, "tile2": tile2,
        }
        return next_state, r1,r2, done, info

    def render(self):
        grid = self.desc.copy().astype(str)
        (y1, x1), (y2, x2) = self.current_state
        if (y1, x1) == (y2, x2):
            grid[y1, x1] = "B"
        else:
            grid[y1, x1] = "1"
            grid[y2, x2] = "2"
        print("\n".join("".join(row) for row in grid))

    # --- add inside class TwoAgentFrozenLake ---

    def safe_indices(self):
        """All non-hole tiles as scalar indices."""
        safe = []
        for r in range(self.n):
            for c in range(self.n):
                t = self.desc[r, c]
                if t == "H":
                    continue
                if t == "G":
                    continue
                safe.append(self.pos_to_idx((r, c)))
        return safe

    def reset_to_indices(self, s1_idx, s2_idx):
        """Reset positions to given scalar indices (can overlap)."""
        self.current_state = (self.idx_to_pos(s1_idx), self.idx_to_pos(s2_idx))
        return self.encode_state(s1_idx, s2_idx)


# ============================================================
#                        AGENT
# ============================================================
class IndependentAgent:
    def __init__(self, state_size, action_size, lr=0.8, gamma=0.95, epsilon=1.0, seed=123):
        self.qtable = np.zeros((state_size, action_size))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state):
        if self.rng.uniform(0, 1) < self.epsilon:
            return self.rng.integers(0, self.qtable.shape[1])
        max_ids = np.where(self.qtable[state, :] == np.max(self.qtable[state, :]))[0]
        return self.rng.choice(max_ids)

    def update(self, s, a, r, ns):
        best_next = np.max(self.qtable[ns, :])
        self.qtable[s, a] += self.lr * (r + self.gamma * best_next - self.qtable[s, a])


# ============================================================
#                        TRAINING
# ============================================================
def train_two_agents_representative(
        episodes_per_start=100,
        map_size=4,
        seed=123,
        max_steps=30,
        eps_start=0.4,
        eps_end=0.02,
        alpha_start=0.8,
        alpha_end=0.2,
):
    env = TwoAgentFrozenLake(map_size=map_size, seed=seed)
    rng = np.random.default_rng(seed)

    state_size = env.n_states ** 2
    action_size = env.n_actions
    agent1 = IndependentAgent(state_size, action_size, seed=seed)
    agent2 = IndependentAgent(state_size, action_size, seed=seed + 1)

    # Build representative start list: all safe x safe (overlap allowed)
    safe = env.safe_indices()
    # start_pairs = [(s1, s2) for s1 in safe for s2 in safe]
    start_pairs = [(0, 0) for _ in range(10)]
    #rng.shuffle(start_pairs)
    total_episodes = len(start_pairs) * episodes_per_start
    ep_counter = 0
    rewards, steps, mean_q1, mean_q2 = [], [], [], []
    epsilons = []  # <--- add this
    agent1.epsilon = eps_start
    agent2.epsilon = eps_start
    for (s1_idx, s2_idx) in tqdm(start_pairs, desc="Start pairs"):
        # eps_decay = .99  # (eps_end / eps_start) ** (1.0 / max(1, episodes_per_start - 1))

        for _ in range(episodes_per_start):
            # GLIE-style schedules (global, not reset per start)
            frac = ep_counter / max(1, total_episodes - 1)
            eps = eps_start + (eps_end - eps_start) * frac
            alpha = alpha_start + (alpha_end - alpha_start) * frac
            agent1.epsilon = agent2.epsilon = max(eps_end, eps)
            agent1.lr = agent2.lr = max(alpha_end, alpha)

            state = env.reset_to_indices(s1_idx, s2_idx)
            done, total_r, t = False, 0.0, 0

            while not done and t < max_steps:
                # Is an agent already parked on the goal at the *current* state?
                s1_idx_curr, s2_idx_curr = env.decode_state(state)
                at_goal1 = env.idx_to_pos(s1_idx_curr) == env.goal
                at_goal2 = env.idx_to_pos(s2_idx_curr) == env.goal
                # Choose actions (frozen agents take a no-op placeholder)
                a1 = 1 if at_goal1 else agent1.choose_action(state)
                a2 = 1 if at_goal2 else agent2.choose_action(state)
                ns, r1, r2, done = env.step(a1, a2)

                # Update only agents that are not already waiting on the goal
                if not at_goal1:
                    agent1.update(state, a1, r1, ns)
                if not at_goal2:
                    agent2.update(state, a2, r2, ns)

                state = ns
                total_r += r1+r2
                t += 1

            # per-episode epsilon decay within the same start
            # agent1.epsilon = max(eps_end, agent1.epsilon * eps_decay)
            # agent2.epsilon = max(eps_end, agent2.epsilon * eps_decay)
            ep_counter += 1
            rewards.append(total_r);
            steps.append(t)
            mean_q1.append(np.mean(agent1.qtable));
            mean_q2.append(np.mean(agent2.qtable))
            epsilons.append(agent1.epsilon)

    return env, agent1, agent2, rewards, steps, mean_q1, mean_q2, epsilons


# ============================================================
#                       VISUALIZATION
# ============================================================
def plot_training_statistics(rewards, steps, mean_q1, mean_q2, epsilons):
    def smooth(x, k=100):
        return np.convolve(x, np.ones(k) / k, mode="valid")

    fig, ax = plt.subplots(4, 1, figsize=(10, 12))

    ax[0].plot(smooth(rewards))
    ax[0].set_title("Smoothed Total Reward per Episode")
    ax[0].set_ylabel("Total Reward")

    ax[1].plot(smooth(steps))
    ax[1].set_title("Smoothed Steps per Episode")
    ax[1].set_ylabel("Steps")

    ax[2].plot(smooth(mean_q1), label="Agent 1 Mean Q")
    ax[2].plot(smooth(mean_q2), label="Agent 2 Mean Q", color="darkorange")
    ax[2].legend()
    ax[2].set_title("Mean Q-values")
    ax[2].set_ylabel("Q")

    ax[3].plot(epsilons)
    ax[3].set_title("Epsilon Decay (Exploration Rate)")
    ax[3].set_ylabel("Epsilon")

    plt.xlabel("Episode")
    plt.tight_layout()
    plt.savefig("results/two_agent_training_stats.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_frozenlake_map(env):
    """Draw map with start, frozen, holes, goal.
    Coordinate system: (0,0) top-left, (n-1,0) bottom-left.
    """
    desc = env.desc
    n = desc.shape[0]
    color_map = {"S": "#90ee90", "F": "#add8e6", "H": "#d3d3d3", "G": "#ffd700"}

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title("FrozenLake Map Layout")

    # draw each tile at (col=j, row=i) without vertical flip
    for i in range(n):
        for j in range(n):
            tile = desc[i, j]
            ax.add_patch(
                plt.Rectangle((j, i), 1, 1,
                              color=color_map.get(tile, "white"),
                              ec="black", lw=0.7)
            )
            ax.text(j + 0.5, i + 0.5, tile,
                    ha="center", va="center", fontsize=14, weight="bold")

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)  # invert Y so row 0 is at the top
    ax.set_xticks(range(n + 1))
    ax.set_yticks(range(n + 1))
    ax.grid(True, color="black", linewidth=0.7)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("results/frozenlake_map_custom.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_joint_policy(agent1, agent2, env):
    nS = env.n_states  # scalar positions per agent (n*n)

    # Best action per joint state (row = pos(agent1), col = pos(agent2))
    q_best1 = np.argmax(agent1.qtable, axis=1).reshape(nS, nS)
    q_best2 = np.argmax(agent2.qtable, axis=1).reshape(nS, nS)

    # --- build "safe" masks for each scalar index (no H or G) ---
    def is_safe_idx(idx):
        r, c = env.idx_to_pos(idx)
        t = env.desc[r, c]
        return t in ("S", "F", "G")

    safe_1 = np.array([is_safe_idx(i) for i in range(nS)])  # agent1 rows
    safe_2 = np.array([is_safe_idx(i) for i in range(nS)])  # agent2 cols
    # joint mask: only plot where BOTH agents are on safe tiles
    M = np.outer(safe_1, safe_2)  # shape (nS, nS), True = keep

    # Quiver fields
    Y, X = np.mgrid[0:nS, 0:nS]
    U1 = np.zeros_like(X, dtype=float);
    V1 = np.zeros_like(Y, dtype=float)
    U2 = np.zeros_like(X, dtype=float);
    V2 = np.zeros_like(Y, dtype=float)

    for s1 in range(nS):
        for s2 in range(nS):
            dy1, dx1 = ACTION_VEC[q_best1[s1, s2]]
            dy2, dx2 = ACTION_VEC[q_best2[s1, s2]]
            U1[s1, s2], V1[s1, s2] = dx1, dy1
            U2[s1, s2], V2[s1, s2] = dx2, dy2

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title("Joint Q-table Policy (Agent1=Black, Agent2=Orange)\n"
                 "Axes show ALL coordinates (row, col)")

    ax.set_xlim(-0.5, nS - 0.5)
    ax.set_ylim(nS - 0.5, -0.5)
    ax.set_xlabel("Agent 2 position (row, col)")
    ax.set_ylabel("Agent 1 position (row, col)")
    ax.grid(True, linestyle="--", linewidth=0.4)

    # Offsets so arrows don’t overlap
    off1x, off1y = AGENT_OFFSETS[1];
    off2x, off2y = AGENT_OFFSETS[2]

    # --- PLOT ONLY WHERE MASK == True ---
    ax.quiver(X[M] + off1x, Y[M] + off1y, U1[M], V1[M],
              color=AGENT_COLORS[1], alpha=0.85, **ARROW_STYLE)
    ax.quiver(X[M] + off2x, Y[M] + off2y, U2[M], V2[M],
              color=AGENT_COLORS[2], alpha=0.85, **ARROW_STYLE)

    # ticks as (row,col)
    ticks = np.arange(nS)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(tuple(map(int, env.idx_to_pos(i)))) for i in ticks], rotation=90, fontsize=8)
    ax.set_yticklabels([str(tuple(map(int, env.idx_to_pos(i)))) for i in ticks], fontsize=8)

    # goal guides (just lines, no arrows on those rows/cols because mask blocks them)
    g_idx = env.pos_to_idx(env.goal)
    ax.axvline(g_idx, color="green", linestyle="--", linewidth=1)
    ax.axhline(g_idx, color="green", linestyle="--", linewidth=1)
    ax.text(g_idx + 0.4, g_idx - 0.6, "Goal", color="green", fontsize=10, weight="bold")

    plt.tight_layout()
    plt.savefig("results/joint_policy_map_quiver.png", bbox_inches="tight", dpi=200)
    plt.show();
    plt.close(fig)


ACTION_NAMES = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}


def debug_episode(env, agent1, agent2, max_steps=200, greedy=True,
                  csv_path="results/debug_episode_log.csv", save_csv=True, verbose_first_n=10):
    """
    Run ONE episode and log, step by step:
      - joint state (and per-agent coords)
      - chosen actions
      - reward components (env, step, proximity, hole)
      - Q-updates for both agents (old, target, td_error, new)
      - a snapshot of each agent's Q-row at the current state
    """
    logs = []
    state = env.reset()

    # optional greedy evaluation
    eps1_bak, eps2_bak = agent1.epsilon, agent2.epsilon
    if greedy:
        agent1.epsilon = 0.0
        agent2.epsilon = 0.0

    for t in range(1, max_steps + 1):
        # Is an agent already parked on the goal at the *current* state?
        s1_idx_curr, s2_idx_curr = env.decode_state(state)
        at_goal1 = env.idx_to_pos(s1_idx_curr) == env.goal
        at_goal2 = env.idx_to_pos(s2_idx_curr) == env.goal
        # Choose actions (frozen agents take a no-op placeholder)
        a1 = 1 if at_goal1 else (np.argmax(agent1.qtable[state, :]) if greedy else agent1.choose_action(state))
        a2 = 1 if at_goal2 else (np.argmax(agent2.qtable[state, :]) if greedy else agent2.choose_action(state))

        # step with info
        ns, r1, r2, done, info = env.step_with_info(a1, a2)

        # ----- compute & apply Q-updates (same as your learner) -----
        # Agent 1
        old_q1 = agent1.qtable[state, a1]
        best_next_1 = np.max(agent1.qtable[ns, :])
        target_1 = r1 + agent1.gamma * best_next_1
        td_err_1 = target_1 - old_q1
        new_q1 = old_q1 + agent1.lr * td_err_1
        agent1.qtable[state, a1] = new_q1

        # Agent 2
        old_q2 = agent2.qtable[state, a2]
        best_next_2 = np.max(agent2.qtable[ns, :])
        target_2 = r2 + agent2.gamma * best_next_2
        td_err_2 = target_2 - old_q2
        new_q2 = old_q2 + agent2.lr * td_err_2
        agent2.qtable[state, a2] = new_q2

        # snapshot current Q-row (for the state we just updated)
        qrow1 = agent1.qtable[state, :].copy()
        qrow2 = agent2.qtable[state, :].copy()

        logs.append({
            "t": t,
            "state_idx": int(state),
            "s1": int(info["s1"]), "s2": int(info["s2"]),
            "s1_yx": info["s1_yx"], "s2_yx": info["s2_yx"],
            "action_a1": int(a1), "action_a1_name": ACTION_NAMES[a1],
            "action_a2": int(a2), "action_a2_name": ACTION_NAMES[a2],
            "ns_idx": int(ns),
            "ns1": int(info["ns1"]), "ns2": int(info["ns2"]),
            "ns1_yx": info["ns1_yx"], "ns2_yx": info["ns2_yx"],
            # reward breakdown
            "r_env_1": info["r_env_1"], "r_env_2": info["r_env_2"],
            "base_step_penalty": info["base_step_penalty"],
            "proximity_gain": info["proximity_gain"],
            "hole_penalty": info["hole_penalty"],
            "dist_before": info["dist_before"], "dist_after": info["dist_after"],
            "reward_r1": info["reward_r1"],
            "reward_r2": info["reward_r2"],
            # Q-update math
            "old_q1": old_q1, "target_1": target_1, "td_err_1": td_err_1, "new_q1": new_q1,
            "old_q2": old_q2, "target_2": target_2, "td_err_2": td_err_2, "new_q2": new_q2,
            # Q row snapshots
            "qrow1_L": qrow1[0], "qrow1_D": qrow1[1], "qrow1_R": qrow1[2], "qrow1_U": qrow1[3],
            "qrow2_L": qrow2[0], "qrow2_D": qrow2[1], "qrow2_R": qrow2[2], "qrow2_U": qrow2[3],
            "done": bool(done),
        })

        state = ns
        if done:
            break

    # restore epsilons
    agent1.epsilon, agent2.epsilon = eps1_bak, eps2_bak

    df = pd.DataFrame(logs)
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"[debug] Saved step-by-step log to {csv_path}")

    # print first few lines for a quick look
    if verbose_first_n > 0:
        print(df.head(verbose_first_n).to_string(index=False))

    return df


def plot_qrow_timeseries(df, which_agent=1, save_path="results/debug_qrow_timeseries.png"):
    """
    Given the df from debug_episode, plot the Q-row (L,D,R,U) for the CURRENT joint state
    as it evolves over the episode. Useful to see why a direction becomes preferred.
    """
    if which_agent == 1:
        cols = ["qrow1_L", "qrow1_D", "qrow1_R", "qrow1_U"]
        title = "Agent 1 Q-row over time (state just updated each step)"
    else:
        cols = ["qrow2_L", "qrow2_D", "qrow2_R", "qrow2_U"]
        title = "Agent 2 Q-row over time (state just updated each step)"

    plt.figure(figsize=(8, 4))
    for c in cols:
        plt.plot(df["t"], df[c], label=c.split("_")[-1])
    plt.title(title)
    plt.xlabel("Step t in episode")
    plt.ylabel("Q-value")
    plt.legend(title="Action")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


def best_action_per_cell(agent, env, who=1):
    """
    For each physical cell of 'who' (1 or 2), pick the action that maximizes
    the sum of Q-values over ALL positions of the *other* agent,
    but only counting other-agent states that are SAFE tiles (S or F).
    Returns (best_action_grid, best_value_grid) of shape (n, n).
    """
    n, nS = env.n, env.n_states
    best_act = np.zeros((n, n), dtype=int)
    best_val = np.zeros((n, n), dtype=float)

    # mask of safe scalar indices (S or F) for either agent
    safe_mask = np.zeros(nS, dtype=bool)
    for idx in range(nS):
        r, c = env.idx_to_pos(idx)
        safe_mask[idx] = env.desc[r, c] in ("S", "F", "G")

    safe_idxs = np.nonzero(safe_mask)[0]

    for r in range(n):
        for c in range(n):
            tile = env.desc[r, c]
            if tile in ("H", "G"):
                # nothing to compute for an unsafe/self-terminal cell
                best_act[r, c] = 0
                best_val[r, c] = np.nan
                continue

            sk = env.pos_to_idx((r, c))  # this agent's scalar state at (r,c)

            if who == 1:
                # rows where agent1 is fixed at sk and agent2 varies over SAFE tiles
                rows = [env.encode_state(sk, s2) for s2 in safe_idxs]
            else:
                # rows where agent2 is fixed at sk and agent1 varies over SAFE tiles
                rows = [env.encode_state(s1, sk) for s1 in safe_idxs]

            # sum Q-values over those rows only (ignore holes/goal for the other agent)
            scores = agent.qtable[rows, :].sum(axis=0)

            a = int(np.argmax(scores))
            best_act[r, c] = a
            best_val[r, c] = scores[a]

    return best_act, best_val


def plot_agent_best_maps_combined(agent1, agent2, env):
    """
    One physical map. For each cell, draw both agents’ best actions
    (max over the other agent’s SAFE positions). Arrows are offset so
    they don’t overlap. No arrows on holes or the goal.
    Coordinate system: (0,0) top-left, (n-1,0) bottom-left.
    """
    n = env.n
    desc = env.desc
    color_map = {"S": "#90ee90", "F": "#add8e6", "H": "#d3d3d3", "G": "#ffd700"}

    # Best action per physical cell for each agent (already excludes H/G in the calc)
    agrid1, _ = best_action_per_cell(agent1, env, who=1)
    agrid2, _ = best_action_per_cell(agent2, env, who=2)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_title("Best Option per Cell (A1=Black, A2=Orange)\n(no arrows on holes or goal)")

    # --- draw map with (row, col) natural placement ---
    for i in range(n):
        for j in range(n):
            tile = desc[i, j]
            ax.add_patch(
                plt.Rectangle((j, i), 1, 1, color=color_map.get(tile, "white"), ec="black", lw=0.7)
            )
            ax.text(j + 0.5, i + 0.5, tile, ha="center", va="center",
                    fontsize=12, weight="bold", color="black")

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)  # <-- invert Y: (0,0) is top-left; (n-1,0) bottom-left
    ax.set_xticks(range(n + 1))
    ax.set_yticks(range(n + 1))
    ax.grid(True, color="black", linewidth=0.4)
    ax.set_aspect('equal')

    # centers in plot coordinates (no flip now)
    Xc_grid, Yc_grid = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    Xc, Yc = Xc_grid, Yc_grid

    def draw_for_agent(agrid, agent_id):
        offx, offy = AGENT_OFFSETS[agent_id]
        U = np.zeros_like(Xc, dtype=float)
        V = np.zeros_like(Yc, dtype=float)
        M = np.zeros_like(Xc, dtype=bool)

        for i in range(n):
            for j in range(n):
                tile = desc[i, j]
                if tile in ("H", "G"):
                    M[i, j] = False
                    continue
                dy, dx = ACTION_VEC[int(agrid[i, j])]
                # with inverted Y-axis, positive dy goes down as desired
                U[i, j] = dx
                V[i, j] = dy
                M[i, j] = True

        ax.quiver(Xc[M] + offx, Yc[M] + offy, U[M], V[M],
                  color=AGENT_COLORS[agent_id], alpha=0.95, **ARROW_STYLE)

    draw_for_agent(agrid1, 1)
    draw_for_agent(agrid2, 2)

    ax.plot([], [], color=AGENT_COLORS[1], label="Agent 1")
    ax.plot([], [], color=AGENT_COLORS[2], label="Agent 2")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("results/agent_best_per_cell_combined.png", bbox_inches="tight", dpi=200)
    plt.show()
    plt.close(fig)


# --- NEW: saving utility ------------------------------------------------------
def save_qtables(env, agent1, agent2,
                 out_npz="results/qtables_final.npz",
                 out_csv_prefix="results/qtable_agent"):
    """
    Save the two Q-tables to disk.

    - NPZ (compressed): includes both tables + map metadata.
    - CSVs: one file per agent (handy for quick inspection).
    """
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)

    # compressed bundle with metadata
    np.savez_compressed(
        out_npz,
        agent1_q=agent1.qtable,
        agent2_q=agent2.qtable,
        map_desc=np.asarray(env.desc),
        n=np.int64(env.n),
        n_states=np.int64(env.n_states),
        n_actions=np.int64(env.n_actions),
        goal_row=np.int64(env.goal[0]),
        goal_col=np.int64(env.goal[1]),
    )

    # optional human-friendly CSVs (one per agent)
    np.savetxt(f"{out_csv_prefix}1.csv", agent1.qtable, delimiter=",")
    np.savetxt(f"{out_csv_prefix}2.csv", agent2.qtable, delimiter=",")

    print(f"[saved] NPZ -> {out_npz}")
    print(f"[saved] CSV -> {out_csv_prefix}1.csv, {out_csv_prefix}2.csv")


# ============================================================
#                           RUN
# ============================================================
if __name__ == "__main__":
    env, agent1, agent2, rewards, steps, mean_q1, mean_q2, epsilons = \
        train_two_agents_representative(
            episodes_per_start=12000,  # 8–20 works well on 4x4
            map_size=4,
            seed=123,
            max_steps=30,
            eps_start=0.4, eps_end=0.02
        )

    # <-- save the final Q-tables
    save_qtables(env, agent1, agent2)
    df_debug = debug_episode(env, agent1, agent2,
                             max_steps=100, greedy=True,
                             csv_path="results/debug_episode_log.csv",
                             save_csv=True, verbose_first_n=12)
    # 1) normal plots
    plot_training_statistics(rewards, steps, mean_q1, mean_q2, epsilons)
    plot_frozenlake_map(env)
    plot_joint_policy(agent1, agent2, env)

    # 2) step-by-step debug of ONE episode (greedy to read the learned policy directly)


    # 3) optional: visualize the evolving Q-row of the updated state
    # plot_qrow_timeseries(df_debug, which_agent=1, save_path="results/debug_qrow_timeseries_agent1.png")
    # plot_qrow_timeseries(df_debug, which_agent=2, save_path="results/debug_qrow_timeseries_agent2.png")
    plot_agent_best_maps_combined(agent1, agent2, env)
