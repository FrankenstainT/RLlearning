import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
#         CONFIG
# =========================
os.makedirs("results", exist_ok=True)
sns.set_theme()

ACTION_VEC = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # (dy, dx) in grid (row, col)


# ============================================================
#                CUSTOM TWO-AGENT FROZEN LAKE ENV
# ============================================================
class TwoAgentFrozenLake:
    """
    Two agents move on a shared 4x4 grid with holes and one goal.
    Each step returns per-agent rewards r1, r2 (team reward = r1+r2).
    If any agent hits a hole -> terminal. If both reach goal -> terminal.
    If an agent is already on G, it stays in place (waits).
    """

    def __init__(self, map_size=4, seed=123, step_penalty=-0.08, prox_coef=0.1):
        self.n = map_size
        self.rng = np.random.default_rng(seed)
        self.desc = np.array([
            list("FFFF"),
            list("FHFH"),
            list("FFFH"),
            list("HFFG"),
        ])
        self.n_states = self.n * self.n
        self.n_actions = 4
        self.goal = (self.n - 1, self.n - 1)
        self.current_state = None
        self.step_penalty = float(step_penalty)
        self.prox_coef = float(prox_coef)

    # ----- Coordinate helpers -----
    def pos_to_idx(self, pos):
        y, x = pos
        return y * self.n + x

    def idx_to_pos(self, idx):
        return divmod(idx, self.n)

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

    def reset_to_indices(self, s1_idx, s2_idx):
        """Reset positions to given scalar indices (can overlap)."""
        self.current_state = (self.idx_to_pos(s1_idx), self.idx_to_pos(s2_idx))
        return self.encode_state(s1_idx, s2_idx)

    def safe_indices(self):
        """All non-hole tiles (optionally excluding G)."""
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

    def step(self, a1, a2):
        (y1, x1), (y2, x2) = self.current_state

        # respect waiting at goal
        ny1, nx1 = (y1, x1) if (y1, x1) == self.goal else self.move(y1, x1, a1)
        ny2, nx2 = (y2, x2) if (y2, x2) == self.goal else self.move(y2, x2, a2)

        tile1, tile2 = self.desc[ny1, nx1], self.desc[ny2, nx2]

        # hole -> terminal; assign -1 to the agent(s) who fell
        r1 = 0.0
        r2 = 0.0
        if tile1 == "H" or tile2 == "H":
            if tile1 == "H":
                r1 = -1.0
            if tile2 == "H":
                r2 = -1.0
            self.current_state = ((ny1, nx1), (ny2, nx2))
            ns = self.encode_state(self.pos_to_idx((ny1, nx1)), self.pos_to_idx((ny2, nx2)))
            return ns, r1, r2, True

        # reaching goal gives +1 (once)
        if (y1, x1) != self.goal and (ny1, nx1) == self.goal:
            r1 += 1.0
        if (y2, x2) != self.goal and (ny2, nx2) == self.goal:
            r2 += 1.0

        # base step penalty and per-agent proximity shaping
        r1 += self.step_penalty
        r2 += self.step_penalty

        gy, gx = self.goal
        # per agent distance change
        r1 += self.prox_coef * ((abs(gy - y1) + abs(gx - x1)) - (abs(gy - ny1) + abs(gx - nx1)))
        r2 += self.prox_coef * ((abs(gy - y2) + abs(gx - x2)) - (abs(gy - ny2) + abs(gx - nx2)))

        done = (tile1 == "G" and tile2 == "G")

        self.current_state = ((ny1, nx1), (ny2, nx2))
        next_state = self.encode_state(self.pos_to_idx((ny1, nx1)), self.pos_to_idx((ny2, nx2)))
        return next_state, r1, r2, done


# ============================================================
#               JOINT ACTION (CENTRALIZED) AGENT
# ============================================================
def pair_to_joint(a1, a2, nA=4):
    return a1 * nA + a2


def joint_to_pair(jidx, nA=4):
    return divmod(jidx, nA)  # -> (a1, a2)


class JointQAgent:
    def __init__(self, state_size, n_actions_per_agent, lr=0.8, gamma=0.95, epsilon=0.1, seed=123):
        self.nA = n_actions_per_agent
        self.qtable = np.zeros((state_size, self.nA * self.nA))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def choose_joint_action(self, state, allowed_mask=None):
        """ε-greedy over joint actions, respecting an optional boolean mask."""
        nJA = self.nA * self.nA
        if allowed_mask is None:
            allowed_mask = np.ones(nJA, dtype=bool)

        if self.rng.uniform() < self.epsilon:
            # explore among allowed
            choices = np.flatnonzero(allowed_mask)
            return int(self.rng.choice(choices))

        # greedy among allowed
        row = self.qtable[state]
        masked_vals = np.where(allowed_mask, row, -np.inf)
        max_val = np.max(masked_vals)
        max_ids = np.flatnonzero(np.isclose(masked_vals, max_val))
        return int(self.rng.choice(max_ids))

    def update(self, s, a_joint, r, ns, next_allowed_mask=None):
        if next_allowed_mask is None:
            best_next = np.max(self.qtable[ns, :])
        else:
            row = self.qtable[ns, :]
            best_next = np.max(np.where(next_allowed_mask, row, -np.inf))
        td_target = r + self.gamma * best_next
        td_error = td_target - self.qtable[s, a_joint]
        self.qtable[s, a_joint] += self.lr * td_error


def allowed_joint_mask(env, state, nA=4, noop_action=1):
    """
    Boolean mask over joint actions (length nA*nA) feasible at 'state'.
    If an agent is at goal, we force its component action to 'noop_action'.
    """
    nJA = nA * nA
    mask = np.ones(nJA, dtype=bool)

    s1_idx, s2_idx = env.decode_state(state)
    at_goal1 = (env.idx_to_pos(s1_idx) == env.goal)
    at_goal2 = (env.idx_to_pos(s2_idx) == env.goal)

    if at_goal1:
        # Only allow joint actions whose a1 == noop_action
        for a1 in range(nA):
            if a1 == noop_action:
                continue
            mask[a1 * nA:(a1 + 1) * nA] = False

    if at_goal2:
        # Only allow joint actions whose a2 == noop_action
        for a1 in range(nA):
            base = a1 * nA
            for a2 in range(nA):
                if a2 != noop_action:
                    mask[base + a2] = False

    return mask


# ============================================================
#                        TRAINING
# ============================================================
def train_joint_q(
        episodes_per_start=100,
        map_size=4,
        seed=123,
        max_steps=30,
        eps_start=0.4,
        eps_end=0.02,
        alpha_start=0.8,
        alpha_end=0.2,
        step_penalty=-0.08,
        prox_coef=0.1,
):
    env = TwoAgentFrozenLake(map_size=map_size, seed=seed,
                             step_penalty=step_penalty, prox_coef=prox_coef)
    rng = np.random.default_rng(seed)

    S = env.n_states ** 2
    nA = env.n_actions
    agent = JointQAgent(S, nA, lr=alpha_start, gamma=0.95, epsilon=eps_start, seed=seed)

    # all safe x safe (overlap allowed), excluding G by default
    safe = env.safe_indices()
    start_pairs = [(s1, s2) for s1 in safe for s2 in safe]
    rng.shuffle(start_pairs)

    total_episodes = len(start_pairs) * episodes_per_start
    # set decay rates so ε≈eps_end and α≈alpha_end near the final episode
    k_eps = np.log(eps_start / eps_end) / max(1, total_episodes - 1)
    k_alpha = np.log(alpha_start / alpha_end) / max(1, total_episodes - 1)

    rewards, steps, mean_q, epsilons, alphas = [], [], [], [], []
    global_ep = 0

    for (s1_idx, s2_idx) in tqdm(start_pairs, desc="Start pairs"):
        for _ in range(episodes_per_start):
            # global monotone schedules
            agent.epsilon = eps_end + (eps_start - eps_end) * np.exp(-k_eps * global_ep)
            agent.lr = alpha_end + (alpha_start - alpha_end) * np.exp(-k_alpha * global_ep)

            state = env.reset_to_indices(s1_idx, s2_idx)
            done, total_r, t = False, 0.0, 0

            while not done and t < max_steps:
                mask = allowed_joint_mask(env, state, nA=nA, noop_action=1)
                a_joint = agent.choose_joint_action(state, allowed_mask=mask)
                a1, a2 = joint_to_pair(a_joint, nA=nA)

                ns, r1, r2, done = env.step(a1, a2)
                r = r1 + r2  # team reward

                next_mask = allowed_joint_mask(env, ns, nA=nA, noop_action=1)
                agent.update(state, a_joint, r, ns, next_allowed_mask=next_mask)

                state = ns
                total_r += r
                t += 1

            rewards.append(total_r)
            steps.append(t)
            mean_q.append(np.mean(agent.qtable))
            epsilons.append(agent.epsilon)
            alphas.append(agent.lr)
            global_ep += 1

    return env, agent, rewards, steps, mean_q, epsilons, alphas


# ============================================================
#                       VISUALIZATION
# ============================================================
def plot_training_statistics(rewards, steps, mean_q, epsilons, alphas, save_path="results/joint_training_stats.png"):
    def smooth(x, k=200):
        if len(x) < k:
            return np.array(x)
        return np.convolve(x, np.ones(k) / k, mode="valid")

    fig, ax = plt.subplots(4, 1, figsize=(10, 12))

    ax[0].plot(smooth(rewards))
    ax[0].set_title("Smoothed Total Reward per Episode")
    ax[0].set_ylabel("Total Reward")

    ax[1].plot(smooth(steps))
    ax[1].set_title("Smoothed Steps per Episode")
    ax[1].set_ylabel("Steps")

    ax[2].plot(smooth(mean_q))
    ax[2].set_title("Mean Q-values (joint table)")
    ax[2].set_ylabel("Q")

    ax[3].plot(epsilons, label="epsilon")
    ax[3].plot(alphas, label="alpha")
    ax[3].legend()
    ax[3].set_title("Exploration (ε) and Learning Rate (α)")
    ax[3].set_ylabel("Value")
    ax[3].set_xlabel("Episode")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


import imageio.v2 as imageio
from pathlib import Path


def greedy_joint_action(agent, state, env, noop_action=1):
    mask = allowed_joint_mask(env, state, nA=agent.nA, noop_action=noop_action)
    row = agent.qtable[state]
    masked_vals = np.where(mask, row, -np.inf)
    max_val = np.max(masked_vals)
    max_ids = np.flatnonzero(np.isclose(masked_vals, max_val))
    a_joint = int(agent.rng.choice(max_ids))
    return a_joint


def draw_frame(env, pos1, pos2, step_idx, save_path):
    """
    Render one frame showing the map and the two agents.
    pos1/pos2 are (row, col). Saves PNG to save_path.
    """
    import matplotlib.pyplot as plt

    n = env.n
    desc = env.desc
    color_map = {"S": "#90ee90", "F": "#add8e6", "H": "#d3d3d3", "G": "#ffd700"}

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_title(f"Joint Q Rollout – step {step_idx}")

    # tiles
    for i in range(n):
        for j in range(n):
            tile = desc[i, j]
            ax.add_patch(
                plt.Rectangle((j, i), 1, 1, color=color_map.get(tile, "white"), ec="black", lw=0.7)
            )
            ax.text(j + 0.5, i + 0.5, tile, ha="center", va="center", fontsize=12, weight="bold")

    # coordinate system: (0,0) top-left
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.set_xticks(range(n + 1))
    ax.set_yticks(range(n + 1))
    ax.grid(True, color="black", linewidth=0.4)

    # agents (slight offset so they don't hide the tile text)
    def draw_agent(rc, color, label):
        r, c = rc
        ax.add_patch(plt.Circle((c + 0.5, r + 0.5), 0.26, fc=color, ec="black", lw=1.0, alpha=0.9))
        ax.text(c + 0.5, r + 0.5, label, ha="center", va="center",
                fontsize=12, color="white", weight="bold")

    draw_agent(pos1, "black", "A1")
    draw_agent(pos2, "darkorange", "A2")

    # goal marker
    gy, gx = env.goal
    ax.add_patch(plt.Rectangle((gx, gy), 1, 1, fill=False, ec="green", lw=2))

    plt.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def rollout_and_record(env, agent, s1_idx, s2_idx, out_dir, max_steps=60, fps=2, noop_action=1):
    """
    Greedy rollout from (s1_idx, s2_idx); saves frames and returns a list of frame paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    state = env.reset_to_indices(s1_idx, s2_idx)
    step = 0

    # initial frame
    (y1, x1), (y2, x2) = env.current_state
    frame_path = out_dir / f"frame_{step:04d}.png"
    draw_frame(env, (y1, x1), (y2, x2), step, frame_path)
    frames.append(frame_path)
    step += 1

    done = False
    while not done and step <= max_steps:
        a_joint = greedy_joint_action(agent, state, env, noop_action=noop_action)
        a1, a2 = joint_to_pair(a_joint, nA=agent.nA)
        state, r1, r2, done = env.step(a1, a2)
        (y1, x1), (y2, x2) = env.current_state

        frame_path = out_dir / f"frame_{step:04d}.png"
        draw_frame(env, (y1, x1), (y2, x2), step, frame_path)
        frames.append(frame_path)
        step += 1

    # If we ended early, duplicate the last frame a few times so the GIF pauses on success.
    for _ in range(5):
        frames.append(frames[-1])

    # Make GIF
    gif_path = out_dir / 'video.gif'
    imgs = [imageio.imread(fp) for fp in frames]
    imageio.mimsave(gif_path, imgs, duration=1.0 / fps)

    # Optional: MP4 (requires imageio-ffmpeg installed)
    try:
        mp4_path = out_dir / 'video.mp4'
        imageio.mimsave(mp4_path, imgs, fps=fps, macro_block_size=None)
    except Exception:
        mp4_path = None  # skip if ffmpeg not available

    return frames, str(gif_path), (str(mp4_path) if mp4_path else None)


def sample_random_starts(env, k=3, seed=0):
    rng = np.random.default_rng(seed)
    safe = env.safe_indices()
    starts = []
    for _ in range(k):
        s1 = int(rng.choice(safe))
        s2 = int(rng.choice(safe))
        starts.append((s1, s2))
    return starts


def all_random_starts(env):
    safe = env.safe_indices()
    starts = []
    for s1 in safe:
        for s2 in safe:
            starts.append((s1, s2))
    return starts


# ============================================================
#                           RUN
# ============================================================
if __name__ == "__main__":
    env, jagent, rewards, steps, mean_q, eps, alphas = train_joint_q(
        episodes_per_start=2000,
        map_size=4,
        seed=123,
        max_steps=30,
        eps_start=0.4, eps_end=0.02,
        alpha_start=0.8, alpha_end=0.2,
        step_penalty=-0.08,  # encourage speed
        prox_coef=0.1,  # distance shaping (can anneal later)
    )

    # Save joint Q-table
    np.savez_compressed(
        "results/joint_qtable_final.npz",
        qtable=jagent.qtable,
        nA=np.int64(jagent.nA),
        n_states=np.int64(env.n_states ** 2),
        map_desc=np.asarray(env.desc),
    )
    print("[saved] results/joint_qtable_final.npz")

    # Plot training curves
    plot_training_statistics(rewards, steps, mean_q, eps, alphas)
    # -------- Rollouts to video --------
    starts = all_random_starts(env)  # sample_random_starts(env, k=3, seed=42)

    for idx, (s1_idx, s2_idx) in enumerate(starts, start=1):
        out_dir = f"results/rollout_{s1_idx}_{s2_idx}"
        frames, gif_path, mp4_path = rollout_and_record(
            env, jagent, s1_idx, s2_idx, out_dir=out_dir, max_steps=60, fps=2, noop_action=1
        )
        print(f"[rollout {idx}] start indices: s1={s1_idx}, s2={s2_idx}")
        print(f"[rollout {idx}] GIF: {gif_path}")
        if mp4_path:
            print(f"[rollout {idx}] MP4: {mp4_path}")
