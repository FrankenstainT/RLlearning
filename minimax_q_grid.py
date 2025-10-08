
"""
minimax_q_grid.py
Self-contained Minimax-Q learning example: pursuer vs evader on a small grid maze.
Requires: numpy, scipy, matplotlib
"""

import numpy as np
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# ------------------------
# Environment: Grid maze
# ------------------------
Action = int
ACTIONS = {
    0: (0, 0),   # stay
    1: (-1, 0),  # up
    2: (0, 1),   # right
    3: (1, 0),   # down
    4: (0, -1),  # left
}
N_ACTIONS = len(ACTIONS)

@dataclass
class GridEnv:
    height: int
    width: int
    obstacles: List[Tuple[int,int]]
    safe_cells: List[Tuple[int,int]]
    max_steps: int = 50

    def __post_init__(self):
        self.free_cells = [(r,c) for r in range(self.height) for c in range(self.width)
                           if (r,c) not in set(self.obstacles)]
        self.cell_to_idx = {pos:i for i,pos in enumerate(self.free_cells)}
        self.idx_to_cell = {i:pos for pos,i in self.cell_to_idx.items()}
        self.n_free = len(self.free_cells)
        self.reset()

    def reset(self, randomize: bool = True):
        # choose starting positions (not overlapping)
        if randomize:
            p = random.choice(self.free_cells)
            e = random.choice(self.free_cells)
            while e == p or e in self.safe_cells:
                e = random.choice(self.free_cells)
            # ensure not start in safe zone for evader
        else:
            # deterministic: first free cell for both (not overlapping)
            p = self.free_cells[0]
            e = self.free_cells[min(1, self.n_free-1)]
        self.pursuer = p
        self.evader = e
        self.t = 0
        self.done = False
        return self.get_state()

    def get_state(self) -> Tuple[int,int]:
        return (self.cell_to_idx[self.pursuer], self.cell_to_idx[self.evader])

    def step(self, action_p: Action, action_e: Action):
        if self.done:
            raise RuntimeError("Step called on terminated episode")
        self.t += 1
        # apply actions
        self.pursuer = self._move(self.pursuer, ACTIONS[action_p])
        self.evader = self._move(self.evader, ACTIONS[action_e])

        # terminal checks
        reward = 0.0  # reward to pursuer
        done = False
        # capture
        if self.pursuer == self.evader:
            reward = 1.0
            done = True
        # evader reaches safe zone
        elif self.evader in self.safe_cells:
            reward = -1.0
            done = True
        elif self.t >= self.max_steps:
            reward = 0.0
            done = True

        self.done = done
        return self.get_state(), reward, done, {}

    def _move(self, pos: Tuple[int,int], delta: Tuple[int,int]) -> Tuple[int,int]:
        r,c = pos
        dr,dc = delta
        nr,nc = r+dr, c+dc
        if 0 <= nr < self.height and 0 <= nc < self.width and (nr,nc) not in set(self.obstacles):
            return (nr,nc)
        return pos

# ------------------------
# Minimax solver (LP)
# ------------------------
def solve_zero_sum_game(A: np.ndarray, tol: float = 1e-9):
    """
    Solve zero-sum matrix game for payoff matrix A (m x n), payoff to row player.
    Returns (value, row_policy (size m), col_policy (size n))
    Uses scipy.optimize.linprog (Highs).
    """
    m, n = A.shape
    # LP for row player: maximize v
    # minimize -v with variables [p_0..p_{m-1}, v]
    c = np.zeros(m + 1)
    c[-1] = -1.0  # minimize -v

    # constraints: for each column j: -sum_i A[i,j]*p_i + v <= 0
    A_ub = []
    b_ub = []
    for j in range(n):
        row = np.zeros(m + 1)
        row[:m] = -A[:, j]
        row[-1] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # equality sum p_i = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, 1.0)] * m + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        p = res.x[:m].clip(0,1)
        v = float(res.x[-1])
    else:
        # fallback: uniform
        p = np.ones(m) / m
        v = float(np.min(p @ A))

    # Solve dual to get column player's strategy
    # column player: minimize w s.t. sum_j q_j * A[i,j] <= w for all i
    c2 = np.zeros(n + 1)
    c2[-1] = 1.0  # minimize w
    A_ub2 = []
    b_ub2 = []
    for i in range(m):
        row = np.zeros(n + 1)
        row[:n] = A[i, :]
        row[-1] = -1.0
        A_ub2.append(row)
        b_ub2.append(0.0)
    A_ub2 = np.array(A_ub2)
    b_ub2 = np.array(b_ub2)
    A_eq2 = np.zeros((1, n + 1))
    A_eq2[0, :n] = 1.0
    b_eq2 = np.array([1.0])
    bounds2 = [(0.0, 1.0)] * n + [(None, None)]
    res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2, method='highs')

    if res2.success:
        q = res2.x[:n].clip(0,1)
        w = float(res2.x[-1])
    else:
        q = np.ones(n) / n
        w = float(np.max(A @ q))

    # numerical smoothing
    if np.isnan(v):
        v = 0.0
    value = 0.5*(v + w)  # average in case of minor numeric mismatch
    return value, p / (p.sum() if p.sum()>0 else 1.0), q / (q.sum() if q.sum()>0 else 1.0)

# ------------------------
# Minimax-Q learner
# ------------------------
class MinimaxQLearner:
    def __init__(self, env: GridEnv, alpha=0.1, gamma=0.9, epsilon=0.2, epsilon_decay=0.9995):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Q is a dict: state -> m x n numpy array
        self.Q: Dict[Tuple[int,int], np.ndarray] = {}
        # prefill Q for all possible state pairs (optional)
        for i in range(env.n_free):
            for j in range(env.n_free):
                self.Q[(i,j)] = np.zeros((N_ACTIONS, N_ACTIONS))

    def get_Q(self, state):
        return self.Q[state]

    def select_action(self, state, agent: str = 'pursuer'):
        """
        Returns action for the agent:
        - By default both agents play the current minimax mixed strategy at the state (with epsilon random).
        agent: 'pursuer' or 'evader'
        """
        A = self.get_Q(state)  # payoff matrix for pursuer
        # solve minimax game to get both policies
        value, p_row, q_col = solve_zero_sum_game(A)
        # p_row is pursuer policy; q_col is evader policy
        if agent == 'pursuer':
            base_policy = p_row
        else:
            base_policy = q_col

        # epsilon-greedy wrt the mixed strategy: with prob eps take uniform random action
        if random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        else:
            # sample from base_policy
            return np.random.choice(np.arange(N_ACTIONS), p=base_policy)

    def update(self, state, a_p, a_e, reward, next_state, done):
        Qs = self.get_Q(state)
        q_sa = Qs[a_p, a_e]
        if done:
            v_next = 0.0
        else:
            A_next = self.get_Q(next_state)
            v_next, _, _ = solve_zero_sum_game(A_next)
        target = reward + self.gamma * v_next
        Qs[a_p, a_e] = (1 - self.alpha) * q_sa + self.alpha * target

    def train(self, n_episodes = 3000, verbose: bool = True):
        ep_rewards = []
        moving_avg = []
        ma_window = 50
        for ep in range(1, n_episodes+1):
            s = self.env.reset(randomize=True)
            total_r = 0.0
            done = False
            while not done:
                a_p = self.select_action(s, agent='pursuer')
                a_e = self.select_action(s, agent='evader')
                s2, r, done, _ = self.env.step(a_p, a_e)
                self.update(s, a_p, a_e, r, s2, done)
                s = s2
                total_r += r
            # decay epsilon
            self.epsilon *= self.epsilon_decay
            ep_rewards.append(total_r)
            if ep >= ma_window:
                moving_avg.append(np.mean(ep_rewards[-ma_window:]))
            else:
                moving_avg.append(np.mean(ep_rewards))

            if verbose and (ep % (n_episodes//10) == 0 or ep <= 10):
                print(f"Episode {ep}/{n_episodes}, eps={self.epsilon:.4f}, recent_reward={total_r:.3f}, avg50={moving_avg[-1]:.3f}")

        return ep_rewards, moving_avg

    def evaluate(self, n_eval=200):
        wins = 0
        draws = 0
        for _ in range(n_eval):
            s = self.env.reset(randomize=True)
            done = False
            while not done:
                # use greedy Nash strategies (no epsilon)
                A = self.get_Q(s)
                _, p_row, q_col = solve_zero_sum_game(A)
                a_p = np.random.choice(np.arange(N_ACTIONS), p=p_row)
                a_e = np.random.choice(np.arange(N_ACTIONS), p=q_col)
                s, r, done, _ = self.env.step(a_p, a_e)
            if r == 1.0:
                wins += 1
            elif r == 0.0:
                draws += 1
        return {'pursuer_wins': wins, 'draws': draws, 'evader_wins': n_eval - wins - draws}

# ------------------------
# Example run
# ------------------------
def example_run():
    # define a small maze
    height, width = 5, 6
    obstacles = [(1,1), (1,2), (2,4), (3,2)]
    safe_cells = [(0, width-1)]  # top right corner is safe zone
    env = GridEnv(height=height, width=width, obstacles=obstacles, safe_cells=safe_cells, max_steps=40)

    learner = MinimaxQLearner(env, alpha=0.2, gamma=0.95, epsilon=0.4, epsilon_decay=0.995)
    ep_rewards, moving_avg = learner.train(n_episodes=1500, verbose=True)

    # plot learning curve
    plt.figure(figsize=(8,4))
    plt.plot(ep_rewards, label='episode reward')
    plt.plot(moving_avg, label='moving avg', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward (to pursuer)')
    plt.legend()
    plt.title('Minimax-Q learning: pursuer reward per episode')
    plt.tight_layout()
    plt.show()

    # evaluate
    eval_res = learner.evaluate(n_eval=200)
    print("Evaluation:", eval_res)

if __name__ == "__main__":
    example_run()
