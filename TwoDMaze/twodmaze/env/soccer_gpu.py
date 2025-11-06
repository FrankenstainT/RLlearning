# soccer.py
import os
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
#                  ACTIONS / UTILS
# ============================================================

ACTIONS = ["N", "S", "W", "E", "Stay"]
DIR = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1), "Stay": (0, 0)}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_and_save(xs, ys, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def moving_average(arr, win=100):
    arr = np.asarray(arr, float)
    if len(arr) < win:
        return arr.copy()
    kernel = np.ones(win) / win
    return np.convolve(arr, kernel, mode="valid")


# ============================================================
#                  ENV: MARKOV SOCCER (Sequential moves)
# ============================================================

@dataclass(frozen=True)
class State:
    Ay: int
    Ax: int
    By: int
    Bx: int
    ball: int  # 0 if A has ball, 1 if B has ball


def owner_goal_distance(env: "MarkovSoccer", Apos, Bpos, ball: int) -> int:
    """
    Heuristic distance from the ball owner to their scoring edge.
    For A (ball==0): horizontal distance to right edge: W-1 - Ax
    For B (ball==1): horizontal distance to left edge: Bx
    """
    (Ay, Ax), (By, Bx) = Apos, Bpos
    return (env.W - 1 - Ax) if ball == 0 else Bx


class MarkovSoccer:
    """
    4x5 grid; **A scores on the right, B scores on the left**. Zero-sum.
    Sequential-moving dynamics with random order each step (50/50 AB vs BA),
    eliminating same-cell states and matching the canonical Markov Soccer setup.
    Reward +1 for A scoring, -1 for B scoring; discount gamma.
    """

    def __init__(self, H: int = 4, W: int = 5, gamma: float = 0.9, seed: int = 0,phi_coeff: float = 0.05, step_penalty_tau: float = 0.01,possession_reward: float = 0.3,
        ball_seek_coeff: float = 0.03,):
        self.H, self.W = H, W
        self.gamma = float(gamma)
        self.rng = random.Random(seed)
        # canonical start positions; ball random each episode
        self.phi_coeff = float(phi_coeff)
        self.tau = float(step_penalty_tau)
        #self._start_no_ball = State(Ay=2, Ax=1, By=1, Bx=3, ball=0)
        self.possession_reward = float(possession_reward)
        self.ball_seek_coeff = float(ball_seek_coeff)
        # enumerate all non-terminal distinct-cell states
        self.states: List[State] = []
        for Ay in range(H):
            for Ax in range(W):
                for By in range(H):
                    for Bx in range(W):
                        if (Ay, Ax) == (By, Bx):
                            continue
                        for ball in (0, 1):
                            self.states.append(State(Ay, Ax, By, Bx, ball))
        self.sid: Dict[State, int] = {s: i for i, s in enumerate(self.states)}

    def in_bounds(self, y, x) -> bool:
        return 0 <= y < self.H and 0 <= x < self.W

    def potential(self, s: State) -> float:
        dA = self.W - 1 - s.Ax  # A’s horizontal distance to right edge
        dB = s.Bx  # B’s horizontal distance to left edge
        if s.ball == 0:
            # A has ball: reward A getting closer (Φ more positive when closer)
            return -self.phi_coeff * float(dA)
        else:
            # B has ball: *penalize A* when B gets closer (Φ more negative when B closer)
            return +self.phi_coeff * float(dB)

    # ----- single-player tentative move (with walls) -----
    def _apply_single(self, y, x, a):
        dy, dx = DIR[a]
        ny, nx = y + dy, x + dx
        if not self.in_bounds(ny, nx):
            return y, x
        return ny, nx

    # ----- goal check (FIXED GOALS):
    # A scores on RIGHT edge by moving E from x == W-1
    # B scores on LEFT edge by moving W from x == 0
    def _score_check(self, s: State, aA: str, aB: str, ball: int):
        # A has ball and tries to exit right
        if ball == 0:
            if s.Ax == self.W - 1 and aA == "E":
                return +1.0, True
        else:
            # B has ball and tries to exit left
            if s.Bx == 0 and aB == "W":
                return -1.0, True
        return 0.0, False

    # ----- sequential step (order = 'AB' or 'BA') -----
    def step_seq(self, s: State, aA: str, aB: str, order: str):
        Ay, Ax, By, Bx, ball = s.Ay, s.Ax, s.By, s.Bx, s.ball

        # scoring based on attempt from edge (fixed to new goals)
        r, done = self._score_check(s, aA, aB, ball)
        if done:
            return s, r, True
        # ----- distances *before* move (for shaping deltas) -----
        # distance between A and B before transition
        prev_dist_AB = abs(Ay - By) + abs(Ax - Bx)
        new_ball = ball

        if order == "AB":
            # A moves first
            ty, tx = self._apply_single(Ay, Ax, aA)
            if (ty, tx) == (By, Bx):
                # into B's current -> cancel, possession to B
                ty, tx = Ay, Ax
                new_ball = 1
            A_new_y, A_new_x = ty, tx

            # B moves second against A's updated position
            ty, tx = self._apply_single(By, Bx, aB)
            if (ty, tx) == (A_new_y, A_new_x):
                ty, tx = By, Bx
                new_ball = 0
            B_new_y, B_new_x = ty, tx

        else:  # 'BA'
            # B moves first
            ty, tx = self._apply_single(By, Bx, aB)
            if (ty, tx) == (Ay, Ax):
                ty, tx = By, Bx
                new_ball = 0
            B_new_y, B_new_x = ty, tx

            # A moves second against B's updated position
            ty, tx = self._apply_single(Ay, Ax, aA)
            if (ty, tx) == (B_new_y, B_new_x):
                ty, tx = Ay, Ax
                new_ball = 1
            A_new_y, A_new_x = ty, tx

        ns = State(A_new_y, A_new_x, B_new_y, B_new_x, new_ball)
        # sanity: never same cell
        assert (ns.Ay, ns.Ax) != (ns.By, ns.Bx), "Invariant violated: same-cell created"
        # ---- per-step penalty from A's perspective ----
        if s.ball == 0:  # A owns ball -> penalize A's dithering
            r -= self.tau
        else:  # B owns ball -> this step is "bad for A" => add (i.e., less negative for A)
            r += self.tau
        swing = False
        # ---- possession swing shaping ----
        if ball == 1 and new_ball == 0:
            r += self.possession_reward  # A stole from B
            swing = True
        elif ball == 0 and new_ball == 1:
            r -= self.possession_reward  # A lost to B
            swing = True

        # ----- chase shaping *by delta* -----
        # distance after transition
        new_dist_AB = abs(ns.Ay - ns.By) + abs(ns.Ax - ns.Bx)
        if not swing:
            if ns.ball == 1:
                # B has ball -> A is the chaser
                # reward A when A gets closer: old - new
                dist_improvement = prev_dist_AB - new_dist_AB
                r += self.ball_seek_coeff * float(dist_improvement)
                # if they stay, dist_improvement = 0 -> no shaping
            else:
                # A has ball -> B is the chaser -> A prefers B to be farther
                # reward A when distance increases: new - old
                dist_separation = new_dist_AB - prev_dist_AB
                r += self.ball_seek_coeff * float(dist_separation)
        # potential-based shaping (preserves optimal policies)
        r += self.gamma * self.potential(ns) - self.potential(s)
        return ns, r, False

    # ----- stochastic one-step: random order 50/50 -----
    def step_det_random(self, s: State, aA: str, aB: str):
        if self.rng.random() < 0.5:
            return self.step_seq(s, aA, aB, "AB")
        else:
            return self.step_seq(s, aA, aB, "BA")

    # def sample_start(self) -> State:
    #     # same positions; ball goes to A or B at random
    #     ball = 0 if self.rng.random() < 0.5 else 1
    #     return State(self._start_no_ball.Ay, self._start_no_ball.Ax,
    #                  self._start_no_ball.By, self._start_no_ball.Bx, ball)


# ============================================================
#     ROBUST ONE-LP SADDLE SOLVER (NUMERICALLY STABLE)
# ============================================================

def solve_both_policies_one_lp(M: np.ndarray, jitter=1e-8, max_retries=2):
    """
    Robust one-LP saddle solver:
    maximize v
      s.t. v <= x^T M[:,j] (∀j),  M[i,:] y <= v (∀i),
           1^T x = 1, x>=0, 1^T y = 1, y>=0
    Returns (x, y, v) for the *original* (unscaled) M.
    """
    M = np.asarray(M, float)
    # sanitize
    if not np.all(np.isfinite(M)):
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    # center & scale for conditioning (affine-invariant policies)
    mu = float(np.mean(M))
    Ms = M - mu
    s = float(np.max(np.abs(Ms)))
    if s < 1e-12:
        # essentially constant matrix -> any x,y; value = that constant
        m, n = M.shape
        x = np.ones(m) / m
        y = np.ones(n) / n
        v = mu
        return x, y, v
    Ms /= s

    def _solve_raw(A):
        m, n = A.shape
        num = m + n + 1
        xs = slice(0, m);
        ys = slice(m, m + n);
        vidx = m + n

        c = np.zeros(num);
        c[vidx] = -1.0
        A_ub = [];
        b_ub = []

        # v <= x^T A[:,j]
        for j in range(n):
            row = np.zeros(num)
            row[xs] = -A[:, j]
            row[vidx] = 1.0
            A_ub.append(row);
            b_ub.append(0.0)

        # A[i,:] y <= v
        for i in range(m):
            row = np.zeros(num)
            row[ys] = A[i, :]
            row[vidx] = -1.0
            A_ub.append(row);
            b_ub.append(0.0)

        A_ub = np.vstack(A_ub);
        b_ub = np.array(b_ub)
        A_eq = np.zeros((2, num));
        A_eq[0, xs] = 1.0;
        A_eq[1, ys] = 1.0
        b_eq = np.array([1.0, 1.0])
        bounds = [(0, None)] * m + [(0, None)] * n + [(None, None)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        return res

    # try solve, with a couple of jittered retries if needed
    A = Ms
    for attempt in range(max_retries + 1):
        res = _solve_raw(A)
        if res.status == 0 and np.isfinite(res.fun):
            z = res.x
            m, n = Ms.shape
            x = np.clip(z[:m], 0, None);
            y = np.clip(z[m:m + n], 0, None);
            v = float(z[m + n])
            sx, sy = x.sum(), y.sum()
            x = x / sx if sx > 0 else np.ones(m) / m
            y = y / sy if sy > 0 else np.ones(n) / n
            # rescale value back
            v = v * s + mu
            return x, y, v
        # add tiny jitter and retry
        A = Ms + np.random.default_rng().normal(scale=jitter, size=Ms.shape)

    # last resort: return uniform to keep training stable
    m, n = M.shape
    x = np.ones(m) / m
    y = np.ones(n) / n
    v = float(np.mean(M))  # harmless placeholder
    return x, y, v


# ============================================================
#     TORCH / GPU GAME SOLVER (extragrad)
# ============================================================
def _torch_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float32
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def project_simplex_torch(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Euclidean projection onto the probability simplex.
    Works on (..., d). Returns tensor of same shape.
    """
    # make last dim the simplex dim
    u, _ = torch.sort(x, descending=True, dim=dim)
    cssv = torch.cumsum(u, dim=dim) - 1
    r = torch.arange(1, x.size(dim) + 1, device=x.device, dtype=x.dtype)
    # shape r to broadcast
    shape = [1] * x.dim()
    shape[dim] = -1
    r = r.view(*shape)
    cond = u * r > cssv
    # rho: index of last True
    rho = cond.sum(dim=dim, keepdim=True) - 1
    theta = cssv.gather(dim, rho) / (rho.to(x.dtype) + 1)
    w = torch.clamp(x - theta, min=0.0)
    # renormalize just in case of tiny numerical drift
    return w / w.sum(dim=dim, keepdim=True)


@torch.no_grad()
def zero_sum_saddle_extragrad_single(
    M_np: np.ndarray,
    lr_list=(0.1, 0.05, 0.02),
    max_steps=8000,
    gap_tol=5e-5,
):
    """
    Solve a single zero-sum matrix game using mirror-prox/extragrad
    on GPU/torch. We try several learning rates and keep the one whose
    value is closest to the true saddle (estimated by the same method).
    Returns (x, y, v).
    """
    device, dtype = _torch_device_and_dtype()
    M = torch.as_tensor(M_np, device=device, dtype=dtype)
    m, n = M.shape

    best = None

    for lr in lr_list:
        # init
        x = torch.full((m,), 1.0 / m, device=device, dtype=dtype)
        y = torch.full((n,), 1.0 / n, device=device, dtype=dtype)

        for step in range(max_steps):
            # current payoffs
            My = M @ y              # (m,)
            xM = (x @ M).view(-1)    # (n,)

            # primal–dual gap
            row_best = My.max()
            col_worst = xM.min()
            gap = (row_best - col_worst).item()

            if gap <= gap_tol:
                break

            # extragrad:
            # 1) extrapolate
            x_tilde = project_simplex_torch(x + lr * My, dim=0)
            y_tilde = project_simplex_torch(y - lr * xM, dim=0)

            # 2) recompute grads at extrapolated point
            My_tilde = M @ y_tilde
            xM_tilde = (x_tilde @ M).view(-1)

            # 3) actual update
            x = project_simplex_torch(x + lr * My_tilde, dim=0)
            y = project_simplex_torch(y - lr * xM_tilde, dim=0)

        v = (x @ M @ y).item()
        err = gap  # we can use the final gap as “quality”

        if (best is None) or (err < best["err"]):
            best = {
                "x": x.detach().cpu().numpy(),
                "y": y.detach().cpu().numpy(),
                "v": float(v),
                "err": err,
            }

    return best["x"], best["y"], best["v"]


# ============================================================
#     UNIFIED GAME SOLVER
# ============================================================
_SOLVE_GAME_TORCH_ANNOUNCED = False

def solve_game(M: np.ndarray):
    """
    Try GPU/torch extragrad first (if available), else fall back to CPU LP.
    Returns (x, y, v).
    """
    global _SOLVE_GAME_TORCH_ANNOUNCED
    # fallback: exact CPU LP
    if not _SOLVE_GAME_TORCH_ANNOUNCED:
        print("[solve_game] using CPU LP solver (scipy.linprog)")
        _SOLVE_GAME_TORCH_ANNOUNCED = True
    return solve_both_policies_one_lp(M)
    has_torch_accel = (
        torch.cuda.is_available()
        or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    )

    if has_torch_accel:
        if not _SOLVE_GAME_TORCH_ANNOUNCED:
            print("[solve_game] using TORCH extragrad solver")
            _SOLVE_GAME_TORCH_ANNOUNCED = True
        return zero_sum_saddle_extragrad_single(M)





# ============================================================
#     SHAPLEY VALUE ITERATION (EXPECTATION OVER ORDERS)
# ============================================================

def stage_matrix(env: MarkovSoccer, s: State, V: Dict[State, float]) -> np.ndarray:
    """
    Expected one-step value under 50/50 AB/BA sequential orders.
    """
    m = len(ACTIONS);
    n = len(ACTIONS)
    M = np.zeros((m, n), float)
    for i, aA in enumerate(ACTIONS):
        for j, aB in enumerate(ACTIONS):
            ns1, r1, d1 = env.step_seq(s, aA, aB, "AB")
            ns2, r2, d2 = env.step_seq(s, aA, aB, "BA")
            v1 = r1 if d1 else (r1 + env.gamma * V[ns1])
            v2 = r2 if d2 else (r2 + env.gamma * V[ns2])
            M[i, j] = 0.5 * (v1 + v2)
    return M


def shapley_value_iteration(env: MarkovSoccer, tol: float = 1e-10, max_iter: int = 2000):
    V = {s: 0.0 for s in env.states}
    PiA = {s: np.ones(len(ACTIONS)) / len(ACTIONS) for s in env.states}
    PiB = {s: np.ones(len(ACTIONS)) / len(ACTIONS) for s in env.states}

    for it in range(max_iter):
        delta = 0.0
        for s in env.states:
            M = stage_matrix(env, s, V)
            x, y, v = solve_game(M)
            delta = max(delta, abs(v - V[s]))
            V[s], PiA[s], PiB[s] = v, x, y
        if delta < tol:
            break
    return V, PiA, PiB


# ============================================================
#                   NASH-Q LEARNER
# ============================================================

class EpsGreedy:
    def __init__(self, eps: float):
        self.eps = float(eps)

    def pick(self, p: np.ndarray, eps: float = None) -> int:
        """ε-greedy pick; if eps is given use it, else fall back to self.eps."""
        p = np.asarray(p, float)
        p = np.clip(p, 0, None)
        ps = p.sum()
        p = p / ps if ps > 0 else np.ones_like(p) / len(p)
        use_eps = self.eps if eps is None else float(eps)
        if np.random.rand() < use_eps:
            return int(np.random.randint(len(p)))
        return int(np.random.choice(len(p), p=p))

    def sample(self, p: np.ndarray) -> int:
        """Pure sampling from a mixed policy (no exploration)."""
        p = np.asarray(p, float)
        p = np.clip(p, 0, None)
        ps = p.sum()
        p = p / ps if ps > 0 else np.ones_like(p) / len(p)
        return int(np.random.choice(len(p), p=p))


class NashQLearner:
    def __init__(self, gamma=0.9, alpha0=0.5, alpha_power=0.6,
                 eps_init=0.3, eps_final=0.02,
                 eps_power=0.6):
        self.gamma = float(gamma)
        self.alpha0 = float(alpha0)
        self.alpha_power = float(alpha_power)

        # ---- state-adaptive ε(s) params ----
        self.eps_init = float(eps_init)
        self.eps_min = float(eps_final)
        self.eps_power = float(eps_power)  # 0.5~1.0 typical

        self.epsA = EpsGreedy(self.eps_init)
        self.epsB = EpsGreedy(self.eps_init)
        self.ep_alpha_sum = 0.0  # <--- NEW
        self.ep_alpha_count = 0  # <--- NEW
        self.alpha_per_episode = []
        self.Q: Dict[State, Dict[Tuple[int, int], float]] = {}
        self.V: Dict[State, float] = {}
        self.PiA: Dict[State, np.ndarray] = {}
        self.PiB: Dict[State, np.ndarray] = {}
        self.vis: Dict[State, Dict[Tuple[int, int], int]] = {}

        self.state_visits: Dict[State, int] = {}
        self.dirty: set = set()

        self.state: State = None  # type: ignore
        self.last_pair: Tuple[int, int] = None  # type: ignore

        self.episode_deltas: List[float] = []
        self._ep_max_delta = 0.0

        self.q_clip = 1.0 / max(1e-6, (1.0 - self.gamma))

    def _epsilon_for_state(self, s: State) -> float:
        n = self.state_visits.get(s, 0)
        # ε(s) = max(ε_min, ε_init / (1+n)^eps_power)
        return max(self.eps_min, self.eps_init / ((1.0 + n) ** self.eps_power))

    def _ensure(self, s: State):
        if s not in self.Q:
            self.Q[s] = {(i, j): 0.0 for i in range(len(ACTIONS)) for j in range(len(ACTIONS))}
            self.V[s] = 0.0
            self.PiA[s] = np.ones(len(ACTIONS)) / len(ACTIONS)
            self.PiB[s] = np.ones(len(ACTIONS)) / len(ACTIONS)
            self.vis[s] = {(i, j): 0 for i in range(len(ACTIONS)) for j in range(len(ACTIONS))}
            self.state_visits.setdefault(s, 0)
            self.dirty.add(s)

    def _matrix_from_Q(self, s: State) -> np.ndarray:
        self._ensure(s)
        M = np.zeros((len(ACTIONS), len(ACTIONS)), float)
        for i in range(len(ACTIONS)):
            for j in range(len(ACTIONS)):
                M[i, j] = self.Q[s][(i, j)]
        return M

    def _solve(self, s: State):
        self._ensure(s)
        if s in self.dirty:
            M = self._matrix_from_Q(s)
            try:
                x, y, v = solve_game(M)
            except Exception:
                M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
                x, y, v = solve_game(M)
            self.PiA[s], self.PiB[s], self.V[s] = x, y, v
            self.dirty.discard(s)
        return self.PiA[s], self.PiB[s], self.V[s]

    def start(self, s: State):
        self.state = s
        self._ensure(s)
        self._solve(s)
        # entering a state counts as a visit for ε(s)
        self.state_visits[s] = self.state_visits.get(s, 0) + 1  # <--- NEW

    def act(self) -> Tuple[int, int]:
        s = self.state
        x, y, _ = self._solve(s)
        eps_s = self._epsilon_for_state(s)  # <--- state-adaptive epsilon
        aA = self.epsA.pick(x, eps=eps_s)
        aB = self.epsB.pick(y, eps=eps_s)
        self.last_pair = (aA, aB)
        return aA, aB

    def _alpha(self, s: State, aA: int, aB: int) -> float:
        n = self.vis[s][(aA, aB)]
        return self.alpha0 / ((1.0 + n) ** self.alpha_power)

    def observe(self, s_next: State, reward: float, done: bool):
        s = self.state
        aA, aB = self.last_pair
        self._ensure(s);
        self._ensure(s_next)

        self._solve(s_next)

        target = reward + (0.0 if done else self.gamma * self.V[s_next])
        old_q = self.Q[s][(aA, aB)]
        alpha = self._alpha(s, aA, aB)
        # log alpha used on this update
        self.ep_alpha_sum += alpha  # <--- NEW
        self.ep_alpha_count += 1
        new_q = old_q + alpha * (target - old_q)

        bound = self.q_clip
        new_q = float(np.clip(new_q, -bound, bound))

        self.Q[s][(aA, aB)] = new_q
        self.vis[s][(aA, aB)] += 1
        self._ep_max_delta = max(self._ep_max_delta, abs(new_q - old_q))

        self.dirty.add(s)
        self.state = s_next
        # count visit for next state as we arrive there
        self.state_visits[s_next] = self.state_visits.get(s_next, 0) + 1

    def end_episode(self):
        self.episode_deltas.append(self._ep_max_delta)
        # push mean alpha for this episode; guard div-by-zero
        mean_alpha = (self.ep_alpha_sum / self.ep_alpha_count) if self.ep_alpha_count else 0.0
        self.alpha_per_episode.append(mean_alpha)  # <--- NEW
        # reset episode accumulators
        self.ep_alpha_sum = 0.0  # <--- NEW
        self.ep_alpha_count = 0
        self._ep_max_delta = 0.0

    def snapshot_policies(self):
        """
        Resolve policies for all known states and return a deep snapshot.
        Format: {"A": {state: np.array}, "B": {state: np.array}}
        """
        for s in list(self.Q.keys()):
            self._solve(s)
        snapA = {s: self.PiA[s].copy() for s in self.PiA}
        snapB = {s: self.PiB[s].copy() for s in self.PiB}
        return {"A": snapA, "B": snapB}

    @staticmethod
    def _l1(a, b):
        return float(np.sum(np.abs(np.asarray(a) - np.asarray(b))))

    def policy_drift(self, prev_snapshot):
        """
        Compare current policies to prev_snapshot using L1 & TV (0.5*L1).
        Returns {"per_state": ..., "agg": ..., "current_snapshot": ...}
        """
        cur = self.snapshot_policies()
        # union of states
        states = set(cur["A"].keys()) | set(prev_snapshot["A"].keys()) | \
                 set(cur["B"].keys()) | set(prev_snapshot["B"].keys())
        nA = len(ACTIONS)
        per_state = {}
        l1A, l1B, tvA, tvB = [], [], [], []

        def get_or_uniform(dic, s):
            if s in dic:
                return dic[s]
            return np.ones(nA) / nA

        for s in states:
            a_prev = get_or_uniform(prev_snapshot["A"], s)
            b_prev = get_or_uniform(prev_snapshot["B"], s)
            a_cur = get_or_uniform(cur["A"], s)
            b_cur = get_or_uniform(cur["B"], s)

            la = self._l1(a_prev, a_cur)
            lb = self._l1(b_prev, b_cur)
            ta, tb = 0.5 * la, 0.5 * lb
            per_state[s] = {"l1_A": la, "l1_B": lb, "tv_A": ta, "tv_B": tb}
            l1A.append(la);
            l1B.append(lb);
            tvA.append(ta);
            tvB.append(tb)

        def agg(xs, fn):
            return float(fn(xs)) if xs else 0.0

        agg_stats = {
            "count": len(states),
            "l1_A_max": agg(l1A, np.max), "l1_A_mean": agg(l1A, np.mean), "l1_A_median": agg(l1A, np.median),
            "l1_B_max": agg(l1B, np.max), "l1_B_mean": agg(l1B, np.mean), "l1_B_median": agg(l1B, np.median),
            "tv_A_max": agg(tvA, np.max), "tv_A_mean": agg(tvA, np.mean), "tv_A_median": agg(tvA, np.median),
            "tv_B_max": agg(tvB, np.max), "tv_B_mean": agg(tvB, np.mean), "tv_B_median": agg(tvB, np.median),
        }
        return {"per_state": per_state, "agg": agg_stats, "current_snapshot": cur}


# ============================================================
#                    EXPLOITABILITY (optional)
# ============================================================

def exploitability_for_state(M: np.ndarray, x: np.ndarray, y: np.ndarray, v: float) -> float:
    row_vals = M @ y
    col_vals = M.T @ x
    row_gain = float(row_vals.max() - v)
    col_gain = float(v - col_vals.min())
    return max(row_gain, col_gain)


# def evaluate_against_ground_truth(env: MarkovSoccer,
#                                   V_star, PiA_star, PiB_star,
#                                   PiA_learned, PiB_learned):
#     eps = []
#     for s in env.states:
#         M = stage_matrix(env, s, V_star)
#         x, y, v = solve_game(M)
#         x = PiA_learned.get(s, np.ones(len(ACTIONS)) / len(ACTIONS))
#         y = PiB_learned.get(s, np.ones(len(ACTIONS)) / len(ACTIONS))
#         eps.append(exploitability_for_state(M, x, y, v))
#     return float(np.max(eps)), float(np.mean(eps))


# ============================================================
#                      SAVE / PLOT HELPERS
# ============================================================

def save_q_tables_pickle(learner: NashQLearner, outpath: str):
    import pickle
    mats = {}
    nA = len(ACTIONS)
    for s in learner.Q.keys():
        M = np.zeros((nA, nA), float)
        for i in range(nA):
            for j in range(nA):
                M[i, j] = learner.Q[s][(i, j)]
        mats[s] = M
    with open(outpath, "wb") as f:
        pickle.dump(mats, f)


def nash_report_all_states(learner: NashQLearner, outdir: str, tol: float = 1e-4):
    """
    For every state that appears in learner.Q:
      - build stage matrix from current Q
      - solve (x,y,v)
      - compute exploitability ε
    Saves a text report, returns a summary dict.
    """
    ensure_dir(outdir)
    nA = len(ACTIONS)
    worst = (-1.0, None, None, None)  # (eps, state, row_gain, col_gain)
    eps_list = []
    lines = []

    for s in learner.Q.keys():
        M = np.zeros((nA, nA), float)
        for i in range(nA):
            for j in range(nA):
                M[i, j] = learner.Q[s][(i, j)]
        # solve on the *current Q* stage-game
        x, y, v = solve_game(M)
        # exploitability
        row_vals = M @ y
        col_vals = M.T @ x
        row_gain = float(row_vals.max() - v)
        col_gain = float(v - col_vals.min())
        eps = max(row_gain, col_gain)

        eps_list.append(eps)
        if eps > worst[0]:
            worst = (eps, s, row_gain, col_gain)
        lines.append(
            f"state={s}\n  v={v:.6g}  eps={eps:.3e}  row_gain={row_gain:.3e}  col_gain={col_gain:.3e}\n"
            f"  x={np.round(x, 6)}\n  y={np.round(y, 6)}\n"
        )

    if eps_list:
        arr = np.asarray(eps_list, float)
        summary = {
            "num_states": len(eps_list),
            "eps_max": float(np.max(arr)),
            "eps_mean": float(np.mean(arr)),
            "eps_median": float(np.median(arr)),
            "within_tol": int(np.sum(arr <= tol)),
            "tol": tol,
            "worst_state": worst[1],
            "worst_eps": worst[0],
            "worst_row_gain": worst[2],
            "worst_col_gain": worst[3],
        }
    else:
        summary = {
            "num_states": 0, "eps_max": None, "eps_mean": None, "eps_median": None,
            "within_tol": 0, "tol": tol, "worst_state": None, "worst_eps": None,
            "worst_row_gain": None, "worst_col_gain": None,
        }

    # save text report
    report_path = os.path.join(outdir, "nash_exploitability_report.txt")
    with open(report_path, "w") as f:
        f.write("Per-state ε-Nash (exploitability) report\n")
        f.write(f"Tolerance: {tol}\n\n")
        for ln in lines:
            f.write(ln + "\n")
        f.write("\n=== Summary ===\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    # also print brief summary
    print("[Nash report]", summary)
    return summary, report_path


def draw_frame(env: MarkovSoccer, s: State, step_idx: int, out_dir: str):
    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(6, 5))

    for y in range(env.H + 1):
        ax.plot([-0.5, env.W - 0.5], [y - 0.5, y - 0.5], linewidth=0.5, color="gray")
    for x in range(env.W + 1):
        ax.plot([x - 0.5, x - 0.5], [-0.5, env.H - 0.5], linewidth=0.5, color="gray")

    ax.fill_betweenx([-0.5, env.H - 0.5], -1.0, -0.5, alpha=0.12, color="purple", label="B Goal (left)")
    ax.fill_betweenx([-0.5, env.H - 0.5], env.W - 0.5, env.W, alpha=0.12, color="green", label="A Goal (right)")

    ax.plot([s.Ax], [s.Ay], marker='o', markersize=12, linestyle='None', label='A', color="tab:blue")
    ax.plot([s.Bx], [s.By], marker='s', markersize=12, linestyle='None', label='B', color="tab:red")

    by, bx = (s.Ay, s.Ax) if s.ball == 0 else (s.By, s.Bx)
    circ = plt.Circle((bx, by), radius=0.2, fill=False, linewidth=2.5, color="black")
    ax.add_patch(circ)

    ax.invert_yaxis()
    ax.set_xlim([-0.6, env.W - 0.4]);
    ax.set_ylim([env.H - 0.6, -0.6])
    ax.set_xticks(range(env.W));
    ax.set_yticks(range(env.H))
    ax.set_title(f"Greedy Evaluation — Step {step_idx}")
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"frame_{step_idx:04d}.png")
    plt.savefig(fname)
    plt.close(fig)
    return fname


def frames_to_gif_mp4(frames_dir: str, out_gif: str, out_mp4: str, fps: int = 3):
    files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
                    if f.lower().endswith(".png")])
    if not files:
        print(f"No frames in {frames_dir}")
        return
    imgs = [imageio.imread(f) for f in files]
    imageio.mimsave(out_gif, imgs, duration=1.0 / fps, loop=0, subrectangles=False)
    imageio.mimsave(out_mp4, imgs, fps=fps, macro_block_size=None)
    print(f"Saved video: {out_gif}")
    print(f"Saved video: {out_mp4}")


# ============================================================
#                          MAIN
# ============================================================

def main():
    outdir = "soccer_mild_stat"
    ensure_dir(outdir)
    WATCH_EPISODE = 2_685_000
    big_drift_records = []

    np.random.seed(0)
    random.seed(0)

    env = MarkovSoccer()  # 4x5, gamma=0.9

    # print("Computing ground-truth Nash via Shapley value iteration...")
    # V_star, PiA_star, PiB_star = shapley_value_iteration(env, tol=1e-10, max_iter=2000)
    # sA = State(2, 1, 1, 3, 0)
    # sB = State(2, 1, 1, 3, 1)
    # with open(os.path.join(outdir, "ground_truth_starts.txt"), "w") as f:
    #     f.write(f"V*(start A-ball) = {V_star[sA]:.6f}\n")
    #     f.write(f"pi_A*(A-ball) over {ACTIONS} = {np.round(PiA_star[sA], 6)}\n")
    #     f.write(f"pi_B*(A-ball) over {ACTIONS} = {np.round(PiB_star[sA], 6)}\n\n")
    #     f.write(f"V*(start B-ball) = {V_star[sB]:.6f}\n")
    #     f.write(f"pi_A*(B-ball) over {ACTIONS} = {np.round(PiA_star[sB], 6)}\n")
    #     f.write(f"pi_B*(B-ball) over {ACTIONS} = {np.round(PiB_star[sB], 6)}\n")
    def all_distinct_starts(H: int, W: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Return all (Ay,Ax),(By,Bx) with A!=B (no overlap)."""
        cells = [(y, x) for y in range(H) for x in range(W)]
        pairs = []
        for Ay, Ax in cells:
            for By, Bx in cells:
                if (Ay, Ax) != (By, Bx):
                    pairs.append(((Ay, Ax), (By, Bx)))
        return pairs

    # ----- curriculum: each (A,B) trained with ball=A and ball=B equally -----
    pairs = list(all_distinct_starts(env.H, env.W))  # (Apos, Bpos)
    starts_with_ball = []
    for Apos, Bpos in pairs:
        starts_with_ball.append((Apos, Bpos, 0))  # ball to A
        starts_with_ball.append((Apos, Bpos, 1))  # ball to B

    # sort ascending by distance(owner → goal)
    starts_with_ball.sort(key=lambda t: owner_goal_distance(env, t[0], t[1], t[2]))

    # how many episodes for EACH (A,B,ball) triple
    EPISODES_PER_TRIPLE = 1000  #  (even across balls)

    learner = NashQLearner(
        gamma=env.gamma,
        alpha0=0.6,
        alpha_power=0.4,
        eps_init=0.6,
        eps_final=0.01,
        eps_power=0.25,  # ε(s) ∝ (1+visits)^-0.6
    )

    episode_lengths = []
    disc_returns = []
    eps_sched = []  # we'll log ε of the initial state each episode (for plots)

    # --- policy drift trackers ---
    policy_drift_rows = []
    prev_snapshot = learner.snapshot_policies()
    def is_corner(pos, H, W):
        y, x = pos
        return (y in (0, H - 1)) and (x in (0, W - 1))

    def on_top_or_bottom_border(pos, H):
        y, _ = pos
        return y in (0, H - 1)

    def start_weight(Apos, Bpos, H, W):
        # If any is corner -> strongest weighting
        if is_corner(Apos, H, W) and is_corner(Bpos, H, W):
            return 16
        if is_corner(Apos, H, W) and on_top_or_bottom_border(Bpos, H) or on_top_or_bottom_border(Apos, H) and is_corner(Bpos, H, W):
            return 8
        # Else if any is on top/bottom border -> medium weighting
        if on_top_or_bottom_border(Apos, H) and on_top_or_bottom_border(Bpos, H) or is_corner(Apos, H, W) or is_corner(Bpos, H, W):
            return 4
        if on_top_or_bottom_border(Apos, H) or on_top_or_bottom_border(Bpos, H):
            return 2
        # Otherwise -> base
        return 1
    ep = 0

    for (Apos, Bpos, ball) in starts_with_ball:
        mult = start_weight(Apos, Bpos, env.H, env.W)
        repeats = EPISODES_PER_TRIPLE * mult
        for _rep in range(repeats):
            Ay, Ax = Apos
            By, Bx = Bpos
            s = State(Ay=Ay, Ax=Ax, By=By, Bx=Bx, ball=ball)

            learner.start(s)
            # log the ε used at the episode start (for visualization)
            eps_sched.append(learner._epsilon_for_state(s))

            G = 0.0
            t = 0
            while True:
                aA_idx, aB_idx = learner.act()
                aA, aB = ACTIONS[aA_idx], ACTIONS[aB_idx]
                ns, r, done = env.step_det_random(s, aA, aB)
                G += (env.gamma ** t) * r
                learner.observe(ns, r, done)
                s = ns
                t += 1
                if done or t > 100:
                    learner.end_episode()
                    episode_lengths.append(t)
                    disc_returns.append(G)
                    break

            # --- policy drift (L1 / TV) between episodes ---
            drift = learner.policy_drift(prev_snapshot)

            if ep >= WATCH_EPISODE:
                cur_snap = drift["current_snapshot"]  # {"A": {state: np.array}, "B": {...}}
                prev_snap = prev_snapshot  # snapshot from previous episode

                for s, stats in drift["per_state"].items():
                    la = stats["l1_A"]
                    lb = stats["l1_B"]
                    if la > 1.0 or lb > 1.0:
                        # get policies now (after update)
                        cur_piA = cur_snap["A"].get(s, np.ones(len(ACTIONS)) / len(ACTIONS))
                        cur_piB = cur_snap["B"].get(s, np.ones(len(ACTIONS)) / len(ACTIONS))

                        # get policies before update
                        prev_piA = prev_snap["A"].get(s, np.ones(len(ACTIONS)) / len(ACTIONS))
                        prev_piB = prev_snap["B"].get(s, np.ones(len(ACTIONS)) / len(ACTIONS))

                        big_drift_records.append({
                            "episode": ep,
                            "state": (s.Ay, s.Ax, s.By, s.Bx, s.ball),
                            "l1_A": float(la),
                            "l1_B": float(lb),
                            "prev_piA": prev_piA.tolist(),
                            "prev_piB": prev_piB.tolist(),
                            "cur_piA": cur_piA.tolist(),
                            "cur_piB": cur_piB.tolist(),
                        })

            prev_snapshot = drift["current_snapshot"]
            policy_drift_rows.append({
                "episode": ep,
                "states": drift["agg"]["count"],
                "l1_A_max": drift["agg"]["l1_A_max"], "l1_A_mean": drift["agg"]["l1_A_mean"],
                "l1_B_max": drift["agg"]["l1_B_max"], "l1_B_mean": drift["agg"]["l1_B_mean"],
                "tv_A_max": drift["agg"]["tv_A_max"], "tv_A_mean": drift["agg"]["tv_A_mean"],
                "tv_B_max": drift["agg"]["tv_B_max"], "tv_B_mean": drift["agg"]["tv_B_mean"],
            })

            ep += 1
            if (ep % 5000) == 0:
                last = disc_returns[-5000:] if len(disc_returns) >= 5000 else disc_returns
                print(f"Episode {ep}: avg discounted return (last block) = {np.mean(last):.4f}, "
                      f"ΔQ_max_last={learner.episode_deltas[-1]:.3e}, eps0≈{eps_sched[-1]:.3f}")

    qpath = os.path.join(outdir, "nash_q_tables.pkl")
    save_q_tables_pickle(learner, qpath)
    print(f"Saved Nash-Q tables to {qpath}")
    # -------- Nash exploitability report (print + save) --------
    summary, report_path = nash_report_all_states(learner, outdir=outdir, tol=1e-4)
    print(f"[saved] exploitability report -> {report_path}")

    plot_and_save(range(1, len(episode_lengths) + 1), episode_lengths,
                  "Episode Lengths", "Episode", "Steps",
                  os.path.join(outdir, "episode_lengths.png"))

    plot_and_save(range(1, len(disc_returns) + 1), disc_returns,
                  "Discounted Return per Episode", "Episode", "G",
                  os.path.join(outdir, "discounted_return.png"))

    ma = moving_average(disc_returns, 500)
    plot_and_save(range(len(ma)), ma,
                  "Discounted Return (Moving Average, 500)", "MA Index", "G (avg)",
                  os.path.join(outdir, "discounted_return_ma500.png"))

    plot_and_save(range(1, len(learner.episode_deltas) + 1), learner.episode_deltas,
                  "Per-episode Max |ΔQ|", "Episode", "max |ΔQ|",
                  os.path.join(outdir, "q_convergence.png"))

    plot_and_save(range(1, len(eps_sched) + 1), eps_sched,
                  "Epsilon Schedule (A)", "Episode", "ε",
                  os.path.join(outdir, "epsilon.png"))
    plot_and_save(range(1, len(learner.alpha_per_episode) + 1), learner.alpha_per_episode,
                  "Learning Rate (alpha) per Episode", "Episode", "alpha",
                  os.path.join(outdir, "alpha_per_episode.png"))

    # -------- Policy drift outputs --------
    import csv
    drift_csv = os.path.join(outdir, "policy_drift.csv")
    with open(drift_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(policy_drift_rows[0].keys()))
        w.writeheader()
        w.writerows(policy_drift_rows)
    print(f"[saved] {drift_csv}")
    import json
    big_drift_path = os.path.join(outdir, "big_policy_drifts.json")
    with open(big_drift_path, "w") as f:
        json.dump(big_drift_records, f, indent=2)
    print(f"[saved] big per-state drifts -> {big_drift_path}")

    xs = [r["episode"] for r in policy_drift_rows]

    # TV means
    plot_and_save(xs, [r["tv_A_mean"] for r in policy_drift_rows],
                  "Policy Drift — TV mean (A)", "Episode", "TV(A)",
                  os.path.join(outdir, "policy_drift_tv_mean_A.png"))
    plot_and_save(xs, [r["tv_B_mean"] for r in policy_drift_rows],
                  "Policy Drift — TV mean (B)", "Episode", "TV(B)",
                  os.path.join(outdir, "policy_drift_tv_mean_B.png"))

    # L1 means
    plot_and_save(xs, [r["l1_A_mean"] for r in policy_drift_rows],
                  "Policy Drift — L1 mean (A)", "Episode", "L1(A)",
                  os.path.join(outdir, "policy_drift_l1_mean_A.png"))
    plot_and_save(xs, [r["l1_B_mean"] for r in policy_drift_rows],
                  "Policy Drift — L1 mean (B)", "Episode", "L1(B)",
                  os.path.join(outdir, "policy_drift_l1_mean_B.png"))

    # (Optional) L1 max
    plot_and_save(xs, [r["l1_A_max"] for r in policy_drift_rows],
                  "Policy Drift — L1 max (A)", "Episode", "L1_max(A)",
                  os.path.join(outdir, "policy_drift_l1_max_A.png"))
    plot_and_save(xs, [r["l1_B_max"] for r in policy_drift_rows],
                  "Policy Drift — L1 max (B)", "Episode", "L1_max(B)",
                  os.path.join(outdir, "policy_drift_l1_max_B.png"))

    # if eval_points:
    #     plot_and_save(eval_points, eval_max_eps,
    #                   "Exploitability: max ε vs episodes", "Episode", "max ε",
    #                   os.path.join(outdir, "exploitability_max.png"))
    #     plot_and_save(eval_points, eval_mean_eps,
    #                   "Exploitability: mean ε vs episodes", "Episode", "mean ε",
    #                   os.path.join(outdir, "exploitability_mean.png"))

    # ============================================================
    #   FINAL GREEDY EVAL: all (A,B) starts once; record videos
    # ============================================================
    print("\n=== Final Greedy Evaluation: all starts once (ball randomized), with per-start videos ===")

    import csv

    # helper: enumerate all distinct non-overlapping starts
    def all_distinct_starts(H: int, W: int):
        cells = [(y, x) for y in range(H) for x in range(W)]
        for Ay, Ax in cells:
            for By, Bx in cells:
                if (Ay, Ax) != (By, Bx):
                    yield (Ay, Ax), (By, Bx)

    # ensure we have solved policies for a state before sampling actions
    def ensure_policy(state: State):
        learner._ensure(state)
        learner._solve(state)
        return learner.PiA[state], learner.PiB[state]


    max_steps_eval = 100
    eval_rows = []
    a_wins = b_wins = truncs = total_steps = 0

    idx = 0
    for (Apos, Bpos) in all_distinct_starts(env.H, env.W):
        Ay, Ax = Apos
        By, Bx = Bpos
        for ball in (0, 1):  # <-- evaluate both A-ball and B-ball
            idx += 1
            s = State(Ay=Ay, Ax=Ax, By=By, Bx=Bx, ball=ball)
            ensure_policy(s)

            case_name = f"eval_A({Ay},{Ax})_B({By},{Bx})_ball{'A' if ball == 0 else 'B'}"
            frames_dir = os.path.join(outdir, case_name)
            ensure_dir(frames_dir)

            # rollout (ε = 0, sample from learned mixed policies)
            step = 0
            discounted_return = 0.0
            draw_frame(env, s, step, frames_dir)

            winner = "None"
            while True:
                x, y = ensure_policy(s)
                aA_idx = EpsGreedy(0).sample(x)
                aB_idx = EpsGreedy(0).sample(y)
                aA, aB = ACTIONS[aA_idx], ACTIONS[aB_idx]

                ns, r, done = env.step_det_random(s, aA, aB)
                discounted_return += (env.gamma ** step) * r
                s = ns
                step += 1
                draw_frame(env, s, step, frames_dir)

                if done or step >= max_steps_eval:
                    if done:
                        if r > 0:
                            winner = "A"; a_wins += 1
                        elif r < 0:
                            winner = "B"; b_wins += 1
                        else:
                            winner = "Tie"
                    else:
                        winner = "Trunc"; truncs += 1
                    total_steps += step
                    break

            gif_path = os.path.join(frames_dir, f"{case_name}.gif")
            mp4_path = os.path.join(frames_dir, f"{case_name}.mp4")
            try:
                frames_to_gif_mp4(frames_dir, gif_path, mp4_path, fps=3)
            except Exception as e:
                print(f"[video export skipped for {case_name}] {e}")
                gif_path, mp4_path = "", ""

            eval_rows.append({
                "idx": idx,
                "Ay": Ay, "Ax": Ax, "By": By, "Bx": Bx,
                "ball": ball,  # 0 = A, 1 = B
                "steps": step,
                "discounted_return": discounted_return,
                "winner": winner,
                "frames_dir": frames_dir,
                "gif": gif_path,
                "mp4": mp4_path,
            })

            if (idx % 100) == 0:
                print(f"  evaluated {idx} start-cases…")

    # save CSV summary
    eval_csv = os.path.join(outdir, "eval_all_starts_with_videos.csv")
    if eval_rows:
        with open(eval_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
            w.writeheader();
            w.writerows(eval_rows)
        print(f"[saved] per-start eval summary -> {eval_csv}")

    # aggregate prints & plots
    n = len(eval_rows)
    avg_steps = total_steps / max(1, n)
    print(f"Final eval summary over {n} starts:")
    print(f"  A wins: {a_wins} | B wins: {b_wins} | truncations: {truncs}")
    print(f"  avg steps per start: {avg_steps:.2f}")

    try:
        winners_list = [r["winner"] for r in eval_rows]
        labels, counts = np.unique(winners_list, return_counts=True)

        plt.figure()
        plt.bar(labels, counts)
        plt.title("Final Eval — Winner counts over all starts")
        plt.xlabel("Outcome");
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "final_eval_winner_counts.png"))
        plt.close()

        plt.figure()
        plt.hist([r["steps"] for r in eval_rows], bins=20)
        plt.title("Final Eval — Steps per start (histogram)")
        plt.xlabel("Steps");
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "final_eval_steps_hist.png"))
        plt.close()
    except Exception as e:
        print(f"[final eval plotting skipped] {e}")

    # env.close()

    with open(os.path.join(outdir, "training_summary.txt"), "w") as f:
        f.write(f"Episodes: {ep}\n")
        f.write(f"Final epsilon ~ {eps_sched[-1] if eps_sched else 'NA'}\n")
        # if eval_points:
        #     f.write(f"Final periodic exploitability: max={eval_max_eps[-1]:.6e}, mean={eval_mean_eps[-1]:.6e}\n")
        f.write(f"Saved Q tables: {qpath}\n")
        f.write(f"Nash exploitability report: {report_path}\n")
        f.write(f"Policy drift CSV: {drift_csv}\n")


if __name__ == "__main__":
    main()
