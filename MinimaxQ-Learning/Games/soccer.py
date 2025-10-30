# Games/Soccer.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# Actions: 0:Left, 1:Up, 2:Right, 3:Down, 4:Stand
MOVE = np.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]], dtype=int)
ACTION_NAMES = ["L", "U", "R", "D", "S"]

@dataclass
class SoccerConfig:
    h: int = 4
    w: int = 5
    pA: tuple[int, int] = (3, 2)   # (x,y)
    pB: tuple[int, int] = (1, 1)
    goal_y_span: tuple[int, int] = (1, 2)  # inclusive [y1, y2] on both left/right walls
    start_owner: int = 0           # 0 = A, 1 = B
    draw_prob: float = 0.0
    seed: int | None = None

class Soccer:
    """
    Littman-style Markov Soccer:
      - Grid is width w (x=0..w-1), height h (y=0..h-1).
      - Goals are *outside* left wall (x=-1) and right wall (x=w), active for y in [y1,y2].
      - Two-step turn with random execution order each time step.
      - If the first mover steps onto the opponent, possession flips.
      - Reward is zero-sum: +1 for A if A scores, +1 for B if B scores (so rA=+1, rB=-1 for A’s goal).
    """

    def __init__(self,
                 h: int = 4, w: int = 5,
                 pA=(3, 2), pB=(1, 1),
                 goalPositions=(1, 2),
                 ballOwner: int = 0,
                 drawProbability: float = 0.0,
                 seed: int | None = None):
        # Accept legacy ctor params but store in a config
        self.cfg = SoccerConfig(
            h=h, w=w, pA=tuple(pA), pB=tuple(pB),
            goal_y_span=tuple(goalPositions),
            start_owner=ballOwner, draw_prob=drawProbability, seed=seed
        )
        self.rng = np.random.default_rng(seed)
        self.h, self.w = self.cfg.h, self.cfg.w
        self.goal_y_span = np.array(self.cfg.goal_y_span, dtype=int)

        self.positions = np.array([self.cfg.pA, self.cfg.pB], dtype=int)  # [[xA,yA],[xB,yB]]
        self.init_positions = self.positions.copy()
        self.ballOwner = int(self.cfg.start_owner)
        self._term = False

        # action space sizes (used by learners)
        self.n_actions = 5
        self.n_states = self._compute_state_space_size()

    # ---------- Public, Gym-like API ----------
    def reset(self, *, pA=None, pB=None, ballOwner: int | None = None, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if pA is not None:
            self.init_positions[0] = np.array(pA, dtype=int)
        if pB is not None:
            self.init_positions[1] = np.array(pB, dtype=int)
        self.positions = self.init_positions.copy()
        self.ballOwner = self._choose_player() if ballOwner is None else int(ballOwner)
        self._term = False
        return self._encode_state()

    def step(self, joint_action: tuple[int, int]):
        """
        joint_action: (aA, aB) with each in {0..4}
        Returns: next_state_id, (rA, rB), done, info
        """
        if self._term:
            # Standard Gym would raise, but keep permissive
            return self._encode_state(), (0.0, 0.0), True, {}

        if self.rng.random() < self.cfg.draw_prob:
            # -2 was a legacy 'draw' sentinel; here we just end with zero reward
            self._term = True
            return self._encode_state(), (0.0, 0.0), True, {"draw": True}

        order_first = self._choose_player()  # 0 or 1
        a = [int(joint_action[0]), int(joint_action[1])]

        m1 = self._move(order_first, a[order_first])
        if m1 is not None:  # terminal (goal)
            self._term = True
            return self._encode_state(), m1, True, {"first": order_first}

        m2 = self._move(1 - order_first, a[1 - order_first])
        if m2 is not None:  # terminal (goal)
            self._term = True
            return self._encode_state(), m2, True, {"first": order_first}

        return self._encode_state(), (0.0, 0.0), False, {"first": order_first}

    # ---------- Legacy API (backwards-compatible) ----------
    def restart(self, pA=None, pB=None, ballOwner: int | None = None):
        self.reset(pA=pA, pB=pB, ballOwner=ballOwner)

    def play(self, actionA: int, actionB: int):
        """
        Legacy single-step:
          returns:
            1  -> B scores (legacy code used 1 - isInGoal)
            0  -> A scores
           -1  -> non-terminal
           -2  -> draw (stochastic)
        """
        s2, (rA, rB), done, info = self.step((actionA, actionB))
        if info.get("draw", False):
            return -2
        if not done:
            return -1
        # Map terminal reward back to historical sentinel:
        # A scores => rA=+1 -> return 0 ; B scores => rB=+1 -> return 1
        if rA > 0:
            return 0
        if rB > 0:
            return 1
        return -1  # shouldn't happen

    # ---------- Helpers ----------
    def _move(self, player: int, action: int):
        opponent = 1 - player
        new_pos = self.positions[player] + MOVE[action]

        # Collision with opponent -> possession flips; no move
        if np.array_equal(new_pos, self.positions[opponent]):
            self.ballOwner = opponent
            return None

        # Check goal (only if ball owner is the mover and step crosses boundary x=-1 or x=w)
        if self.ballOwner == player:
            goal_side = self._is_in_goal(new_pos[0], new_pos[1])
            if goal_side >= 0:
                # goal_side == 1 means left goal (A’s side), == 0 means right goal (B’s side)
                # Convention: +1 reward to the team that scored:
                if goal_side == 1:     # x == -1 (left of board) -> A scores
                    return (+1.0, -1.0)
                else:                  # x == w (right of board) -> B scores
                    return (-1.0, +1.0)

        # Normal board move
        if self._in_board(new_pos[0], new_pos[1]):
            self.positions[player] = new_pos

        return None

    def _in_board(self, x: int, y: int) -> bool:
        return (0 <= x < self.w) and (0 <= y < self.h)

    def _is_in_goal(self, x: int, y: int) -> int:
        y1, y2 = self.goal_y_span
        if y1 <= y <= y2:
            if x == -1:
                return 1  # left goal (A scores)
            if x == self.w:
                return 0  # right goal (B scores)
        return -1

    def _choose_player(self) -> int:
        return int(self.rng.integers(0, 2))

    # ---------- State encoding (for Q tables & diagnostics) ----------
    def _compute_state_space_size(self) -> int:
        # positions for A and B: w*h each; owner 2
        return (self.w * self.h) * (self.w * self.h) * 2

    def _encode_state(self) -> int:
        xA, yA = self.positions[0]
        xB, yB = self.positions[1]
        o = self.ballOwner
        return (((yA * self.w + xA) * (self.w * self.h) + (yB * self.w + xB)) * 2 + o)

    def decode_state(self, sid: int):
        o = sid % 2; sid //= 2
        flatB = sid % (self.w * self.h); sid //= (self.w * self.h)
        flatA = sid
        xA, yA = flatA % self.w, flatA // self.w
        xB, yB = flatB % self.w, flatB // self.w
        return np.array([[xA, yA], [xB, yB]]), int(o)

    # ---------- Pretty print ----------
    def draw(self, positions=None, ballOwner=None):
        positions = self.positions if positions is None else np.array(positions, dtype=int)
        ballOwner = self.ballOwner if ballOwner is None else int(ballOwner)

        out = []
        for y in range(self.h - 1, -1, -1):
            row = []
            for x in range(self.w):
                if np.array_equal([x, y], positions[0]):
                    row.append('A' if ballOwner == 0 else 'a')
                elif np.array_equal([x, y], positions[1]):
                    row.append('B' if ballOwner == 1 else 'b')
                else:
                    row.append('-')
            out.append(''.join(row))
        print('\n'.join(out))

# Quick smoke test
if __name__ == "__main__":
    env = Soccer()
    s = env.reset()
    env.draw()
    for _ in range(5):
        s, r, d, _ = env.step((env.rng.integers(0,5), env.rng.integers(0,5)))
        print("r=", r, "done=", d)
        env.draw()
        if d: break
