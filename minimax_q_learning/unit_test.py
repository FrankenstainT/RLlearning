# pip install numpy scipy
import numpy as np
from scipy.optimize import linprog

# ============================================================
#                    Linear-programming helpers
# ============================================================
def solve_row_minimax(M):
    """
    Row-only LP (classic): max_x v  s.t.  M^T x >= v * 1,  1^T x = 1,  x >= 0
    Returns (v, x).
    """
    n, m = M.shape
    c = np.zeros(n + 1); c[-1] = -1.0  # minimize -v
    G = np.hstack([-M.T, np.ones((m, 1))])  # -M^T x + v*1 <= 0
    h = np.zeros(m)
    Aeq = np.zeros((1, n + 1)); Aeq[0, :n] = 1.0
    beq = np.array([1.0])
    bounds = [(0.0, 1.0)] * n + [(M.min(), M.max())]
    res = linprog(c, A_ub=G, b_ub=h, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("LP failed: " + res.message)
    x = res.x[:n]; v = res.x[-1]
    return float(v), (x / max(x.sum(), 1e-12))

def solve_zero_sum_both_policies(M):
    """
    Solve a single LP that yields BOTH players' mixed strategies for a zero-sum matrix game
    with row payoff M. Variables are [x (n), y (m), v].

    Maximize v
    s.t.
        M^T x >= v * 1_m        (row feasibility against all columns)
        M y   <= v * 1_n        (column feasibility against all rows)
        1^T x = 1,  1^T y = 1
        x >= 0,   y >= 0
    """
    n, m = M.shape
    num_vars = n + m + 1  # x (n), y (m), v (1)
    # Minimize -v  <=> maximize v
    c = np.zeros(num_vars); c[-1] = -1.0

    A_ub = []
    b_ub = []

    # -M^T x + v*1 <= 0   (m constraints)
    G1 = np.zeros((m, num_vars))
    G1[:, :n] = -M.T
    G1[:, -1] = 1.0
    A_ub.append(G1); b_ub.append(np.zeros(m))

    #  M y - v*1 <= 0     (n constraints)
    G2 = np.zeros((n, num_vars))
    G2[:, n:n+m] = M
    G2[:, -1] = -1.0
    A_ub.append(G2); b_ub.append(np.zeros(n))

    A_ub = np.vstack(A_ub); b_ub = np.concatenate(b_ub)

    # Equalities: sum x = 1; sum y = 1
    Aeq = np.zeros((2, num_vars))
    Aeq[0, :n] = 1.0
    Aeq[1, n:n+m] = 1.0
    beq = np.array([1.0, 1.0])

    # Bounds
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * m + [(M.min(), M.max())]  # v bounded

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("Joint LP failed: " + res.message)

    x = res.x[:n]
    y = res.x[n:n+m]
    v = res.x[-1]
    # Normalize defensively
    x = x / max(x.sum(), 1e-12)
    y = y / max(y.sum(), 1e-12)
    return float(v), x, y

# ============================================================
#                       Diagnostics
# ============================================================
def kl(p, q, eps=1e-12):
    p = np.clip(p, eps, 1); q = np.clip(q, eps, 1)
    return float((p * (np.log(p) - np.log(q))).sum())

def value(p, A, q):
    return float(p @ A @ q)

def best_response_row(A, q):
    vals = A @ q
    i = int(np.argmax(vals))
    return float(vals[i]), i

def best_response_col(A, p):
    vals = A.T @ p
    j = int(np.argmin(vals))
    return float(vals[j]), j

def support_equal_payoff_checks(A, p, q, v, tol=5e-3, thresh=1e-6):
    row_payoffs = A @ q
    col_payoffs = A.T @ p
    supp_row = np.where(p > thresh)[0]
    supp_col = np.where(q > thresh)[0]

    row_support_max = float(np.max(np.abs(row_payoffs[supp_row] - v))) if len(supp_row) else float('nan')
    row_off_max_excess = float(np.max(row_payoffs - v)) if len(row_payoffs) else 0.0

    col_support_max = float(np.max(np.abs(col_payoffs[supp_col] - v))) if len(supp_col) else float('nan')
    col_off_min_deficit = float(v - np.min(col_payoffs)) if len(col_payoffs) else 0.0

    return {
        "row_support_close": (np.isnan(row_support_max) or row_support_max <= tol),
        "row_offsupport_ok": (row_off_max_excess <= tol),
        "row_support_max_abs_err": row_support_max,
        "row_offsupport_max_excess": max(0.0, row_off_max_excess),
        "col_support_close": (np.isnan(col_support_max) or col_support_max <= tol),
        "col_offsupport_ok": (col_off_min_deficit <= tol),
        "col_support_max_abs_err": col_support_max,
        "col_offsupport_max_deficit": max(0.0, col_off_min_deficit),
    }

def symmetry_check(p, q, tol=5e-3):
    return float(np.max(np.abs(p - q))) <= tol

# ============================================================
#                 Minimax Q-learning (one-state)
# ============================================================
class MinimaxQMatrix:
    def __init__(self, A, alpha=0.1, gamma=0.95, eps=0.2, seed=0):
        self.A = A.astype(float)
        self.nA, self.nB = A.shape
        self.Q = np.zeros_like(A, dtype=float)
        self.alpha, self.gamma, self.eps = alpha, gamma, eps
        self.rng = np.random.default_rng(seed)

    def row_policy(self):
        return solve_row_minimax(self.Q)[1]

    def col_policy(self):
        return solve_row_minimax((-self.Q).T)[1]  # from dual

    def estimate_values_on_Q(self):
        v_row = solve_row_minimax(self.Q)[0]
        v_col = -solve_row_minimax((-self.Q).T)[0]
        v_bar = 0.5 * (v_row + v_col)
        return float(v_row), float(v_col), float(v_bar)

    def step(self):
        p = self.row_policy()
        q = self.col_policy()
        ai = self.rng.choice(self.nA, p=(1 - self.eps) * p + self.eps / self.nA)
        aj = self.rng.choice(self.nB, p=(1 - self.eps) * q + self.eps / self.nB)
        r = self.A[ai, aj]
        v_row, v_col, v_bar = self.estimate_values_on_Q()
        td = r + self.gamma * v_bar - self.Q[ai, aj]
        self.Q[ai, aj] += self.alpha * td

# ============================================================
#                  Nash Q-learning (one-state)
# ============================================================
class NashQMatrix:
    """
    Zero-sum Nash-Q that solves BOTH policies from one LP on Q each step.
    """
    def __init__(self, A, alpha=0.1, gamma=0.95, eps=0.2, seed=42):
        self.A = A.astype(float)
        self.nA, self.nB = A.shape
        self.Q = np.zeros_like(A, dtype=float)
        self.alpha, self.gamma, self.eps = alpha, gamma, eps
        self.rng = np.random.default_rng(seed)

    def stage_equilibrium_on_Q(self):
        return solve_zero_sum_both_policies(self.Q)  # (v, p, q)

    def row_policy(self):
        return self.stage_equilibrium_on_Q()[1]

    def col_policy(self):
        return self.stage_equilibrium_on_Q()[2]

    def estimate_values_on_Q(self):
        # From joint LP and individual sides for consistency diagnostics
        v_joint, _, _ = self.stage_equilibrium_on_Q()
        v_row = solve_row_minimax(self.Q)[0]
        v_col = -solve_row_minimax((-self.Q).T)[0]
        return float(v_row), float(v_col), float(v_joint)

    def step(self):
        v, p, q = self.stage_equilibrium_on_Q()
        ai = self.rng.choice(self.nA, p=(1 - self.eps) * p + self.eps / self.nA)
        aj = self.rng.choice(self.nB, p=(1 - self.eps) * q + self.eps / self.nB)
        r = self.A[ai, aj]
        v_row, v_col, v_joint = self.estimate_values_on_Q()
        td = r + self.gamma * v_joint - self.Q[ai, aj]
        self.Q[ai, aj] += self.alpha * td

# ============================================================
#                       Test harness
# ============================================================
def run_unit_test(
    name, A, algo_cls,
    true_NE_row, true_NE_col=None,
    steps=20000, print_every=2000,
    tol_consistency=5e-3, tol_support=5e-3, symmetric=False, seed=0
):
    print(f"\n=== {name} :: {algo_cls.__name__} ===")
    agent = algo_cls(A, alpha=0.1, gamma=0.95, eps=0.2, seed=seed)
    avg_return = 0.0

    for t in range(1, steps + 1):
        p = agent.row_policy()
        q = agent.col_policy()
        avg_return = 0.99 * avg_return + 0.01 * value(p, A, q)

        agent.step()

        if t % print_every == 0 or t == steps:
            v_row_Q, v_col_Q, v_bar_Q = agent.estimate_values_on_Q()
            consistency_gap = abs(v_row_Q - v_col_Q)

            v_now = value(p, A, q)
            br_row_val, br_i = best_response_row(A, q)
            br_col_val, br_j = best_response_col(A, p)
            row_adv = max(0.0, br_row_val - v_now)
            col_adv = max(0.0, v_now - br_col_val)
            total_expl = row_adv + col_adv

            supp = support_equal_payoff_checks(A, p, q, v_now, tol=tol_support)
            kl_row = kl(p, true_NE_row) if true_NE_row is not None else np.nan
            kl_col = kl(q, true_NE_col) if true_NE_col is not None else np.nan

            print(f"t={t:5d}  V_row(Q)={v_row_Q:+.4f}  V_col(Q)={v_col_Q:+.4f}  "
                  f"V̄/Joint={v_bar_Q:+.4f}  |gap|={consistency_gap:.3e}  avg_ret≈{avg_return:+.4f}")
            print(f"        KL(row||row*)={kl_row:.3e}   KL(col||col*)={kl_col:.3e}")
            print(f"        exploitability: row_adv={row_adv:.3e}  col_adv={col_adv:.3e}  total={total_expl:.3e}  "
                  f"(BR_i={br_i}, BR_j={br_j})")
            print(f"        support: row_close={supp['row_support_close']} "
                  f"(max|err|={supp['row_support_max_abs_err']:.3e}), "
                  f"row_off_ok={supp['row_offsupport_ok']} "
                  f"(excess={supp['row_offsupport_max_excess']:.3e}); "
                  f"col_close={supp['col_support_close']} "
                  f"(max|err|={supp['col_support_max_abs_err']:.3e}), "
                  f"col_off_ok={supp['col_offsupport_ok']} "
                  f"(deficit={supp['col_offsupport_max_deficit']:.3e})")
            if symmetric:
                print(f"        symmetry_ok={symmetry_check(p, q)}")

    # Final snapshot
    p = agent.row_policy(); q = agent.col_policy()
    v_row_Q, v_col_Q, v_bar_Q = agent.estimate_values_on_Q()
    v_now = value(p, A, q)
    br_row_val, _ = best_response_row(A, q)
    br_col_val, _ = best_response_col(A, p)

    print("\nFinal summary:")
    print(f"  row_policy={np.round(p,4)}  col_policy={np.round(q,4)}")
    if true_NE_row is not None:
        print(f"  KL(row||row*)={kl(p, true_NE_row):.3e}")
    if true_NE_col is not None:
        print(f"  KL(col||col*)={kl(q, true_NE_col):.3e}")
    print(f"  saddle consistency |V_row(Q)-V_col(Q)| = {abs(v_row_Q - v_col_Q):.3e} (Joint/¯V={v_bar_Q:+.4f})")
    print(f"  exploitability: row_adv={max(0.0, br_row_val - v_now):.3e}, "
          f"col_adv={max(0.0, v_now - br_col_val):.3e}, total={max(0.0, br_row_val - v_now)+max(0.0, v_now - br_col_val):.3e}")

# ============================================================
#                        Demo / Unit tests
# ============================================================
if __name__ == "__main__":
    # Matching Pennies — NE = (0.5, 0.5), value 0
    A_mp = np.array([[+1, -1],
                     [-1, +1]], dtype=float)

    # Rock–Paper–Scissors — NE = (1/3,1/3,1/3), value 0
    A_rps = np.array([[ 0,-1,+1],
                      [+1, 0,-1],
                      [-1,+1, 0]], dtype=float)

    # ---------- Minimax-Q ----------
    run_unit_test(
        "Matching Pennies", A=A_mp, algo_cls=MinimaxQMatrix,
        true_NE_row=np.array([0.5, 0.5]), true_NE_col=np.array([0.5, 0.5]),
        steps=20000, print_every=2000, symmetric=True, seed=0
    )
    run_unit_test(
        "Rock–Paper–Scissors", A=A_rps, algo_cls=MinimaxQMatrix,
        true_NE_row=np.array([1/3,1/3,1/3]), true_NE_col=np.array([1/3,1/3,1/3]),
        steps=25000, print_every=2500, symmetric=True, seed=1
    )

    # ---------- Nash-Q (joint LP for both policies) ----------
    run_unit_test(
        "Matching Pennies", A=A_mp, algo_cls=NashQMatrix,
        true_NE_row=np.array([0.5, 0.5]), true_NE_col=np.array([0.5, 0.5]),
        steps=20000, print_every=2000, symmetric=True, seed=2
    )
    run_unit_test(
        "Rock–Paper–Scissors", A=A_rps, algo_cls=NashQMatrix,
        true_NE_row=np.array([1/3,1/3,1/3]), true_NE_col=np.array([1/3,1/3,1/3]),
        steps=25000, print_every=2500, symmetric=True, seed=3
    )
