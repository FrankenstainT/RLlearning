# pip install numpy scipy
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(0)

# ---------- Utilities ----------
def solve_row_minimax(M):
    """
    Solve: max_x v  s.t.  M^T x >= v * 1,  1^T x = 1,  x >= 0
    Returns (v, x).
    """
    n, m = M.shape
    c = np.zeros(n + 1); c[-1] = -1.0  # minimize -v
    G = np.hstack([-M.T, np.ones((m, 1))])  # -M^T x + v*1 <= 0
    h = np.zeros(m)
    Aeq = np.zeros((1, n + 1)); Aeq[0, :n] = 1.0
    beq = np.array([1.0])
    bounds = [(0.0, 1.0)] * n + [(M.min(), M.max())]  # safe v bounds
    res = linprog(c, A_ub=G, b_ub=h, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("LP failed: " + res.message)
    x = res.x[:n]; v = res.x[-1]
    return v, x / x.sum()

def kl(p, q, eps=1e-12):
    p = np.clip(p, eps, 1); q = np.clip(q, eps, 1)
    return float((p * (np.log(p) - np.log(q))).sum())

def value(p, A, q):
    return float(p @ A @ q)

def best_response_row(A, q):
    # returns (val, argmax)
    vals = A @ q
    i = int(np.argmax(vals))
    return float(vals[i]), i

def best_response_col(A, p):
    # column minimizes row payoff
    vals = A.T @ p
    j = int(np.argmin(vals))
    return float(vals[j]), j

def support_equal_payoff_checks(A, p, q, v, tol=5e-3, thresh=1e-6):
    """
    For zero-sum saddle point (p*, q*, v):
      - Row side: (A q)_i <= v, equality for i in supp(p*)
      - Col side: (A^T p)_j >= v, equality for j in supp(q*)
    Returns dict with booleans and diagnostic gaps.
    """
    row_payoffs = A @ q
    col_payoffs = (A.T @ p)

    supp_row = np.where(p > thresh)[0]
    supp_col = np.where(q > thresh)[0]

    # Row: max gap on support (should be near 0)
    if len(supp_row) > 0:
        row_support_max = float(np.max(np.abs(row_payoffs[supp_row] - v)))
    else:
        row_support_max = float('nan')

    # Row: off-support should be <= v + tol
    row_off_max_excess = float(np.max(row_payoffs - v)) if len(row_payoffs) else 0.0

    # Column: support near v
    if len(supp_col) > 0:
        col_support_max = float(np.max(np.abs(col_payoffs[supp_col] - v)))
    else:
        col_support_max = float('nan')

    # Column: off-support should be >= v - tol (since column minimizes)
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

# ---------- Minimax-Q learner for one-state matrix game ----------
class MinimaxQMatrix:
    def __init__(self, A, alpha=0.1, gamma=0.95, eps=0.2, seed=0):
        self.A = A.astype(float)
        self.nA, self.nB = A.shape
        self.Q = np.zeros_like(A, dtype=float)
        self.alpha, self.gamma, self.eps = alpha, gamma, eps
        self.rng = np.random.default_rng(seed)

    def row_policy(self):
        _, pi = solve_row_minimax(self.Q)
        return pi

    def col_policy(self):
        # min over columns of Q  <=>  max over rows of (-Q)^T
        _, sigma = solve_row_minimax((-self.Q).T)
        return sigma

    def estimate_values_on_Q(self):
        v_row, _ = solve_row_minimax(self.Q)
        v_col = -solve_row_minimax((-self.Q).T)[0]
        return float(v_row), float(v_col)

    def step(self):
        pi = self.row_policy()
        sigma = self.col_policy()

        # exploration
        ai = self.rng.choice(self.nA, p=(1 - self.eps) * pi + self.eps / self.nA)
        aj = self.rng.choice(self.nB, p=(1 - self.eps) * sigma + self.eps / self.nB)
        r = self.A[ai, aj]

        v_next = self.estimate_values_on_Q()[0]  # one-state => V(s') = V(s)
        td = r + self.gamma * v_next - self.Q[ai, aj]
        self.Q[ai, aj] += self.alpha * td

# ---------- Runner with extra checks ----------
def run_unit_test(
    name,
    A,
    true_NE_row,
    true_NE_col=None,         # optional
    steps=20000,
    print_every=2000,
    tol_consistency=5e-3,
    tol_support=5e-3,
    symmetric=False
):
    print(f"\n=== {name} ===")
    agent = MinimaxQMatrix(A, alpha=0.1, gamma=0.95, eps=0.2, seed=0)
    avg_return = 0.0

    for t in range(1, steps + 1):
        # diagnostics (using current policies on TRUE game A)
        p = agent.row_policy()
        q = agent.col_policy()
        avg_return = 0.99 * avg_return + 0.01 * value(p, A, q)

        agent.step()

        if t % print_every == 0 or t == steps:
            # LP values on Q for consistency
            v_row_Q, v_col_Q = agent.estimate_values_on_Q()
            consistency_gap = abs(v_row_Q - v_col_Q)

            # Exploitability (one-step best responses on TRUE A)
            v_now = value(p, A, q)
            br_row_val, br_i = best_response_row(A, q)
            br_col_val, br_j = best_response_col(A, p)
            row_adv = max(0.0, br_row_val - v_now)   # how much row can gain by deviating
            col_adv = max(0.0, v_now - br_col_val)   # how much column can gain by deviating
            total_expl = row_adv + col_adv

            # Support / equal-payoff checks against current v_now
            supp_checks = support_equal_payoff_checks(A, p, q, v_now, tol=tol_support)

            # KLs to target NE (if provided)
            kl_row = kl(p, true_NE_row) if true_NE_row is not None else np.nan
            kl_col = kl(q, true_NE_col) if true_NE_col is not None else np.nan

            # Optional symmetry check
            sym_ok = symmetry_check(p, q) if symmetric else None

            print(f"t={t:5d}  V_row(Q)={v_row_Q:+.4f}  V_col(Q)={v_col_Q:+.4f}  |gap|={consistency_gap:.3e}  "
                  f"avg_ret≈{avg_return:+.4f}")
            print(f"        KL(row||row*)={kl_row:.3e}   KL(col||col*)={kl_col:.3e}")
            print(f"        exploitability: row_adv={row_adv:.3e}  col_adv={col_adv:.3e}  total={total_expl:.3e}  "
                  f"(BR_i={br_i}, BR_j={br_j})")
            print(f"        support: row_close={supp_checks['row_support_close']} "
                  f"(max|err|={supp_checks['row_support_max_abs_err']:.3e}), "
                  f"row_off_ok={supp_checks['row_offsupport_ok']} "
                  f"(excess={supp_checks['row_offsupport_max_excess']:.3e}); "
                  f"col_close={supp_checks['col_support_close']} "
                  f"(max|err|={supp_checks['col_support_max_abs_err']:.3e}), "
                  f"col_off_ok={supp_checks['col_offsupport_ok']} "
                  f"(deficit={supp_checks['col_offsupport_max_deficit']:.3e})")
            if sym_ok is not None:
                print(f"        symmetry_ok={sym_ok}")

    # Final concise summary
    p = agent.row_policy(); q = agent.col_policy()
    v_row_Q, v_col_Q = agent.estimate_values_on_Q()
    print("\nFinal summary:")
    print(f"  row_policy={np.round(p,4)}")
    if true_NE_row is not None:
        print(f"  KL(row||row*)={kl(p, true_NE_row):.3e}")
    if true_NE_col is not None:
        print(f"  col_policy={np.round(q,4)}  KL(col||col*)={kl(q, true_NE_col):.3e}")
    print(f"  saddle_consistency |V_row(Q)-V_col(Q)| = {abs(v_row_Q - v_col_Q):.3e}")
    v_now = value(p, A, q)
    br_row_val, _ = best_response_row(A, q)
    br_col_val, _ = best_response_col(A, p)
    print(f"  exploitability: row_adv={max(0.0, br_row_val - v_now):.3e}, "
          f"col_adv={max(0.0, v_now - br_col_val):.3e}, total={max(0.0, br_row_val - v_now)+max(0.0, v_now - br_col_val):.3e}")

# ---------- Examples ----------
if __name__ == "__main__":
    # Matching Pennies (zero-sum, symmetric), NE = (0.5, 0.5), value 0
    A_mp = np.array([[+1, -1],
                     [-1, +1]], dtype=float)
    run_unit_test(
        "Matching Pennies",
        A=A_mp,
        true_NE_row=np.array([0.5, 0.5]),
        true_NE_col=np.array([0.5, 0.5]),
        steps=20000,
        print_every=2000,
        symmetric=True
    )

    # Rock–Paper–Scissors, NE = (1/3,1/3,1/3), value 0
    A_rps = np.array([[ 0,-1,+1],
                      [+1, 0,-1],
                      [-1,+1, 0]], dtype=float)
    run_unit_test(
        "Rock–Paper–Scissors",
        A=A_rps,
        true_NE_row=np.array([1/3,1/3,1/3]),
        true_NE_col=np.array([1/3,1/3,1/3]),
        steps=25000,
        print_every=2500,
        symmetric=True
    )
