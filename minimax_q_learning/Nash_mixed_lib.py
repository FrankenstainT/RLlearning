import numpy as np
from scipy.optimize import linprog


def solve_both_policies_one_lp(M):
    """
    Solve both players' optimal mixed strategies from a single LP:
        maximize v
        s.t.   v <= x^T M[:,j]   for all j
               M[i,:] y <= v      for all i
               1^T x = 1, x >= 0
               1^T y = 1, y >= 0
    Uses SciPy HiGHS via linprog by minimizing -v.
    Returns x, y, v.
    """
    M = np.asarray(M, dtype=float)
    m, n = M.shape

    # Decision vector z = [x(0..m-1), y(0..n-1), v]
    num_vars = m + n + 1
    x_slice = slice(0, m)
    y_slice = slice(m, m + n)
    v_idx = m + n

    # Objective: minimize -v  (equivalently, maximize v)
    c = np.zeros(num_vars)
    c[v_idx] = -1.0

    # Inequalities (A_ub @ z <= b_ub):
    A_ub = []
    b_ub = []

    # 1) v <= x^T M[:,j]  ->  -M[:,j]^T x + 1*v <= 0
    for j in range(n):
        row = np.zeros(num_vars)
        row[x_slice] = -M[:, j]
        row[v_idx] = 1.0
        A_ub.append(row);
        b_ub.append(0.0)

    # 2) M[i,:] y <= v     ->  M[i,:] y - 1*v <= 0
    for i in range(m):
        row = np.zeros(num_vars)
        row[y_slice] = M[i, :]
        row[v_idx] = -1.0
        A_ub.append(row);
        b_ub.append(0.0)

    A_ub = np.vstack(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None

    # Equalities: 1^T x = 1,  1^T y = 1
    A_eq = np.zeros((2, num_vars))
    b_eq = np.array([1.0, 1.0])
    A_eq[0, x_slice] = 1.0
    A_eq[1, y_slice] = 1.0

    # Bounds: x >= 0, y >= 0, v free
    bounds = [(0, None)] * m + [(0, None)] * n + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")

    z = res.x
    x = z[x_slice].clip(min=0)
    y = z[y_slice].clip(min=0)
    v = z[v_idx]

    # Normalize (robust to tiny numerical drift)
    sx, sy = x.sum(), y.sum()
    if sx > 0: x /= sx
    if sy > 0: y /= sy

    return x, y, v


# ================== TEST SUITE FOR ONE-LP SOLVER ==================
from math import isclose

TOL = 1e-6


def saddle_check(M, x, y, v, name="", verbose=False):
    """
    Verify saddle-point conditions:
      min_j x^T M[:,j] ≈ v
      max_i (M[i,:] y) ≈ v
    Also ensure x, y are valid distributions.
    """
    M = np.asarray(M, dtype=float)
    m, n = M.shape
    assert np.all(x >= -1e-12), f"{name}: x has negative entries"
    assert np.all(y >= -1e-12), f"{name}: y has negative entries"
    assert isclose(np.sum(x), 1.0, rel_tol=1e-9, abs_tol=1e-9), f"{name}: x does not sum to 1"
    assert isclose(np.sum(y), 1.0, rel_tol=1e-9, abs_tol=1e-9), f"{name}: y does not sum to 1"

    col_vals = M.T @ x  # E_x[M[:,j]] for each column j
    row_vals = M @ y  # E_y[M[i,:]] for each row i
    min_col = float(np.min(col_vals)) if col_vals.size else 0.0
    max_row = float(np.max(row_vals)) if row_vals.size else 0.0

    if verbose:
        print(f"\n[{name}]")
        print("M:\n", np.round(M, 3))
        print("x:", np.round(x, 6))
        print("y:", np.round(y, 6))
        print("v:", round(v, 6))
        print("E_x[M[:,j]] by column:", np.round(col_vals, 6))
        print("E_y[M[i,:]] by row   :", np.round(row_vals, 6))
        print("min_j E_x[...] =", round(min_col, 6), "  v =", round(v, 6))
        print("max_i E_y[...] =", round(max_row, 6), "  v =", round(v, 6))

    assert abs(min_col - v) <= 5 * TOL, f"{name}: min_j E_x[...]={min_col} not ≈ v={v}"
    assert abs(max_row - v) <= 5 * TOL, f"{name}: max_i E_y[...]={max_row} not ≈ v={v}"
    return True


def build_cases():
    cases = []

    # 1) Zero 2x2 (degenerate): any strategy is optimal; v=0
    cases.append(("Zero_2x2", np.array([[0, 0], [0, 0]], float)))

    # 2) Constant matrix (degenerate): any x,y; v=that constant
    cases.append(("Constant_3x3", np.full((3, 3), 2.5)))

    # 3) Matching Pennies (value 0, uniform (0.5,0.5))
    cases.append(("MatchingPennies", np.array([[1, -1],
                                               [-1, 1]], float)))

    # 4) Rock–Paper–Scissors (value 0, uniform (1/3,1/3,1/3))
    cases.append(("RPS", np.array([[0, -1, 1],
                                   [1, 0, -1],
                                   [-1, 1, 0]], float)))

    # 5) Row domination (row 0 strictly dominates row 1)
    cases.append(("RowDominates", np.array([[2, 2],
                                            [1, 1]], float)))

    # 6) Column domination (column 0 dominates column 1 for the column player)
    cases.append(("ColDominates", np.array([[0, 1],
                                            [0, 1]], float)))

    # 7) Rectangular 3x2
    cases.append(("Rectangular_3x2", np.array([[1, 0],
                                               [0, 1],
                                               [0, 0]], float)))

    # 8) Rectangular 2x3
    cases.append(("Rectangular_2x3", np.array([[2, -1, 0.5],
                                               [1, 0, -2]], float)))

    # 9) Rank-1 matrix (outer product) – easy structure
    a = np.array([2.0, -1.0, 3.0])
    b = np.array([-1.0, 4.0])
    cases.append(("Rank1_3x2", np.outer(a, b)))

    # 10) Skew-symmetric (zero-sum symmetric game, value 0)
    K = np.array([[0, 2, -3],
                  [-2, 0, 1],
                  [3, -1, 0]], float)
    cases.append(("SkewSymmetric_3x3", K))

    # 11) Near-degenerate (tiny differences)
    cases.append(("NearDegenerate_4x4", np.array([
        [1.0001, 1.0, 1.0, 1.0],
        [1.0, 1.0002, 1.0, 1.0],
        [1.0, 1.0, 1.0003, 1.0],
        [1.0, 1.0, 1.0, 1.0004]
    ], float)))

    # 12) Integer random (seeded) 4x4
    rng = np.random.default_rng(123)
    cases.append(("RandomInt_4x4", rng.integers(-3, 6, size=(4, 4)).astype(float)))

    # 13) Real random (seeded) 5x3
    rng = np.random.default_rng(456)
    cases.append(("RandomReal_5x3", rng.uniform(-5, 10, size=(5, 3))))

    # 14) Duplicate rows/cols (ties & degeneracy)
    cases.append(("DuplicateRowsCols_4x4", np.array([
        [2, -1, 0, 1],
        [2, -1, 0, 1],  # duplicate row
        [0, 0, 0, 0],
        [3, 3, -2, -2]
    ], float)))

    # 15) Adversarial spread (large-magnitude spread)
    cases.append(("WideSpread_3x3", np.array([
        [1000, -999, -500],
        [-800, 50, 400],
        [-200, 100, -300]
    ], float)))

    return cases


def _saddle_ok(M, x, y, v, tol=1e-7):
    M = np.asarray(M, float)
    col_vals = M.T @ x  # E_x[M[:,j]]
    row_vals = M @ y  # E_y[M[i,:]]
    return (abs(col_vals.min() - v) <= 5 * tol) and (abs(row_vals.max() - v) <= 5 * tol)


def _kkt_tightness_ok(M, x, y, v, tol=1e-7):
    """Any action played with positive prob should be tight at value v."""
    M = np.asarray(M, float)
    col_vals = M.T @ x  # columns evaluated under x
    row_vals = M @ y  # rows evaluated under y
    # If x_i > 0, then row i must be tight: M[i,:] y == v
    tight_rows_ok = True
    for i, xi in enumerate(x):
        if xi > 1e-10 and abs(row_vals[i] - v) > 5 * tol:
            tight_rows_ok = False;
            break
    # If y_j > 0, then column j must be tight: x^T M[:,j] == v
    tight_cols_ok = True
    for j, yj in enumerate(y):
        if yj > 1e-10 and abs(col_vals[j] - v) > 5 * tol:
            tight_cols_ok = False;
            break
    return tight_rows_ok and tight_cols_ok


def test_shift_and_scale_invariance(M):
    # Base solution
    x1, y1, v1 = solve_both_policies_one_lp(M)
    x1 = np.maximum(x1, 0);
    x1 /= x1.sum() if x1.sum() > 0 else 1
    y1 = np.maximum(y1, 0);
    y1 /= y1.sum() if y1.sum() > 0 else 1
    assert _saddle_ok(M, x1, y1, v1), "Base saddle check failed"
    assert _kkt_tightness_ok(M, x1, y1, v1), "Base KKT tightness failed"

    # Shift invariance (policies may differ; value shifts by c)
    c = 3.7
    x2, y2, v2 = solve_both_policies_one_lp(M + c)
    x2 = np.maximum(x2, 0);
    x2 /= x2.sum() if x2.sum() > 0 else 1
    y2 = np.maximum(y2, 0);
    y2 /= y2.sum() if y2.sum() > 0 else 1
    assert _saddle_ok(M + c, x2, y2, v2), "Shifted saddle check failed"
    assert _kkt_tightness_ok(M + c, x2, y2, v2), "Shifted KKT tightness failed"
    # Value shift only
    assert abs(v2 - (v1 + c)) <= 1e-6, f"v shift failed: v2={v2}, expected {v1 + c}"

    # Positive scaling invariance (policies may differ; value scales by k)
    k = 2.5
    x3, y3, v3 = solve_both_policies_one_lp(M * k)
    x3 = np.maximum(x3, 0);
    x3 /= x3.sum() if x3.sum() > 0 else 1
    y3 = np.maximum(y3, 0);
    y3 /= y3.sum() if y3.sum() > 0 else 1
    assert _saddle_ok(M * k, x3, y3, v3), "Scaled saddle check failed"
    assert _kkt_tightness_ok(M * k, x3, y3, v3), "Scaled KKT tightness failed"
    # Value scale only
    assert abs(v3 - k * v1) <= 1e-6, f"v scale failed: v3={v3}, expected {k * v1}"


def run_all_tests(verbose=False):
    cases = build_cases()
    for name, M in cases:
        x, y, v = solve_both_policies_one_lp(M)
        # robust renorm (tiny numerical drift)
        x = np.maximum(x, 0);
        sx = x.sum();
        x = x / sx if sx > 0 else np.full_like(x, 1 / len(x))
        y = np.maximum(y, 0);
        sy = y.sum();
        y = y / sy if sy > 0 else np.full_like(y, 1 / len(y))

        saddle_check(M, x, y, v, name=name, verbose=verbose)
        print(f"✓ {name} passed.")

    # Extra invariance tests on a few cases
    print("\nRunning invariance checks...")
    for key in ["MatchingPennies", "RPS", "RandomInt_4x4"]:
        M = dict(cases)[key]
        test_shift_and_scale_invariance(M)
        print(f"✓ Invariance checks passed for {key}.")

    print("\nAll tests passed.\n")


# --------- quick demo & verification ----------
if __name__ == "__main__":
    np.random.seed(7)
    M = np.random.uniform(-5, 10, size=(4, 4))  # 4x4 example
    # M = np.zeros((4,4))
    x, y, v = solve_both_policies_one_lp(M)
    print("M:\n", np.round(M, 2))
    print("x:", np.round(x, 6))
    print("y:", np.round(y, 6))
    print("v*:", round(v, 6))

    # Saddle checks
    col_vals = M.T @ x  # E_x[M[:,j]] for each column j
    row_vals = M @ y  # E_y[M[i,:]] for each row i
    print("min_j E_x[M[:,j]] =", round(col_vals.min(), 6))
    print("max_i E_y[M[i,:]] =", round(row_vals.max(), 6))
    # run_all_tests(verbose=True)
