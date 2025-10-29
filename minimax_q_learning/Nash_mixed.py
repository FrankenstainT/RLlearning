import random
from math import isclose

EPS = 1e-9


def simplex(c, A, b):
    m = len(A)
    n = len(c)

    tableau = [[0.0] * (n + m + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            tableau[i][j] = A[i][j]
        tableau[i][n + i] = 1.0  # slack variable
        tableau[i][-1] = b[i]
    for j in range(n):
        tableau[m][j] = -c[j]

    basis = [n + i for i in range(m)]

    def pivot(pr, pc):
        pv = tableau[pr][pc]
        for j in range(len(tableau[0])):
            tableau[pr][j] /= pv
        for i in range(m + 1):
            if i == pr:
                continue
            factor = tableau[i][pc]
            for j in range(len(tableau[0])):
                tableau[i][j] -= factor * tableau[pr][j]
        basis[pr] = pc

    while True:
        entering = None
        for j in range(len(tableau[0]) - 1):
            if tableau[m][j] < -EPS:
                entering = j
                break
        if entering is None:
            break

        leaving = None
        min_ratio = float('inf')
        for i in range(m):
            if tableau[i][entering] > EPS:
                ratio = tableau[i][-1] / tableau[i][entering]
                if ratio < min_ratio - EPS:
                    min_ratio = ratio
                    leaving = i
        if leaving is None:
            raise RuntimeError("Unbounded")

        pivot(leaving, entering)

    x = [0.0] * (n + m)
    for i in range(m):
        x[basis[i]] = tableau[i][-1]
    return tableau[m][-1], x[:n]


def solve_both_policies(M):
    """
    Solve both players' optimal mixed strategies in one LP:
        max v
        s.t.   v <= x^T M[:,j]   for all columns j
               M[i,:] y <= v     for all rows i
               1^T x = 1, x >= 0
               1^T y = 1, y >= 0
    Implemented in <= form with v = v_pos - v_neg, using your simplex().
    M: list of lists or 2D array-like, shape (m, n)
    Returns: x (len m), y (len n), v (float)
    """
    # Convert to python lists and sizes
    m = len(M)
    n = len(M[0]) if m > 0 else 0

    # Variables (all >= 0 for simplex):
    # x_0..x_{m-1}, y_0..y_{n-1}, v_pos, v_neg  --> total var_count = m + n + 2
    var_count = m + n + 2
    x_off = 0
    y_off = m
    vpos_idx = m + n
    vneg_idx = m + n + 1

    A = []
    b = []

    # (1) Agent1 safety: v <= sum_i x_i M[i,j]  for each column j
    #  => -sum_i M[i,j] x_i + v_pos - v_neg <= 0
    for j in range(n):
        row = [0.0] * var_count
        for i in range(m):
            row[x_off + i] = -float(M[i][j])
        row[vpos_idx] = 1.0
        row[vneg_idx] = -1.0
        A.append(row);
        b.append(0.0)

    # (2) Agent2 cap: sum_j M[i,j] y_j <= v  for each row i
    #  => sum_j M[i,j] y_j - v_pos + v_neg <= 0
    for i in range(m):
        row = [0.0] * var_count
        for j in range(n):
            row[y_off + j] = float(M[i][j])
        row[vpos_idx] = -1.0
        row[vneg_idx] = 1.0
        A.append(row);
        b.append(0.0)

    # (3) 1^T x = 1  -> two inequalities
    row = [0.0] * var_count
    for i in range(m):
        row[x_off + i] = 1.0
    A.append(row.copy());
    b.append(1.0)
    for i in range(m):
        row[x_off + i] = -1.0
    A.append(row.copy());
    b.append(-1.0)

    # (4) 1^T y = 1  -> two inequalities
    row = [0.0] * var_count
    for j in range(n):
        row[y_off + j] = 1.0
    A.append(row.copy());
    b.append(1.0)
    for j in range(n):
        row[y_off + j] = -1.0
    A.append(row.copy());
    b.append(-1.0)

    # Objective: maximize v = v_pos - v_neg
    c = [0.0] * var_count
    c[vpos_idx] = 1.0
    c[vneg_idx] = -1.0

    opt, x_all = simplex(c, A, b)

    # Extract variables
    x = [max(0.0, x_all[x_off + i]) for i in range(m)]
    y = [max(0.0, x_all[y_off + j]) for j in range(n)]
    v = x_all[vpos_idx] - x_all[vneg_idx]

    # Normalize x, y in case of tiny numerical drift or degeneracy
    sx = sum(x)
    sy = sum(y)
    if sx <= EPS:
        x = [1.0 / m] * m
    else:
        x = [xi / sx for xi in x]
    if sy <= EPS:
        y = [1.0 / n] * n
    else:
        y = [yi / sy for yi in y]

    return x, y, v

# ------------------ TEST HARNESS FOR solve_both_policies ------------------

from math import isclose

EPS = 1e-7

def almost_equal(a, b, tol=1e-6):
    return abs(a - b) <= tol

def vec_close(v1, v2, tol=1e-6):
    if len(v1) != len(v2):
        return False
    return all(abs(a - b) <= tol for a, b in zip(v1, v2))

def normalize_or_uniform(v):
    s = sum(max(0.0, x) for x in v)
    if s <= 1e-12:
        return [1.0/len(v)] * len(v)
    return [max(0.0, x)/s for x in v]

def saddle_checks(M, x, y, v, tol=1e-5, verbose=False):
    """Verify saddle conditions:
       min_j E_x[M[:,j]] ≈ v and max_i E_y[M[i,:]] ≈ v.
       Also verify distributions sum to 1 and are nonnegative.
    """
    m = len(M)
    n = len(M[0]) if m else 0

    # sanity on distributions
    assert all(z >= -1e-10 for z in x), "x has negative entries"
    assert all(z >= -1e-10 for z in y), "y has negative entries"
    assert almost_equal(sum(x), 1.0, tol=1e-6), "x does not sum to 1"
    assert almost_equal(sum(y), 1.0, tol=1e-6), "y does not sum to 1"

    # E_x[M[:,j]] for each column j
    col_vals = []
    for j in range(n):
        col_vals.append(sum(x[i] * M[i][j] for i in range(m)))
    # E_y[M[i,:]] for each row i
    row_vals = []
    for i in range(m):
        row_vals.append(sum(M[i][j] * y[j] for j in range(n)))

    min_col = min(col_vals) if col_vals else 0.0
    max_row = max(row_vals) if row_vals else 0.0

    if verbose:
        print("E_x[M[:,j]] by column:", [round(z, 6) for z in col_vals])
        print("E_y[M[i,:]] by row   :", [round(z, 6) for z in row_vals])
        print("min_j E_x[...] =", round(min_col, 6), "  v =", round(v, 6))
        print("max_i E_y[...] =", round(max_row, 6), "  v =", round(v, 6))

    assert abs(min_col - v) <= tol, f"min_j E_x[M[:,j]] = {min_col} not ≈ v={v}"
    assert abs(max_row - v) <= tol, f"max_i E_y[M[i,:]] = {max_row} not ≈ v={v}"
    # Safety: each constraint should hold with <= or >= (within tol)
    for j, val in enumerate(col_vals):
        assert val + 1e-7 >= v - tol, f"x vs col {j} violates v≤E_x[M[:,j]]"
    for i, val in enumerate(row_vals):
        assert val - 1e-7 <= v + tol, f"row {i} vs y violates E_y[M[i,:]]≤v"
    return True

def run_case(name, M, expect_x=None, expect_y=None, expect_v=None,
             check_exact=False, verbose=False):
    print(f"\n=== Case: {name} ===")
    x, y, v = solve_both_policies(M)

    # normalize for safety (your solver should already do this)
    x = normalize_or_uniform(x)
    y = normalize_or_uniform(y)

    print("M:")
    for row in M:
        print(" ", [round(z, 3) for z in row])
    print("x:", [round(t, 6) for t in x])
    print("y:", [round(t, 6) for t in y])
    print("v:", round(v, 6))

    # Saddle checks
    saddle_checks(M, x, y, v, tol=1e-5, verbose=verbose)

    # Optional exact/near-exact checks
    if expect_x is not None:
        if check_exact:
            assert vec_close(x, expect_x, tol=1e-5), f"x != expected {expect_x}"
        else:
            assert almost_equal(sum(x), 1.0), "x not prob. vector"
    if expect_y is not None:
        if check_exact:
            assert vec_close(y, expect_y, tol=1e-5), f"y != expected {expect_y}"
        else:
            assert almost_equal(sum(y), 1.0), "y not prob. vector"
    if expect_v is not None:
        assert almost_equal(v, expect_v, tol=1e-5), f"v {v} != expected {expect_v}"

    print("✓ Passed.")

def add_constant(M, c):
    return [[M[i][j] + c for j in range(len(M[0]))] for i in range(len(M))]

def scale_matrix(M, k):
    return [[k * M[i][j] for j in range(len(M[0]))] for i in range(len(M))]

def test_all():
    # 1) Zero matrix (degenerate: any strategies ok; expect v=0; we accept uniform)
    M0 = [[0,0],
          [0,0]]
    run_case("Zero 2x2", M0, expect_v=0.0, verbose=True)

    # 2) Matching Pennies (value 0, uniform strategies)
    #    Row payoff: +1 on match for row? Classic form: [[1,-1],[-1,1]]
    MP = [[ 1, -1],
          [-1,  1]]
    run_case("Matching Pennies", MP,
             expect_x=[0.5, 0.5], expect_y=[0.5,0.5], expect_v=0.0, check_exact=True)

    # 3) Rock–Paper–Scissors (value 0, uniform strategies)
    #    R beats S, S beats P, P beats R; tie=0, win=+1, loss=-1
    RPS = [[ 0, -1,  1],   # R vs (R,P,S)
           [ 1,  0, -1],   # P vs (R,P,S)
           [-1,  1,  0]]   # S vs (R,P,S)
    run_case("RPS", RPS,
             expect_x=[1/3,1/3,1/3], expect_y=[1/3,1/3,1/3], expect_v=0.0, check_exact=True)

    # 4) Row domination: row 0 strictly dominates row 1
    M_dom_row = [[2, 2],
                 [1, 1]]
    # Expect x puts full mass on row 0; y arbitrary; v=2
    run_case("Row Dominates", M_dom_row,
             expect_x=[1.0, 0.0], expect_v=2.0)

    # 5) Column domination (for the opponent), e.g., left column always worse for row player
    # Make column 1 (index 0) always smaller than column 2 (index 1)
    M_dom_col = [[0, 1],
                 [0, 1]]
    # Here the column player (Agent2) should put all mass on column 0 to minimize payoff (since row wants max)
    # But since both rows give [0,1], the column player will choose column 0 and value v=0;
    # Row player's x can be anything (we only check v and saddle).
    run_case("Column Dominates (for column player)", M_dom_col, expect_v=0.0)

    # 6) Rectangular 3x2: optimal x puts mass on 1st two rows (intuitively), y nontrivial
    M_rect = [[1, 0],
              [0, 1],
              [0, 0]]
    run_case("Rectangular 3x2", M_rect)

    # 7) Random 4x4 (reproducible): check only saddle conditions
    random.seed(123)
    M_rand = [[round(random.uniform(-5, 10), 2) for _ in range(4)] for __ in range(4)]
    run_case("Random 4x4", M_rand)

    # 8) Shift invariance: policies same, value shifts by c
    c = 5.0
    x1, y1, v1 = solve_both_policies(MP)
    x2, y2, v2 = solve_both_policies(add_constant(MP, c))
    x1 = normalize_or_uniform(x1); y1 = normalize_or_uniform(y1)
    x2 = normalize_or_uniform(x2); y2 = normalize_or_uniform(y2)
    assert vec_close(x1, x2, tol=1e-6), "Shift invariance failed for x"
    assert vec_close(y1, y2, tol=1e-6), "Shift invariance failed for y"
    assert almost_equal(v2, v1 + c, tol=1e-6), "Shift invariance failed for v"
    print("\n✓ Shift invariance (policies same, value +c) passed.")

    # 9) Positive scaling: policies same, value scales
    k = 3.5
    x1, y1, v1 = solve_both_policies(RPS)
    x2, y2, v2 = solve_both_policies(scale_matrix(RPS, k))
    x1 = normalize_or_uniform(x1); y1 = normalize_or_uniform(y1)
    x2 = normalize_or_uniform(x2); y2 = normalize_or_uniform(y2)
    assert vec_close(x1, x2, tol=1e-6), "Scale invariance failed for x"
    assert vec_close(y1, y2, tol=1e-6), "Scale invariance failed for y"
    assert almost_equal(v2, k*v1, tol=1e-6), "Scale invariance failed for v"
    print("✓ Positive scaling invariance (policies same, value *k) passed.")

    print("\nAll tests completed.\n")



if __name__ == "__main__":
    random.seed(7)
    m = n = 4
    M = [[round(random.uniform(-5, 10), 2) for _ in range(n)] for __ in range(m)]
    #M = [[0 for _ in range(n)] for __ in range(m)]
    # Solve both policies at once
    x, y, v = solve_both_policies(M)

    print("Payoff matrix M:")
    for row in M:
        print(row)
    print("\nAgent1 policy x:", [round(t, 4) for t in x])
    print("Agent2 policy y:", [round(t, 4) for t in y])
    print("Game value v*:", round(v, 6))

    # Sanity checks (saddle property):
    # expected payoff vs each opponent pure column, using x
    evs_cols = [sum(x[i] * M[i][j] for i in range(m)) for j in range(n)]
    # expected payoff vs each opponent pure row, using y
    evs_rows = [sum(M[i][j] * y[j] for j in range(n)) for i in range(m)]
    print("\nE_x[M[:,j]] by column:", [round(z, 6) for z in evs_cols])
    print("E_y[M[i,:]] by row   :", [round(z, 6) for z in evs_rows])
    # Should have:  min_j E_x[M[:,j]]  ==  v  (up to num. error)
    #               max_i E_y[M[i,:]]  ==  v
    print("min_j E_x[M[:,j]] =", round(min(evs_cols), 6), "≈ v")
    print("max_i E_y[M[i,:]] =", round(max(evs_rows), 6), "≈ v")
    test_all()
