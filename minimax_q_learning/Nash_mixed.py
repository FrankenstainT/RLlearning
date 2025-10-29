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


if __name__ == "__main__":
    random.seed(7)
    m = n = 4
    M = [[round(random.uniform(-5, 10), 2) for _ in range(n)] for __ in range(m)]

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
