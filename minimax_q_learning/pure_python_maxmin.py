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


def solve_maxmin(actions, Q_state):
    n = len(actions)
    var_count = n + 2  # pi_i, v_pos, v_neg

    A = []
    b = []

    for a2 in actions:
        row = [0.0] * var_count
        for i, a1 in enumerate(actions):
            row[i] = -Q_state[(a1, a2)]
        row[n] = 1.0
        row[n + 1] = -1.0
        A.append(row)
        b.append(0.0)

    # sum pi_i = 1 → two inequalities
    row = [0.0] * var_count
    for i in range(n):
        row[i] = 1.0
    A.append(row.copy())
    b.append(1.0)
    for i in range(n):
        row[i] = -1.0
    A.append(row.copy())
    b.append(-1.0)

    c = [0.0] * var_count
    c[n] = 1.0
    c[n + 1] = -1.0

    opt, x = simplex(c, A, b)
    pi = [max(0.0, x[i]) for i in range(n)]
    v = x[n] - x[n + 1]

    s = sum(pi)
    if s <= EPS:  # all zero? fallback to uniform strategy
        pi = [1.0 / n] * n
    else:
        pi = [p / s for p in pi]

    return pi, v


if __name__ == "__main__":
    n = 4
    actions = [f"A{i}" for i in range(n)]

    Q = {}
    random.seed(42)
    for a1 in actions:
        for a2 in actions:
            Q[(a1, a2)] = round(random.uniform(-5, 10), 2)
            #Q[(a1, a2)] = 0

    # Prepare output as string
    output_lines = []
    output_lines.append("Payoff matrix Q(a1,a2):")
    for a1 in actions:
        output_lines.append(str([Q[(a1, a2)] for a2 in actions]))

    pi, v = solve_maxmin(actions, Q)
    output_lines.append("\nOptimal mixed strategy π = " + str([round(p, 4) for p in pi]))
    output_lines.append("Max–min value v* = " + str(round(v, 4)))

    output_lines.append("\nExpected payoffs vs each opponent action:")
    for a2 in actions:
        val = sum(pi[i] * Q[(actions[i], a2)] for i in range(n))
        output_lines.append(f"{a2}: {val:.4f}")

    # Print to stdout
    print("\n".join(output_lines))

    # Print to file
    with open("maxmin_output.txt", "w") as f:
        f.write("\n".join(output_lines))
