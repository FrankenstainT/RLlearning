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
