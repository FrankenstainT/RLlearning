import numpy as np
from scipy.optimize import linprog
import torch


# =========================================================
# 1) CPU reference: your LP-style solver
# =========================================================
def solve_both_policies_one_lp(M: np.ndarray, jitter=1e-8, max_retries=2):
    M = np.asarray(M, float)
    if not np.all(np.isfinite(M)):
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    mu = float(np.mean(M))
    Ms = M - mu
    s = float(np.max(np.abs(Ms)))
    if s < 1e-12:
        m, n = M.shape
        x = np.ones(m) / m
        y = np.ones(n) / n
        return x, y, mu

    Ms /= s

    def _solve_raw(A):
        m, n = A.shape
        num = m + n + 1
        xs = slice(0, m)
        ys = slice(m, m + n)
        vidx = m + n

        c = np.zeros(num)
        c[vidx] = -1.0  # maximize v

        A_ub = []
        b_ub = []

        # v <= x^T A[:, j]
        for j in range(n):
            row = np.zeros(num)
            row[xs] = -A[:, j]
            row[vidx] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)

        # A[i, :] y <= v
        for i in range(m):
            row = np.zeros(num)
            row[ys] = A[i, :]
            row[vidx] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

        A_ub = np.vstack(A_ub)
        b_ub = np.array(b_ub)

        A_eq = np.zeros((2, num))
        A_eq[0, xs] = 1.0
        A_eq[1, ys] = 1.0
        b_eq = np.array([1.0, 1.0])

        bounds = [(0, None)] * m + [(0, None)] * n + [(None, None)]

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        return res

    A = Ms
    for attempt in range(max_retries + 1):
        res = _solve_raw(A)
        if res.status == 0 and np.isfinite(res.fun):
            z = res.x
            m, n = Ms.shape
            x = np.clip(z[:m], 0, None)
            y = np.clip(z[m:m + n], 0, None)
            v = float(z[m + n])

            sx, sy = x.sum(), y.sum()
            x = x / sx if sx > 0 else np.ones(m) / m
            y = y / sy if sy > 0 else np.ones(n) / n

            v = v * s + mu  # rescale
            return x, y, v

        # jitter and retry
        A = Ms + np.random.default_rng().normal(scale=jitter, size=Ms.shape)

    # fallback
    m, n = M.shape
    x = np.ones(m) / m
    y = np.ones(n) / n
    v = float(np.mean(M))
    return x, y, v


# =========================================================
# 2) Torch helper: projection to simplex
# =========================================================
def project_simplex_torch(x, dim=-1):
    # x: (..., d)
    u, _ = torch.sort(x, descending=True, dim=dim)
    cssv = torch.cumsum(u, dim=dim) - 1
    idx = torch.arange(x.size(dim), device=x.device).view(
        *([1] * (x.dim() - 1)), -1
    ) + 1
    cond = u > cssv / idx
    rho = cond.sum(dim=dim, keepdim=True)
    theta = cssv.gather(dim, rho - 1) / rho
    return torch.clamp(x - theta, min=0.0)


# =========================================================
# 3) Torch version: extragradient + primal–dual gap
# =========================================================
def zero_sum_saddle_extragrad(
    M: torch.Tensor,
    lr=0.05,
    max_steps=8000,
    gap_tol=1e-4,
):
    """
    M: (B, m, n)
    returns x, y, v, gap
    Uses extragradient (mirror-prox style) and stops when
    max_i (M y)_i - min_j (x^T M)_j <= gap_tol
    """
    B, m, n = M.shape
    device = M.device
    x = torch.full((B, m), 1.0 / m, device=device, dtype=M.dtype)
    y = torch.full((B, n), 1.0 / n, device=device, dtype=M.dtype)

    for step in range(max_steps):
        # current payoffs
        My = torch.matmul(M, y.unsqueeze(-1)).squeeze(-1)     # (B, m)
        xM = torch.matmul(x.unsqueeze(1), M).squeeze(1)       # (B, n)

        # best-response values for gap
        row_best = My.max(dim=1).values        # max over i
        col_worst = xM.min(dim=1).values       # min over j
        gap = (row_best - col_worst).max().item()

        if gap <= gap_tol:
            break

        # -------- extragradient step --------
        # 1) extrapolate
        x_tilde = project_simplex_torch(x + lr * My)
        y_tilde = project_simplex_torch(y - lr * xM)

        # 2) recompute grads at extrapolated point
        My_tilde = torch.matmul(M, y_tilde.unsqueeze(-1)).squeeze(-1)
        xM_tilde = torch.matmul(x_tilde.unsqueeze(1), M).squeeze(1)

        # 3) actual update
        x = project_simplex_torch(x + lr * My_tilde)
        y = project_simplex_torch(y - lr * xM_tilde)

    # final value
    v = (x.unsqueeze(1) @ M @ y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return x, y, v, gap


# =========================================================
# 4) test harness
# =========================================================
def run_one_test(M):
    print("=" * 60)
    print("Payoff matrix M:")
    print(M)

    # CPU reference
    cpu_x, cpu_y, cpu_v = solve_both_policies_one_lp(M)
    print("\nCPU LP solution:")
    print("x (row):", cpu_x)
    print("y (col):", cpu_y)
    print("value  :", cpu_v)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Mt = torch.tensor(M, dtype=torch.float32, device=device).unsqueeze(0)

    # we'll try a few lrs and pick the best by value error
    best = None
    for lr in (0.1, 0.05, 0.02):
        x_t, y_t, v_t, gap = zero_sum_saddle_extragrad(
            Mt, lr=lr, max_steps=8000, gap_tol=5e-5
        )
        x_np = x_t[0].detach().cpu().numpy()
        y_np = y_t[0].detach().cpu().numpy()
        v_np = float(v_t[0].detach().cpu().item())
        err = abs(cpu_v - v_np)

        if best is None or err < best["err"]:
            best = {
                "lr": lr,
                "x": x_np,
                "y": y_np,
                "v": v_np,
                "gap": gap,
                "err": err,  # <<< store error
            }

    print("\nTorch extragrad solution (device={}, lr={}):".format(device, best["lr"]))
    print("x (row):", best["x"])
    print("y (col):", best["y"])
    print("value  :", best["v"])
    print("primal–dual gap (max over batch):", best["gap"])

    # diffs
    print("\nDiffs vs CPU:")
    print("||x_cpu - x_torch||_1 =", np.abs(cpu_x - best["x"]).sum())
    print("||y_cpu - y_torch||_1 =", np.abs(cpu_y - best["y"]).sum())
    print("|v_cpu - v_torch|     =", abs(cpu_v - best["v"]))


def main():
    # 1) classic 2x2
    M1 = np.array([[1.0, -1.0],
                   [-1.0, 1.0]], dtype=float)

    # 2) your random 4x4
    M2 = np.array([
        [0.12573022, -0.13210486, 0.64042265, 0.10490012],
        [-0.53566937, 0.36159505, 1.30400005, 0.94708096],
        [-0.70373524, -1.26542147, -0.62327446, 0.04132598],
        [-2.32503077, -0.21879166, -1.24591095, -0.73226735],
    ], dtype=float)

    # 3) your 3x3 slightly biased
    M3 = np.array([
        [0.5, 0.2, -0.1],
        [1.0, -0.3, 0.0],
        [0.0, 0.1, 0.2],
    ], dtype=float)

    for M in (M1, M2, M3):
        run_one_test(M)


if __name__ == "__main__":
    main()
