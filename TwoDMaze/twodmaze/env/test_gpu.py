import numpy as np
import torch
from scipy.optimize import linprog


# --------------------------------------------------------
# CPU reference (your one-LP)
# --------------------------------------------------------
def solve_cpu_lp(M: np.ndarray):
    M = np.asarray(M, float)
    mu = float(np.mean(M))
    Ms = M - mu
    s = float(np.max(np.abs(Ms)))
    if s < 1e-12:
        m, n = M.shape
        x = np.ones(m)/m
        y = np.ones(n)/n
        return x, y, mu

    Ms /= s
    m, n = Ms.shape
    num = m + n + 1
    xs = slice(0, m); ys = slice(m, m+n); vidx = m+n

    c = np.zeros(num); c[vidx] = -1.0
    A_ub = []; b_ub = []
    for j in range(n):
        row = np.zeros(num)
        row[xs] = -Ms[:, j]
        row[vidx] = 1.0
        A_ub.append(row); b_ub.append(0.0)
    for i in range(m):
        row = np.zeros(num)
        row[ys] = Ms[i, :]
        row[vidx] = -1.0
        A_ub.append(row); b_ub.append(0.0)
    A_ub = np.vstack(A_ub); b_ub = np.array(b_ub)

    A_eq = np.zeros((2, num))
    A_eq[0, xs] = 1.0
    A_eq[1, ys] = 1.0
    b_eq = np.array([1.0, 1.0])

    bounds = [(0, None)]*m + [(0, None)]*n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    z = res.x
    x = z[:m]; y = z[m:m+n]; v = z[m+n]
    x = np.clip(x, 0, None); y = np.clip(y, 0, None)
    x /= x.sum(); y /= y.sum()
    v = v * s + mu
    return x, y, v


# --------------------------------------------------------
# torch helper
# --------------------------------------------------------
def get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float64
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    else:
        return torch.device("cpu"), torch.float64


def project_simplex(z: torch.Tensor) -> torch.Tensor:
    orig_1d = (z.dim() == 1)
    if orig_1d:
        z = z.unsqueeze(0)

    u, _ = torch.sort(z, dim=1, descending=True)
    cumsum = torch.cumsum(u, dim=1)
    r = torch.arange(1, z.size(1)+1, device=z.device, dtype=z.dtype).view(1, -1)
    cond = u * r > (cumsum - 1)
    rho = cond.sum(dim=1) - 1
    theta = (cumsum.gather(1, rho.view(-1, 1)) - 1) / (rho + 1).to(z.dtype).view(-1, 1)
    w = torch.clamp(z - theta, min=0.0)
    w = w / w.sum(dim=1, keepdim=True)
    if orig_1d:
        return w.squeeze(0)
    return w


@torch.no_grad()
def solve_zero_sum_torch_fast(M_np: np.ndarray,
                              max_iters: int = 4000,
                              lr: float = 0.3,
                              tol_v: float = 1e-5):
    """
    Faster extragrad version:
    - fewer iters
    - lr decay
    - early stop on value change
    """
    device, dtype = get_device_and_dtype()
    M = torch.as_tensor(M_np, device=device, dtype=dtype)
    m, n = M.shape

    x = torch.full((m,), 1.0 / m, device=device, dtype=dtype)
    y = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
    x_bar = torch.zeros_like(x)
    y_bar = torch.zeros_like(y)

    prev_v = None

    for t in range(1, max_iters + 1):
        # simple lr decay
        lr_t = lr / (1.0 + 0.0005 * t)

        My = M @ y
        MTx = M.t() @ x

        x_tilde = project_simplex(x + lr_t * My)
        y_tilde = project_simplex(y - lr_t * MTx)

        My_tilde = M @ y_tilde
        MTx_tilde = M.t() @ x_tilde

        x = project_simplex(x + lr_t * My_tilde)
        y = project_simplex(y - lr_t * MTx_tilde)

        # averaged
        x_bar = x_bar + (x - x_bar) / t
        y_bar = y_bar + (y - y_bar) / t

        # compute value every ~100 steps for early stop
        if t % 100 == 0 or t == max_iters:
            v = (x_bar @ M @ y_bar).item()
            if prev_v is not None and abs(v - prev_v) < tol_v:
                break
            prev_v = v

    v = (x_bar @ M @ y_bar).item()
    return x_bar.cpu().numpy(), y_bar.cpu().numpy(), float(v), str(device)


# --------------------------------------------------------
# quick test harness
# --------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    tests = [
        ("rock-paper-scissors",
         np.array([[0., 1., -1.],
                   [-1., 0., 1.],
                   [1., -1., 0.]])),
        ("random_5x5", np.random.randn(5, 5)),
        ("almost_const_4x4", 2.0 + 1e-4 * np.random.randn(4, 4)),
    ]

    for name, M in tests:
        x_cpu, y_cpu, v_cpu = solve_cpu_lp(M)
        x_t, y_t, v_t, dev = solve_zero_sum_torch_fast(M)
        print("="*70)
        print(f"Test: {name}, shape={M.shape}")
        print(f"CPU v:   {v_cpu:.12f}")
        print(f"Torch v: {v_t:.12f} (device={dev})")
        print(f"Δv = {abs(v_cpu - v_t):.12e}")
        print(f"L1(x): {np.sum(np.abs(x_cpu - x_t)):.6e}")
        print(f"L1(y): {np.sum(np.abs(y_cpu - y_t)):.6e}")
