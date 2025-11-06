import numpy as np
import torch

# --------------------------------------------------------
# device + dtype helper
# --------------------------------------------------------
def get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float64  # real GPU, use 64-bit
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple M-series: MPS is float32 only right now
        return torch.device("mps"), torch.float32
    else:
        # CPU: we can use float64
        return torch.device("cpu"), torch.float64


def project_simplex(z: torch.Tensor) -> torch.Tensor:
    """
    Project each row vector of z onto the probability simplex.
    Works for 1D too.
    """
    orig_shape = z.shape
    if z.dim() == 1:
        z = z.unsqueeze(0)

    # sort descending
    u, _ = torch.sort(z, dim=1, descending=True)
    cumsum = torch.cumsum(u, dim=1)
    rhos = torch.arange(1, z.size(1) + 1, device=z.device, dtype=z.dtype).view(1, -1)
    cond = u * rhos > (cumsum - 1)
    rho = cond.sum(dim=1) - 1

    theta = (cumsum.gather(1, rho.view(-1, 1)) - 1) / (rho + 1).to(z.dtype).view(-1, 1)
    w = torch.clamp(z - theta, min=0.0)
    w = w / w.sum(dim=1, keepdim=True)  # safety renorm

    if orig_shape == torch.Size([z.size(1)]):
        return w.squeeze(0)
    return w


@torch.no_grad()
def solve_zero_sum_torch_extragrad(M_np: np.ndarray,
                                   iters: int = 20000,
                                   lr: float = 0.2):
    """
    Approximate zero-sum matrix game (row maximizes, col minimizes)
    using *extragradient* + ergodic averaging.

    If run on CUDA/CPU in float64, this can get Δv ~ 1e-8 vs the LP
    for small matrices. On MPS (float32) expect ~1e-6 to 1e-5.
    """
    device, dtype = get_device_and_dtype()

    M = torch.as_tensor(M_np, device=device, dtype=dtype)
    m, n = M.shape

    # start uniform
    x = torch.full((m,), 1.0 / m, device=device, dtype=dtype)
    y = torch.full((n,), 1.0 / n, device=device, dtype=dtype)

    # running (ergodic) averages
    x_bar = torch.zeros_like(x)
    y_bar = torch.zeros_like(y)

    for t in range(1, iters + 1):
        # ----- 1) gradient at current point -----
        My = M @ y          # (m,)
        MTx = M.t() @ x     # (n,)

        # ----- 2) look-ahead (extragrad step) -----
        x_tilde = project_simplex(x + lr * My)
        y_tilde = project_simplex(y - lr * MTx)

        # ----- 3) gradient at look-ahead point -----
        My_tilde = M @ y_tilde
        MTx_tilde = M.t() @ x_tilde

        # ----- 4) final update using look-ahead gradient -----
        x = project_simplex(x + lr * My_tilde)
        y = project_simplex(y - lr * MTx_tilde)

        # ----- 5) ergodic averaging (improves value accuracy) -----
        # weighted average: x_bar = (1 - 1/t)*x_bar + (1/t)*x
        x_bar = x_bar + (x - x_bar) / t
        y_bar = y_bar + (y - y_bar) / t

    # value with the averaged strategies
    v = (x_bar @ M @ y_bar).item()

    return (
        x_bar.detach().cpu().numpy(),
        y_bar.detach().cpu().numpy(),
        float(v),
        str(device),
        str(dtype),
    )
from scipy.optimize import linprog

def solve_cpu_lp(M: np.ndarray):
    # your existing one-LP solver, lightly inlined for test
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

    c = np.zeros(num); c[vidx] = -1
    A_ub = []; b_ub = []
    for j in range(n):
        row = np.zeros(num)
        row[xs] = -Ms[:, j]
        row[vidx] = 1
        A_ub.append(row); b_ub.append(0)
    for i in range(m):
        row = np.zeros(num)
        row[ys] = Ms[i, :]
        row[vidx] = -1
        A_ub.append(row); b_ub.append(0)
    A_ub = np.vstack(A_ub); b_ub = np.array(b_ub)

    A_eq = np.zeros((2, num))
    A_eq[0, xs] = 1
    A_eq[1, ys] = 1
    b_eq = np.array([1., 1.])

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


def test_one(M, name):
    x_cpu, y_cpu, v_cpu = solve_cpu_lp(M)
    x_t, y_t, v_t, device, dtype = solve_zero_sum_torch_extragrad(M, iters=20000, lr=0.2)

    print("="*70)
    print(f"Test: {name}, shape={M.shape}")
    print(f"CPU v:   {v_cpu:.12f}")
    print(f"Torch v: {v_t:.12f} (device={device}, dtype={dtype})")
    print(f"Δv = {abs(v_cpu - v_t):.12e}")
    print(f"L1(x): {np.sum(np.abs(x_cpu - x_t)):.6e}")
    print(f"L1(y): {np.sum(np.abs(y_cpu - y_t)):.6e}")


if __name__ == "__main__":
    np.random.seed(0)

    M1 = np.array([[0., 1., -1.],
                   [-1., 0., 1.],
                   [1., -1., 0.]])
    test_one(M1, "rock-paper-scissors")

    M2 = np.random.randn(5, 5)
    test_one(M2, "random_5x5")

    # almost-constant case
    base = 2.0
    noise = 1e-4 * np.random.randn(4, 4)
    M3 = base + noise
    test_one(M3, "almost_const_4x4")
