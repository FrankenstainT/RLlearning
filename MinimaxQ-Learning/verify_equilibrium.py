import pickle, numpy as np
from scipy.optimize import linprog
from Games.soccer import Soccer

ACTION_NAMES = ["L","U","R","D","S"]

def minimax_row(M):
    nA, nB = M.shape
    # vars = [v, x_0..x_{nA-1}]
    c = np.zeros(nA+1); c[0] = -1.0                     # maximize v -> minimize -v
    A_ub = np.hstack([ np.ones((nB,1)), -M.T ])         # 1*v - M^T x <= 0  (i.e., M^T x >= v 1)
    b_ub = np.zeros(nB)
    A_eq = np.zeros((1,nA+1)); A_eq[0,1:] = 1.0         # sum x = 1
    b_eq = np.array([1.0])
    bounds = [(None,None)] + [(0.0,1.0)]*nA
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    x = res.x[1:]; v = res.x[0]
    return x, v, res

with open("artifacts/soccer_minimaxq.pkl","rb") as f:
    ckpt = pickle.load(f)

Q = ckpt["Q_A"]     # shape [S, A, B]
env = Soccer()
states_to_check = [0, 5, 17, 42]   # replace with your interesting states

for s in states_to_check:
    M = Q[s]                        # [A,B]
    x, v, _ = minimax_row(M)
    payoffs = M.T @ x               # value vs each column action
    j = int(np.argmin(payoffs))
    gap = payoffs[j] - v
    pos, owner = env.decode_state(s)
    supp = [ACTION_NAMES[i] for i,p in enumerate(x) if p>1e-5]
    print(f"s={s} A@{tuple(pos[0])} B@{tuple(pos[1])} own={'A' if owner==0 else 'B'} | "
          f"v*={v:+.4f} vs BR={payoffs[j]:+.4f} gap={gap:+.4f} | supp={supp}, col_BR={ACTION_NAMES[j]}")
