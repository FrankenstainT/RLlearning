# run_soccer.py
import os, pickle, time
import numpy as np

from Games.soccer import Soccer                      # adjust to Games.Soccer if your file is capitalized
from Players.minimaxq_player import MinimaxQPlayer   # this is your class shown above

# ---------------- Hyperparams ----------------
EPISODES = 200_000
GAMMA    = 0.90
EPS      = 0.10     # exploration prob used by your class
# Your class uses alpha multiplicatively: alpha_t <- alpha_{t-1} * decay, starting at 1.0
# Pick a decay close to 1 so alpha decays slowly (e.g., ~0.995 per episode worth of steps).
DECAY    = 0.99999

# ---------------- Build env & agents ----------------
env = Soccer()                 # must support reset()/step()
s  = env.reset()
nS = getattr(env, "n_states", None)
nA = getattr(env, "n_actions", 5)   # Soccer has 5 actions

if nS is None:
    raise RuntimeError("Env must expose n_states (int). If you still use the legacy env, upgrade to the modern version I sent.")

A = MinimaxQPlayer(numStates=nS, numActionsA=nA, numActionsB=nA, decay=DECAY, expl=EPS, gamma=GAMMA)
B = MinimaxQPlayer(numStates=nS, numActionsA=nA, numActionsB=nA, decay=DECAY, expl=EPS, gamma=GAMMA)

# ---------------- Train ----------------
returns = []
for ep in range(1, EPISODES + 1):
    s = env.reset()
    done = False
    G = 0.0
    while not done:
        aA = A.chooseAction(s)
        aB = B.chooseAction(s)
        s2, (rA, rB), done, _ = env.step((aA, aB))

        # Your MinimaxQPlayer expects: (initialState, finalState, actions, reward_for_row_player)
        A.getReward(initialState=s, finalState=s2, actions=(aA, aB), reward=rA)
        B.getReward(initialState=s, finalState=s2, actions=(aA, aB), reward=rB)

        G += rA
        s = s2

    returns.append(G)
    if ep % 5000 == 0:
        print(f"[{ep}] avg return (last 5k) = {np.mean(returns[-5000:]):+.4f}")

# ---------------- Save ----------------
os.makedirs("artifacts", exist_ok=True)
ckpt = {
    "Q_A": A.Q, "Q_B": B.Q,
    "pi_A": A.pi, "pi_B": B.pi,
    "V_A": A.V, "V_B": B.V,
    "meta": dict(episodes=EPISODES, gamma=GAMMA, decay=DECAY, epsilon=EPS, ts=time.time())
}
with open("artifacts/soccer_minimaxq.pkl", "wb") as f:
    pickle.dump(ckpt, f)
print("Saved -> artifacts/soccer_minimaxq.pkl")
