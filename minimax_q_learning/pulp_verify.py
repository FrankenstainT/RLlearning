import pulp
import random

# Generate random 4x4 matrix
n = 4
actions = [f"A{i}" for i in range(n)]
Q = {}
# random.seed(42)
# for a1 in actions:
#     for a2 in actions:
#         #Q[(a1, a2)] = round(random.uniform(-5, 10), 2)
#         Q[(a1, a2)] = 0
import numpy as np
np.random.seed(7)
Q = np.random.uniform(-5, 10, size=(4, 4))  # 4x4 example
Q = {(actions[i], actions[j]): float(Q[i, j]) for i in range(n) for j in range(n)}
# Define LP
pi = pulp.LpVariable.dicts("pi", range(n), lowBound=0)
v = pulp.LpVariable("v")
model = pulp.LpProblem("MaxMin", pulp.LpMaximize)
model += v

# Constraints
model += pulp.lpSum(pi[i] for i in range(n)) == 1
for a2 in actions:
    model += v <= pulp.lpSum(pi[i] * Q[(actions[i], a2)] for i in range(n))

# Solve
model.solve(pulp.PULP_CBC_CMD(msg=0))
pi_opt = [pulp.value(pi[i]) for i in range(n)]
v_opt = pulp.value(v)

print("PuLP optimal π =", [round(p, 4) for p in pi_opt])
print("PuLP optimal v =", round(v_opt, 4))
print("Payoff matrix Q(a1,a2):")
for a1 in actions:
    print([Q[(a1, a2)] for a2 in actions])
