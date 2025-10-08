from discrete import EpsGreedyPolicy, PursuitEvasionParallelEnv
pursuer_policy = EpsGreedyPolicy(epsilon=0.2)
evader_policy = EpsGreedyPolicy(epsilon=0.2)
# set policies deterministic (no exploration)
pursuer_policy.epsilon = 0.0
evader_policy.epsilon = 0.0
env = PursuitEvasionParallelEnv(grid_size=5, max_steps=30, obstacles=[(2,2), (1,1)])
# place pursuer left of evader, test swapping capture
env.pos = {"pursuer": [3,3], "evader":[3,4]}
obs = env._get_obs()
actions = {"pursuer": 3, "evader": 2}  # pursuer -> right, evader -> left => swap
next_obs, rewards, terms, truncs, infos = env.step(actions)
print("Swap test -> pos:", env.pos, "rewards:", rewards, "winner:", infos['pursuer']['winner'])

# test safe zone: evader moves to safe zone, pursuer far
env.pos = {"pursuer": [0,0], "evader": [4,4]}  # grid_size=5
actions = {"pursuer": 4, "evader": 2}  # evader moves left toward safe zone maybe
for t in range(5):
    next_obs, rewards, terms, truncs, infos = env.step(actions)
    print("step", t, "pos", env.pos, "rewards", rewards, "winner", infos['pursuer']['winner'])
    if any(terms.values()) or any(truncs.values()):
        break
