import numpy as np
import pulp_verify
import sys


class MiniMaxQLearner():
    def __init__(self,
                 aid=None,
                 alpha=0.1,
                 policy=None,
                 gamma=0.8,
                 ini_state="nonstate",
                 actions=None):

        self.aid = aid
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.actions = actions
        self.state = ini_state
        self.q = {}
        self.q[ini_state] = {}
        self.pi = {}
        self.pi[ini_state] = np.repeat(1.0 / len(self.actions), len(self.actions))
        self.v = {}

        self.previous_action = None
        self.reward_history = []
        self.episode_max_delta = 0.0
        self.episode_deltas = []  # stores per-episode max deltas

    def act(self, training=True):
        if training:
            action_id = self.policy.select_action(self.pi[self.state])
            action = self.actions[action_id]
            self.previous_action = action
        else:
            action_id = self.policy.select_greedy_action(self.pi)
            action = self.actions[action_id]

        return action

    def observe(self, state="nonstate", reward=None, opponent_action=None, is_learn=True):
        if not is_learn:
            return 0.0
        self.check_new_state(state)
        # --- get old Q before update ---
        old_q = self.q[state].copy()

        # do learning
        self.learn(state, reward, opponent_action)

        # --- get new Q after update ---
        new_q = self.q[state]

        # measure max change across all Q-values in this state
        diffs = []
        for k in new_q.keys():
            old_val = old_q.get(k, 0.0)  # if not existed before, treat as 0
            diffs.append(abs(new_q[k] - old_val))
        self.episode_max_delta = max(self.episode_max_delta, max(diffs) if diffs else 0.0)

    def learn(self, state_after_step, reward, opponent_action):
        self.reward_history.append(reward)
        self.q[self.state][(self.previous_action, opponent_action)] = self.compute_q(state_after_step, reward, opponent_action)
        self.pi[self.state] = self.compute_pi()
        self.v[self.state] = self.compute_v()

    def compute_q(self, state_after_step, reward, opponent_action):
        if (self.previous_action, opponent_action) not in self.q[self.state].keys():
            self.q[self.state][(self.previous_action, opponent_action)] = 0.0
        q = self.q[self.state][(self.previous_action, opponent_action)]
        if state_after_step not in self.v.keys():
            self.v[state_after_step] = 0
        updated_q = q + (self.alpha * (reward + self.gamma * self.v[state_after_step] - q))

        return updated_q

    def compute_v(self):
        min_expected_value = sys.maxsize
        for action2 in self.actions:
            expected_value = sum(
                [self.pi[self.state][action1] * self.q[self.state][(action1, action2)] for action1 in self.actions])
            if expected_value < min_expected_value:
                min_expected_value = expected_value

        return min_expected_value

    def compute_pi(self):
        pi = pulp.LpVariable.dicts("pi", range(len(self.actions)), 0, 1)
        max_min_value = pulp.LpVariable("max_min_value")
        lp_prob = pulp.LpProblem("Maxmin Problem", pulp.LpMaximize)
        lp_prob += (max_min_value, "Objective")

        lp_prob += (pulp.lpSum([pi[i] for i in range(len(self.actions))]) == 1)
        for action2 in self.actions:
            values = pulp.lpSum(
                [pi[idx] * self.q[self.state][(action1, action2)] for idx, action1 in enumerate(self.actions)])
            conditon = max_min_value <= values
            lp_prob += conditon
        lp_prob.solve(pulp.PULP_CBC_CMD(msg=0))

        return np.array([pi[i].value() for i in range(len(self.actions))])


    def check_new_state(self, state):
        if state not in self.q.keys():
            self.q[state] = {}

        for action1 in self.actions:
            for action2 in self.actions:
                if state not in self.pi.keys():
                    self.pi[state] = np.repeat(1.0 / len(self.actions), len(self.actions))
                    self.v[state] = np.random.random()
                if (action1, action2) not in self.q[state].keys():
                    self.q[state][(action1, action2)] = np.random.random()

    def end_episode(self):
        self.episode_deltas.append(self.episode_max_delta)
        self.episode_max_delta = 0.0
