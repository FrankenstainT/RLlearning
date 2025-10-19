"""
Solving FrozenLake with Tabular Q-Learning for Two Agents (fixed)
Each agent trains independently on its own Gym env instance but uses the same map layout.
"""
from typing import NamedTuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import os

os.makedirs("results", exist_ok=True)
sns.set_theme()


class Params(NamedTuple):
    total_episodes: int
    learning_rate: float
    gamma: float
    epsilon: float
    map_size: int
    seed: int
    is_slippery: bool
    n_runs: int
    action_size: int
    state_size: int
    proba_frozen: float


params = Params(
    total_episodes=2000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=5,
    seed=123,
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
)

rng = np.random.default_rng(params.seed)


class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        # inplace update
        delta = reward + self.gamma * np.max(self.qtable[new_state, :]) - self.qtable[state, action]
        self.qtable[state, action] += self.learning_rate * delta

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, action_space, state, qtable):
        if rng.uniform(0, 1) < self.epsilon:
            return action_space.sample()
        max_ids = np.where(qtable[state, :] == np.max(qtable[state, :]))[0]
        return rng.choice(max_ids)


def run_env_two_agents(envs, agents, explorers, params, n_agents=2):
    """
    envs: list of Gym env instances (one per agent)
    """
    rewards = np.zeros((params.total_episodes, params.n_runs, n_agents))
    steps = np.zeros((params.total_episodes, params.n_runs, n_agents))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, n_agents, params.state_size, params.action_size))
    all_states = [[] for _ in range(n_agents)]
    all_actions = [[] for _ in range(n_agents)]

    for run in range(params.n_runs):
        # reset Q-tables each run
        for agent in agents:
            agent.reset_qtable()

        for episode in tqdm(episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False):
            # Reset each agent's environment independently but with same map
            states = [envs[i].reset(seed=(params.seed + run))[0] for i in range(n_agents)]
            done_flags = [False] * n_agents
            step_counts = [0] * n_agents
            total_rewards = [0] * n_agents

            done_episode = False
            while not done_episode:
                actions = [
                    explorers[i].choose_action(envs[i].action_space, states[i], agents[i].qtable)
                    if not done_flags[i]
                    else 0  # placeholder if already done (we won't step)
                    for i in range(n_agents)
                ]

                new_states = []
                for i in range(n_agents):
                    if done_flags[i]:
                        # if agent already done, keep same state and don't step that env
                        new_states.append(states[i])
                        continue

                    # step the agent's own environment (no cross-talk)
                    new_state, reward, terminated, truncated, info = envs[i].step(actions[i])
                    done_flags[i] = terminated or truncated
                    agents[i].update(states[i], actions[i], reward, new_state)

                    new_states.append(new_state)
                    all_states[i].append(states[i])
                    all_actions[i].append(actions[i])
                    total_rewards[i] += reward
                    step_counts[i] += 1

                states = new_states
                done_episode = all(done_flags)

            for i in range(n_agents):
                rewards[episode, run, i] = total_rewards[i]
                steps[episode, run, i] = step_counts[i]
                qtables[run, i, :, :] = agents[i].qtable

    return rewards, steps, episodes, qtables, all_states, all_actions


def postprocess(episodes, params, rewards, steps, map_size, n_agents=2):
    res_all = []
    st_all = []
    for i in range(n_agents):
        res = pd.DataFrame({
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards[:, :, i].flatten(order="F"),
            "Steps": steps[:, :, i].flatten(order="F"),
        })
        res["cum_rewards"] = rewards[:, :, i].cumsum(axis=0).flatten(order="F")
        res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])
        res["Agent"] = i + 1
        res_all.append(res)

        st = pd.DataFrame({"Episodes": episodes, "Steps": steps[:, :, i].mean(axis=1)})
        st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
        st["Agent"] = i + 1
        st_all.append(st)
    return pd.concat(res_all), pd.concat(st_all)


def qtable_directions_map(qtable, map_size):
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            qtable_directions[idx] = directions[val]
    return qtable_val_max, qtable_directions.reshape(map_size, map_size)


def plot_states_actions_distribution_multi(states_list, actions_list, map_size):
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}
    n_agents = len(states_list)
    fig, axes = plt.subplots(nrows=n_agents, ncols=2, figsize=(15, 5 * n_agents))
    for i in range(n_agents):
        sns.histplot(data=states_list[i], ax=axes[i, 0], kde=True)
        axes[i, 0].set_title(f"Agent {i + 1} States")
        sns.histplot(data=actions_list[i], ax=axes[i, 1])
        axes[i, 1].set_xticks(list(labels.values()), labels=labels.keys())
        axes[i, 1].set_title(f"Agent {i + 1} Actions")
    plt.tight_layout()
    plt.savefig(f"results/q_states_actions_distribution_multi_{map_size}.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_q_values_map_multi(qtables, map_size):
    n_agents = qtables.shape[0]
    fig, axes = plt.subplots(1, n_agents, figsize=(8 * n_agents, 6))
    # if only one agent, axes may not be an array
    if n_agents == 1:
        axes = [axes]
    for i in range(n_agents):
        qtable_val_max, qtable_directions = qtable_directions_map(qtables[i], map_size)
        sns.heatmap(
            qtable_val_max, annot=qtable_directions, fmt="", ax=axes[i],
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.7, linecolor="black",
            xticklabels=[], yticklabels=[], annot_kws={"fontsize": "xx-large"}
        ).set(title=f"Agent {i + 1} Learned Q-values")
        for _, spine in axes[i].spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("black")
    plt.tight_layout()
    plt.savefig(f"results/q_values_map_multi_{map_size}.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_steps_and_rewards(rewards_df, steps_df):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.lineplot(data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", style="Agent", ax=ax[0])
    ax[0].set(ylabel="Cumulated rewards")
    sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="map_size", style="Agent", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")
    for axi in ax:
        axi.legend(title="map size / Agent")
    plt.tight_layout()
    plt.savefig(f"results/steps_and_rewards_multi.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)


# Main loop over map sizes
map_sizes = [4]#[4, 7, 9, 11]
res_all = pd.DataFrame()
st_all = pd.DataFrame()
n_agents = 2

for map_size in map_sizes:
    # create a single map layout (desc) and then create one env per agent with same desc
    desc = generate_random_map(size=map_size, p=params.proba_frozen, seed=params.seed)
    envs = [
        gym.make("FrozenLake-v1", is_slippery=params.is_slippery, render_mode="rgb_array", desc=desc)
        for _ in range(n_agents)
    ]

    # update params based on one env's action/state space
    params = params._replace(action_size=envs[0].action_space.n)
    params = params._replace(state_size=envs[0].observation_space.n)
    for env in envs:
        env.action_space.seed(params.seed)

    agents = [Qlearning(params.learning_rate, params.gamma, params.state_size, params.action_size) for _ in
              range(n_agents)]
    explorers = [EpsilonGreedy(params.epsilon) for _ in range(n_agents)]

    print(f"Map size: {map_size}x{map_size}")
    rewards, steps, episodes, qtables, all_states, all_actions = run_env_two_agents(envs, agents, explorers, params,
                                                                                    n_agents)

    res, st = postprocess(episodes, params, rewards, steps, map_size, n_agents)
    res_all = pd.concat([res_all, res])
    st_all = pd.concat([st_all, st])

    qtable_avg = qtables.mean(axis=0)  # average Q-tables across runs
    plot_states_actions_distribution_multi(all_states, all_actions, map_size)
    plot_q_values_map_multi(qtable_avg, map_size)

    for env in envs:
        env.close()

plot_steps_and_rewards(res_all, st_all)
