import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
from enum import Enum
from collections import deque

class ExpectedSARSAMode(Enum):
    TRAIN = 0
    TEST = 1

class ExpectedSARSAAgent:
    def __init__(self, env):
        self.gamma = 0.99
        self.learning_rate = 0.02
        self.epsilon = 0.01
        self.states_n = env.observation_space.n
        self.actions_n = env.action_space.n
        self.q = np.zeros((self.states_n, self.actions_n))

    def reset(self, mode: ExpectedSARSAMode):
        self.mode = mode
        if self.mode == ExpectedSARSAMode.TRAIN:
            self.trajectory = deque(maxlen=8)

    def step(self, observation, reward, terminated):
        if self.mode == ExpectedSARSAMode.TRAIN and np.random.uniform() < self.epsilon:
            action = np.random.randint(self.actions_n)
        else:
            action = self.q[observation].argmax()

        if self.mode == ExpectedSARSAMode.TRAIN:
            self.trajectory.extend([observation, reward, terminated, action])
            if len(self.trajectory) >= 8:
                self.learn()
        return action

    def close(self):
        pass

    def learn(self):
        state, _, _, action, next_state, reward, terminated, _ = list(self.trajectory)
        v = self.q[next_state].mean() * self.epsilon + self.q[next_state].max() * (1 - self.epsilon)
        target = reward + self.gamma * v * (1 - terminated)
        td_error = target - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error

def play_episode(env, agent, seed=None, mode=ExpectedSARSAMode.TRAIN):
    observation, _ = env.reset(seed=seed)
    reward, terminated, truncated = 0, False, False
    agent.reset(mode)
    episode_reward, elapsed_steps = 0, 0

    done = False
    while not done:
        action = agent.step(observation, reward, terminated)
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        elapsed_steps += 1
        done = terminated or truncated

    agent.close()
    return episode_reward, elapsed_steps


def train(env, agent, num_episodes, seed=None):
    episode_rewards = []
    episode_rewards_averaged = []
    progress = tqdm(range(num_episodes))
    for episode in progress:
        episode_reward, elapsed_steps = play_episode(env, agent, seed=seed, mode=ExpectedSARSAMode.TRAIN)
        episode_rewards.append(episode_reward)

        mean_reward = np.mean(episode_rewards[-200:])
        progress.set_postfix({'mean_reward': mean_reward})
        if episode % 500 == 0:
            episode_rewards_averaged.append(mean_reward)

        if mean_reward > env.spec.reward_threshold:
            print("breaking")
            break

    plt.figure()
    plt.plot(episode_rewards_averaged)
    plt.show()

if __name__ == "__main__":
    env = gym.make('Taxi-v3')
    agent = ExpectedSARSAAgent(env)
    train(env, agent, 100000)