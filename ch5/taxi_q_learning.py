from collections import deque

import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from enum import Enum


class QLearningMode(Enum):
    TRAIN = 0
    TEST = 1

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = 0.01
        self.gamma = 0.99
        self.learning_rate = 0.02
        self.q = np.zeros((self.state_size, self.action_size))

    def reset(self, mode):
        self.mode = mode
        if self.mode == QLearningMode.TRAIN:
            self.trajectory = deque(maxlen=8)

    def step(self, observation, reward, terminated):
        if self.mode == QLearningMode.TRAIN and np.random.uniform() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = self.q[observation].argmax()

        if self.mode == QLearningMode.TRAIN:
            self.trajectory.extend([observation, reward, terminated, action])
            if len(self.trajectory) >= 8:
                self.learn()
        return action

    def close(self):
        pass

    def learn(self):
        state, _, _, action, next_state, reward, terminated, _ = list(self.trajectory)
        #v = reward + self.gamma * self.q[next_state].max() * (1 - terminated)
        #v = reward + self.gamma * self.q[next_state].max() * (1 - terminated)
        target = reward + self.gamma * self.q[next_state].max() * (1 - terminated)
        td_error = target - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error

def play_episode(env, agent, seed=None, mode=QLearningMode.TRAIN):
    observation, _ = env.reset(seed=seed)
    reward, terminated, truncated = 0, False, False
    agent.reset(mode=mode)
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

def train(env, agent, no_episodes=int(1e6)):
    episode_rewards = []
    episode_numbers = []
    episode_rewards_averaged = []
    progress = tqdm(range(no_episodes))
    for episode in progress:
        episode_reward, elapsed_steps = play_episode(env, agent, seed=episode)
        episode_rewards.append(episode_reward)

        mean_reward = np.mean(episode_rewards[-200:])
        progress.set_postfix({"mean_reward": mean_reward})
        if episode % 250 == 0:
            episode_numbers.append(episode)
            episode_rewards_averaged.append(mean_reward)

        if mean_reward > env.spec.reward_threshold:
            print("breaking")
            break

    plt.figure()
    plt.plot(episode_numbers, episode_rewards_averaged)
    plt.show()


if __name__ == "__main__":
    env = gym.make('Taxi-v3')
    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    train(env, agent)