from collections import deque

import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from enum import Enum

class SARSALMode(Enum):
    TRAIN = 0
    TEST = 1

class SARSALAgent:
    def __init__(self, no_states, no_actions):
        self.no_states = no_states
        self.no_actions = no_actions

        self.epsilon = 0.01
        self.lambd = 0.6
        self.gamma = 0.99
        self.beta = 1
        self.learning_rate = 0.02
        self.q = np.zeros((self.no_states, self.no_actions))
    
    def reset(self, mode):
        self.mode = mode
        if mode == SARSALMode.TRAIN:
            self.trajectory = deque(maxlen=8)
            self.e = np.zeros(self.q.shape)
        
    def step(self, observation, reward, terminated):
        """Select action."""
        if self.mode == SARSALMode.TRAIN and np.random.uniform() < self.epsilon:
            action = np.random.randint(self.no_actions)
        else:
            action = self.q[observation].argmax()
        
        if self.mode == SARSALMode.TRAIN:
            self.trajectory.extend([observation, reward, terminated, action])
            if len(self.trajectory) >= 8:
                self.learn()

        return action

    def close(self):
        pass

    def learn(self):
        state, _, _, action, next_state, reward, terminated, next_action = list(self.trajectory)

        self.e *= (self.lambd * self.gamma)
        self.e[state, action] = 1 + self.beta * self.e[state, action]

        target = reward + self.gamma * self.q[next_state, next_action] * (1 - terminated)

        td_error = target - self.q[state, action]
        self.q += self.learning_rate * self.e * td_error


def play_episode(env, agent, seed=None, mode=SARSALMode.TRAIN):
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


def train(env, agent, no_episodes):
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
    env = gym.make("Taxi-v3")
    agent = SARSALAgent(env.observation_space.n, env.action_space.n)
    train(env, agent, int(1e5))

