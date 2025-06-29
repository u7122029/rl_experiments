import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
from collections import deque
from enum import Enum

class SARSAMode(Enum):
    TRAIN = 0
    TEST = 1

class SARSAAgent:
    def __init__(self, env):
        self.gamma = 0.9
        self.learning_rate = 0.02
        self.epsilon = 0.01
        self.state_n = env.observation_space.n
        self.action_n = env.action_space.n
        self.q = np.zeros((self.state_n, self.action_n))

    def reset(self, mode: SARSAMode):
        self.mode = mode
        if self.mode == SARSAMode.TRAIN:
            self.trajectory = deque(maxlen=8)

    def step(self, observation, reward, terminated):
        """
        Choose an action according to the current policy during an episode.
        :param observation:
        :param reward:
        :param terminated:
        :return:
        """
        if self.mode == SARSAMode.TRAIN and np.random.uniform() < self.epsilon:
            action = np.random.randint(self.action_n)
        else:
            action = self.q[observation].argmax()

        if self.mode == SARSAMode.TRAIN:
            self.trajectory.extend([observation, reward, terminated, action])
            if len(self.trajectory) >= 8:
                self.learn()
        return action

    def close(self):
        pass

    def learn(self):
        state, _, _, action, next_state, reward, terminated, next_action = list(self.trajectory)
        target = reward + self.gamma * (1 - terminated) * self.q[next_state, next_action]
        td_error = target - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error

def play_episode(env, agent, seed=None, mode=None):
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


def train(env, agent, no_episodes, seed=None):
    episode_rewards = []
    progress = tqdm(range(no_episodes))
    for episode in progress:
        episode_reward, elapsed_steps = play_episode(env, agent, seed=seed, mode=SARSAMode.TRAIN)
        episode_rewards.append(episode_reward)

        mean_reward = np.mean(episode_rewards[-200:])
        progress.set_postfix({"mean_reward": mean_reward})
        if mean_reward > env.spec.reward_threshold:
            print("breaking")
            break

    plt.figure()
    plt.plot(episode_rewards)
    plt.show()

if __name__ == "__main__":
    env = gym.make('Taxi-v3')
    agent = SARSAAgent(env)
    train(env, agent, 100000)


