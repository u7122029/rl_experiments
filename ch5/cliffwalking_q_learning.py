import numpy as np
import gymnasium as gym
from tqdm import tqdm
from enum import Enum
from collections import deque
from statistics import mean
import matplotlib.pyplot as plt


class QLearningMode(Enum):
    TRAIN = 0
    TEST = 1

class QLearningAgent:
    def __init__(self, no_states, no_actions):
        self.no_states = no_states
        self.no_actions = no_actions

        self.epsilon = 0.01
        self.gamma = 0.99
        self.learning_rate = 0.1
        self.q = np.zeros((self.no_states, self.no_actions))
    
    def close(self):
        pass
    
    def reset(self, mode):
        self.mode = mode
        if self.mode == QLearningMode.TRAIN:
            self.trajectory = deque(maxlen=8)

    def step(self, observation, reward, terminated):
        if self.mode == QLearningMode.TRAIN and np.random.uniform() < self.epsilon:
            action = np.random.choice(self.no_actions)
        else:
            action = self.q[observation].argmax()
        
        if self.mode == QLearningMode.TRAIN:
            self.trajectory.extend([observation, reward, terminated, action])
            if len(self.trajectory) >= 8:
                self.learn()
        return action
    
    def learn(self):
        # Draw from trajectory
        state, _, _, action, next_state, reward, terminated, _ = list(self.trajectory)

        # Compute U_t
        target = reward + self.gamma * (1 - terminated) * self.q[next_state].max()

        # Compute TD error
        td_error = target - self.q[state, action]

        # Update Q
        self.q[state, action] += self.learning_rate * td_error


def play_episode(env, agent, seed=None, mode=QLearningMode.TRAIN):
    episode_reward, elapsed_steps = 0, 0
    state, _ = env.reset(seed=seed)
    agent.reset(mode)

    reward, terminated = 0, False
    done = False
    while not done:
        action = agent.step(state, reward, terminated)
        state, reward, terminated, truncated, _ = env.step(action)
        elapsed_steps += 1
        episode_reward += reward
        done = terminated or truncated
    
    agent.close()
    return episode_reward, elapsed_steps

def train(env, agent, no_episodes):
    episode_rewards = deque(maxlen=200)
    episode_rewards_averaged = []

    progress = tqdm(range(no_episodes))
    for episode in progress:
        episode_reward, elapsed_steps = play_episode(env, agent, episode)
        episode_rewards.append(episode_reward)

        mean_reward = mean(episode_rewards)
        progress.set_postfix({"mean_reward": mean_reward})
        episode_rewards_averaged.append(mean_reward)
    
    plt.figure()
    plt.plot(episode_rewards_averaged)
    plt.show()

if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")
    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    train(env, agent, int(0.5e4))
    print(agent.q.argmax(axis=1).reshape(4, -1))
    

