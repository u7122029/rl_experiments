import numpy as np
from tqdm import tqdm
import gymnasium as gym
from enum import Enum
from collections import deque
import matplotlib.pyplot as plt


class DoubleQLearningMode(Enum):
    TRAIN = 0
    TEST = 1

class DoubleQLearningAgent:
    def __init__(self, no_states, no_actions):
        self.no_states = no_states
        self.no_actions = no_actions

        self.gamma = 0.99
        self.learning_rate = 0.1
        self.epsilon = 0.01
        self.qs = [np.zeros((self.no_states, self.no_actions)), 
                   np.zeros((self.no_states, self.no_actions))]

    def reset(self, mode: DoubleQLearningMode):
        self.mode = mode
        if self.mode == DoubleQLearningMode.TRAIN:
            self.trajectory = deque(maxlen=8)

    def step(self, observation, reward, terminated):
        # epsilon-greedy action selection
        if self.mode == DoubleQLearningMode.TRAIN and np.random.rand() < self.epsilon:
            action = np.random.randint(self.no_actions)
        else:
            action = (self.qs[0] + self.qs[1])[observation].argmax()

        if self.mode == DoubleQLearningMode.TRAIN:
            self.trajectory.extend([observation, reward, terminated, action])
            if len(self.trajectory) >= 8:
                self.learn()

        return action

    def close(self):
        pass

    def learn(self):
        state, _, _, action, next_state, reward, terminated, _ = list(self.trajectory)

        if np.random.choice([True, False]):
            self.qs = self.qs[::-1]
            
        a = self.qs[0][next_state].argmax()
        v = self.qs[1][next_state, a]
        target = reward + self.gamma * v * (1 - terminated)
        td_error = target - self.qs[0][state, action]
        self.qs[0][state, action] += self.learning_rate * td_error

def play_episode(env, agent: DoubleQLearningAgent, mode=DoubleQLearningMode.TRAIN):
    state, _ = env.reset()
    agent.reset(mode=mode)

    episode_reward, elapsed_steps = 0, 0
    reward, terminated = 0, 0
    done = False
    while not done:
        action = agent.step(state, reward, terminated)
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        elapsed_steps += 1
        done = terminated or truncated
        
    agent.close()
    return episode_reward, elapsed_steps


def train(env, agent, num_episodes, seed=None):
    episode_rewards, episode_rewards_averaged = [], []
    episode_numbers = []
    progress = tqdm(range(num_episodes))
    for episode in progress:
        episode_reward, elapsed_steps = play_episode(env, agent, DoubleQLearningMode.TRAIN)
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
    agent = DoubleQLearningAgent(env.observation_space.n, env.action_space.n)
    train(env, agent, int(3e5))
    print(agent.q.argmax(axis=1).reshape(4, -1))

