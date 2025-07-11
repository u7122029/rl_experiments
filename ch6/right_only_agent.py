import gymnasium as gym
import numpy as np

class RightOnlyAgent:
    def __init__(self):
        pass

    def reset(self, train=True):
        self.train = train
    
    def step(self, observation, reward, terminated):
        action = 2
        return action
    
    def close(self):
        pass

    def learn(self):
        pass
    