import numpy as np
import math

class OUNoise:
    """
    Ref: https://github.com/songrotek/DDPG/blob/master/ou_noise.py
    """
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


class GaussNoise:
    def __init__(self, action_dimension, mu, std):
        self.action_dimension = action_dimension
        self.mu = mu
        self.std = std

    def reset(self):
        pass

    def noise(self):
        return np.random.normal(self.mu, self.std, size=self.action_dimension)
