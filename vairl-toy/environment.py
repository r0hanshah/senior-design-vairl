import numpy as np

class SimpleEnv:
    def __init__(self, goal=np.array([5.0, 5.0])):
        self.goal = goal
        self.reset()

    def reset(self):
        self.state = np.random.uniform(-1, 1, size=2)
        return self.state

    def step(self, action):
        self.state = self.state + action
        done = np.linalg.norm(self.state - self.goal) < 0.5
        return self.state, done
