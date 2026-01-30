import torch
import torch.nn as nn
import numpy as np

from environment import SimpleEnv
from policy import Policy

def rollout_policy(env: SimpleEnv, policy: Policy, n_traj=20, horizon=30):
    trajectories = []

    for _ in range(n_traj):
        s = env.reset()
        traj = []

        for _ in  range(horizon):
            with torch.no_grad():
                a = policy(torch.tensor(s, dtype=torch.float32)).numpy()
            traj.append(s.copy())
            s, done = env.step(a)
            if done:
                break

        trajectories.append(np.array(traj))

    return trajectories
