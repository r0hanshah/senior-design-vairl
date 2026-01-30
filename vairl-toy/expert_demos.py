import numpy as np

from environment import SimpleEnv

def generate_expert_trajectories(env: SimpleEnv, n_traj=50, horizon=30):
    trajectories = []

    for _ in range(n_traj):
        s = env.reset()
        traj = []

        for _ in range(horizon):
            direction = env.goal - s
            action = 0.5 * direction / (np.linalg.norm(direction) + 1e-6)
            traj.append(s.copy())
            s, done = env.step(action)
            if done:
                break

        trajectories.append(np.array(traj))

    return trajectories
