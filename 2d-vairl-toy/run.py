import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =============================
# Environment (2D)
# =============================
STATE_MIN = -10.0
STATE_MAX = 10.0
GOAL = np.array([5, 6])
T = 40
GAMMA = 0.99
MAX_STEP = 0.5


def step(s, a):
    s_next = s + a
    return np.clip(s_next, STATE_MIN, STATE_MAX)


# =============================
# Expert demonstrations
# =============================
def generate_expert_trajectories(n_trajs=30):
    trajectories = []
    for _ in range(n_trajs):
        s = np.random.uniform(-3, -2, size=2)
        traj = []
        for _ in range(T):
            direction = GOAL - s
            direction /= (np.linalg.norm(direction) + 1e-8)
            a = MAX_STEP * direction
            s_next = step(s, a)
            traj.append((s.copy(), a.copy(), s_next.copy()))
            s = s_next
        trajectories.append(traj)
    return trajectories


# =============================
# Policy (Generator)
# =============================
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
            nn.Tanh()
        )

    def forward(self, s):
        return MAX_STEP * self.net(s)


def rollout_policy(policy, n_trajs=20):
    trajectories = []
    for _ in range(n_trajs):
        s = np.random.uniform(-3, -2, size=2)
        traj = []
        for _ in range(T):
            s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            a = policy(s_t).squeeze(0).detach().numpy()
            s_next = step(s, a)
            traj.append((s.copy(), a.copy(), s_next.copy()))
            s = s_next
        trajectories.append(traj)
    return trajectories


# =============================
# AIRL-style Reward Model (2D)
# r(s,a,s') = f(s,a) + γ h(s') − h(s)
# =============================
class AIRLReward(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.h = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, s, a, s_next):
        sa = torch.cat([s, a], dim=-1)
        return self.f(sa) + GAMMA * self.h(s_next) - self.h(s)


# =============================
# Losses
# =============================
def reward_loss(reward_model, expert_trajs, policy_trajs):
    exp = torch.tensor(
        [np.concatenate([s, a, sp]) for traj in expert_trajs for s, a, sp in traj],
        dtype=torch.float32
    )
    pol = torch.tensor(
        [np.concatenate([s, a, sp]) for traj in policy_trajs for s, a, sp in traj],
        dtype=torch.float32
    )

    s_exp, a_exp, sp_exp = exp[:, :2], exp[:, 2:4], exp[:, 4:6]
    s_pol, a_pol, sp_pol = pol[:, :2], pol[:, 2:4], pol[:, 4:6]

    r_exp = reward_model(s_exp, a_exp, sp_exp)
    r_pol = reward_model(s_pol, a_pol, sp_pol)

    return -(r_exp.mean() - r_pol.mean())


def policy_loss(policy, reward_model, policy_trajs, entropy_coef=0.01):
    data = torch.tensor(
        [np.concatenate([s, a, sp]) for traj in policy_trajs for s, a, sp in traj],
        dtype=torch.float32
    )

    s = data[:, :2]
    a = policy(s)
    sp = torch.clamp(s + a, STATE_MIN, STATE_MAX)

    r = reward_model(s, a, sp)
    entropy = -(a ** 2).mean()

    return -(r.mean() + entropy_coef * entropy)


# =============================
# Visualization
# =============================
def plot_trajectories(policy, expert_trajs, iteration):
    plt.figure(figsize=(6, 6))

    for traj in expert_trajs:
        states = np.array([s for s, _, _ in traj])
        plt.plot(states[:, 0], states[:, 1], color="green", alpha=0.3)

    policy_trajs = rollout_policy(policy, n_trajs=10)
    for traj in policy_trajs:
        states = np.array([s for s, _, _ in traj])
        plt.plot(states[:, 0], states[:, 1], color="red", alpha=0.7)

    plt.scatter(*GOAL, c="blue", marker="*", s=120, label="Goal")
    plt.xlim(STATE_MIN, STATE_MAX)
    plt.ylim(STATE_MIN, STATE_MAX)
    plt.title(f"Iteration {iteration} | Green=Expert Red=Policy")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_reward_field(reward_model):
    xs = np.linspace(-1, 1, 40)
    ys = np.linspace(-1, 1, 40)

    R = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            s = torch.tensor([[x, y]], dtype=torch.float32)
            a = torch.zeros_like(s)
            sp = torch.clamp(s, STATE_MIN, STATE_MAX)
            R[j, i] = reward_model(s, a, sp).item()

    plt.imshow(R, extent=(-1, 1, -1, 1), origin="lower")
    plt.colorbar(label="Reward")
    plt.title("Learned AIRL Reward Field")
    plt.show()


# =============================
# Training Loop
# =============================
def train():
    policy = Policy()
    reward_model = AIRLReward()

    policy_opt = optim.Adam(policy.parameters(), lr=3e-4)
    reward_opt = optim.Adam(reward_model.parameters(), lr=1e-4)

    expert_trajs = generate_expert_trajectories()

    for it in range(601):
        policy_trajs = rollout_policy(policy)

        # Reward update
        if it % 4 == 0:
            reward_opt.zero_grad()
            r_loss = reward_loss(reward_model, expert_trajs, policy_trajs)
            r_loss.backward()
            reward_opt.step()

        # Policy update
        policy_opt.zero_grad()
        p_loss = policy_loss(policy, reward_model, policy_trajs)
        p_loss.backward()
        policy_opt.step()

        if it % 30 == 0:
            print(
                f"Iter {it:03d} | Reward loss {r_loss.item():+.3f} | "
                f"Policy loss {p_loss.item():+.3f}"
            )
            plot_trajectories(policy, expert_trajs, it)

    plot_reward_field(reward_model)


if __name__ == "__main__":
    train()

