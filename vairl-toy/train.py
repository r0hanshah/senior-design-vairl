import torch
import numpy as np

import visualize
from environment import SimpleEnv
from expert_demos import generate_expert_trajectories
from policy import Policy
from reward_model import RewardModel, reward_loss, policy_loss
from roll_out_policy import rollout_policy

env = SimpleEnv()
expert_trajs = generate_expert_trajectories(env)

policy = Policy()
reward_model = RewardModel()

policy_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
reward_opt = torch.optim.Adam(reward_model.parameters(), lr=1e-4)

for iteration in range(200):
    # 1. Rollout policy
    policy_trajs = rollout_policy(env, policy)

    # 2. Sample states
    expert_states = torch.tensor(
        np.concatenate(expert_trajs), dtype=torch.float32
    )
    policy_states = torch.tensor(
        np.concatenate(policy_trajs), dtype=torch.float32
    )

    # 3. Update reward model
    reward_opt.zero_grad()
    r_loss = reward_loss(reward_model, expert_states, policy_states)
    r_loss.backward()
    reward_opt.step()

    # 4. Update policy
    policy_opt.zero_grad()
    p_loss = policy_loss(policy, reward_model, policy_states)
    p_loss.backward()
    policy_opt.step()

    if iteration % 20 == 0:
        print(f"Iter {iteration}: Reward loss {r_loss.item():.3f}")

visualize.plot_trajectories(expert_trajs, policy_trajs)
visualize.plot_reward(reward_model)
