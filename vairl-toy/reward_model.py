"""
This is where the VA in VAIRL comes from.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class RewardModel(nn.Module): # Essentially the discriminator
    def __init__(self, latent_dim=4):
        
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2) # mean + logvar
        )

        self.reward_net = nn.Sequential(
            nn.Linear(2 + latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def encode(self, s):
        params = self.encoder(s)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar

    def sample_z(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward(self, s):
        mu, logvar = self.encode(s)
        z = self.sample_z(mu, logvar)
        r = self.reward_net(torch.cat([s, z], dim=-1))
        return r, mu, logvar

def reward_loss(reward_model: RewardModel, expert_states, policy_states, beta=0.01):
    expert_r, mu_e, logvar_e = reward_model(expert_states) # r, mu, logvar
    policy_r, _, _ = reward_model(policy_states)

    # Adversarial loss
    loss_disc = -(
        F.binary_cross_entropy_with_logits(expert_r, torch.ones_like(expert_r)) -
        F.binary_cross_entropy_with_logits(policy_r, torch.zeros_like(policy_r))
    )

    # KL regularization
    kl = -0.5 * torch.mean(
        1 + logvar_e - mu_e.pow(2) - logvar_e.exp()
    )

    return loss_disc + beta * kl

def policy_loss(policy, reward_model, states): # Later replace with PPO
    actions = policy(states)
    rewards, _, _ = reward_model(states)
    return -torch.mean(rewards)
