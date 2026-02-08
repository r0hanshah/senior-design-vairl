"""
Single-file VAIL (Variational Adversarial Imitation Learning) on a simple 2D grid.
Refactored to mirror the mujoco/vail implementation in this repo.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from metrics import MetricsLogger
from display import plot_expert_vs_generator

try:
    import gym
except ImportError:  # keep running without gym; env is simple
    gym = None


# =====================
# Environment settings
# =====================
GRID_MIN = -10.0
GRID_MAX = 10.0

RANDOM_SPAWN_MIN_X = -3.0
RANDOM_SPAWN_MIN_Y = -3.0

RANDOM_SPAWN_MAX_X = -2.0
RANDOM_SPAWN_MAX_Y = -2.0

GOAL = np.array([5.0, 5.0], dtype=np.float32)

WALL_X = 2.5
WALL_Y_MIN = -10.0
WALL_Y_MAX = 6.0

MAX_STEP_COUNT = 40
MAX_STEP_SIZE = 0.5


# =====================
# Utilities
# =====================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def hits_wall(state: np.ndarray, next_state: np.ndarray) -> bool:
    if (state[0] < WALL_X <= next_state[0]) or (next_state[0] < WALL_X <= state[0]):
        t = (WALL_X - state[0]) / (next_state[0] - state[0] + 1e-8)
        y_cross = state[1] + t * (next_state[1] - state[1])
        if WALL_Y_MIN <= y_cross <= WALL_Y_MAX:
            return True
    return False


def step_state(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.clip(state + action * MAX_STEP_SIZE, GRID_MIN, GRID_MAX)


# =====================
# Environment
# =====================


class VAIL2DEnv(gym.Env if gym is not None else object):
    """Simple 2D point-mass environment used for imitation."""

    def __init__(self, enable_wall: bool = False, seed: int | None = None):
        self.enable_wall = enable_wall
        self.state: np.ndarray | None = None
        self.t = 0
        self.rng = np.random.RandomState(seed)

        if gym is not None:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            )

    def seed(self, seed: int) -> None:
        self.rng = np.random.RandomState(seed)

    def reset(self) -> np.ndarray:
        self.state = np.array(
            [
                self.rng.uniform(RANDOM_SPAWN_MIN_X, RANDOM_SPAWN_MAX_X),
                self.rng.uniform(RANDOM_SPAWN_MIN_Y, RANDOM_SPAWN_MAX_Y),
            ],
            dtype=np.float32,
        )
        self.t = 0
        return self.state.copy()

    def step(self, action: np.ndarray):
        self.t += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        next_state = self.state + action * MAX_STEP_SIZE

        env_reward = 0.0
        if self.enable_wall and hits_wall(self.state, next_state):
            next_state = self.state.copy()
            env_reward = -1.0

        self.state = next_state
        dist = np.linalg.norm(next_state - GOAL)
        done = dist < 0.5 or self.t >= MAX_STEP_COUNT
        if done and dist < 0.5:
            env_reward += 1.0

        return next_state.copy(), float(env_reward), bool(done), {}


# =====================
# Expert trajectory generation
# =====================


def generate_s_shaped_expert_trajectories(
    n_steps: int = MAX_STEP_COUNT,
    amplitude: float = 1.0,
    frequency: float = 2.0,
    n_trajs: int = 30,
    rng: np.random.RandomState | None = None,
):
    rng = rng or np.random
    trajectories = []

    for _ in range(n_trajs):
        state = np.array(
            [
                rng.uniform(RANDOM_SPAWN_MIN_X, RANDOM_SPAWN_MAX_X),
                rng.uniform(RANDOM_SPAWN_MIN_Y, RANDOM_SPAWN_MAX_Y),
            ],
            dtype=np.float32,
        )

        traj = []

        direction = GOAL - state
        direction /= np.linalg.norm(direction) + 1e-8
        perp = np.array([-direction[1], direction[0]])

        for t in range(n_steps):
            alpha = t / n_steps
            offset = amplitude * np.sin(2 * np.pi * frequency * alpha)
            move_dir = direction + offset * perp
            move_dir /= np.linalg.norm(move_dir) + 1e-8

            action = move_dir
            next_state = step_state(state, action)

            traj.append((state.copy(), action.copy(), next_state.copy()))
            state = next_state

        trajectories.append(traj)

    return trajectories


def generate_curved_expert_trajectories(
    n_steps: int = MAX_STEP_COUNT,
    curvature: float = 10.0,
    n_trajs: int = 30,
    rng: np.random.RandomState | None = None,
):
    rng = rng or np.random
    trajectories = []

    for _ in range(n_trajs):
        state = np.array(
            [
                rng.uniform(RANDOM_SPAWN_MIN_X, RANDOM_SPAWN_MAX_X),
                rng.uniform(RANDOM_SPAWN_MIN_Y, RANDOM_SPAWN_MAX_Y),
            ],
            dtype=np.float32,
        )

        traj = []

        for t in range(n_steps):
            alpha = t / n_steps

            direction = GOAL - state
            perp = np.array([-direction[1], direction[0]])
            perp /= np.linalg.norm(perp) + 1e-8

            curve_offset = curvature * np.sin(np.pi * alpha)
            move_dir = direction + curve_offset * perp
            move_dir /= np.linalg.norm(move_dir) + 1e-8

            action = move_dir
            next_state = step_state(state, action)

            traj.append((state.copy(), action.copy(), next_state.copy()))
            state = next_state

        trajectories.append(traj)

    return trajectories


def generate_straight_expert_trajectories(
    n_trajs: int = 30,
    n_steps: int = MAX_STEP_COUNT,
    rng: np.random.RandomState | None = None,
):
    rng = rng or np.random
    trajectories = []

    for _ in range(n_trajs):
        state = np.array(
            [
                rng.uniform(RANDOM_SPAWN_MIN_X, RANDOM_SPAWN_MAX_X),
                rng.uniform(RANDOM_SPAWN_MIN_Y, RANDOM_SPAWN_MAX_Y),
            ],
            dtype=np.float32,
        )
        traj = []

        for _ in range(n_steps):
            direction = GOAL - state
            direction /= (np.linalg.norm(direction) + 1e-8)

            action = direction
            next_state = step_state(state, action)

            traj.append((state.copy(), action.copy(), next_state.copy()))
            state = next_state

        trajectories.append(traj)

    return trajectories


def flatten_expert_trajectories(trajectories: list) -> tuple[np.ndarray, np.ndarray]:
    states = []
    actions = []
    for traj in trajectories:
        for state, action, _next_state in traj:
            states.append(state)
            actions.append(action)
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


# =====================
# Running state normalization (from mujoco/vail)
# =====================


class RunningStat:
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape, dtype=np.float32)
        self._S = np.zeros(shape, dtype=np.float32)

    def push(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n

    @property
    def mean(self):
        return self._M

    @mean.setter
    def mean(self, M):
        self._M = M

    @property
    def sum_square(self):
        return self._S

    @sum_square.setter
    def sum_square(self, S):
        self._S = S

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)


class ZFilter:
    """y = (x-mean)/std using running estimates of mean,std"""

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip is not None:
            x = np.clip(x, -self.clip, self.clip)
        return x


def normalize_states(states: np.ndarray, running_state: ZFilter | None) -> np.ndarray:
    if running_state is None:
        return states
    return np.vstack([running_state(s, update=False) for s in states])


# =====================
# Models (from mujoco/vail)
# =====================


class Actor(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, args):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, num_outputs)

        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std


class Critic(nn.Module):
    def __init__(self, num_inputs: int, args):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)

        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.fc3(x)
        return v


class VDB(nn.Module):
    def __init__(self, num_inputs: int, args):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.z_size)
        self.fc3 = nn.Linear(args.hidden_size, args.z_size)
        self.fc4 = nn.Linear(args.z_size, args.hidden_size)
        self.fc5 = nn.Linear(args.hidden_size, 1)

        self.fc5.weight.data.mul_(0.1)
        self.fc5.bias.data.mul_(0.0)

    def encoder(self, x):
        h = torch.tanh(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def discriminator(self, z):
        h = torch.tanh(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        prob = self.discriminator(z)
        return prob, mu, logvar


# =====================
# VAIL utilities (from mujoco/vail)
# =====================


def get_action(mu, std):
    action = torch.normal(mu, std)
    return action.data.numpy()


def get_entropy(mu, std):
    dist = torch.distributions.Normal(mu, std)
    return dist.entropy().mean()


def log_prob_density(x, mu, std):
    log_prob = -(x - mu).pow(2) / (2 * std.pow(2)) - 0.5 * math.log(2 * math.pi)
    return log_prob.sum(1, keepdim=True)


def kl_divergence(mu, logvar):
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1)


def get_reward(vdb: VDB, state: np.ndarray, action: np.ndarray) -> float:
    state_action = np.concatenate([state, action], axis=0)
    state_action_t = torch.tensor(state_action, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prob, _mu, _logvar = vdb(state_action_t)
        prob = torch.clamp(prob, min=1e-8, max=1.0)
        return float(-math.log(prob.item()))


def save_checkpoint(state: dict, filename: str) -> None:
    torch.save(state, filename)


# =====================
# Training helpers (from mujoco/vail)
# =====================


def train_vdb(
    vdb: VDB,
    memory: deque,
    vdb_optim: optim.Optimizer,
    expert_states: np.ndarray,
    expert_actions: np.ndarray,
    beta: float,
    args,
    running_state: ZFilter | None = None,
):
    memory = np.array(memory, dtype=object)
    states = np.vstack(memory[:, 0])
    actions = np.vstack(memory[:, 1])

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)

    expert_states_norm = normalize_states(expert_states, running_state)
    demonstrations = np.concatenate([expert_states_norm, expert_actions], axis=1)
    demonstrations = torch.tensor(demonstrations, dtype=torch.float32)

    criterion = torch.nn.BCELoss()

    vdb_loss_val = 0.0

    for _ in range(args.vdb_update_num):
        learner, l_mu, l_logvar = vdb(torch.cat([states, actions], dim=1))
        expert, e_mu, e_logvar = vdb(demonstrations)

        l_kld = kl_divergence(l_mu, l_logvar).mean()
        e_kld = kl_divergence(e_mu, e_logvar).mean()

        kld = 0.5 * (l_kld + e_kld)
        bottleneck_loss = kld - args.i_c

        beta = max(0.0, beta + args.alpha_beta * bottleneck_loss.item())

        vdb_loss = (
            criterion(learner, torch.ones((states.shape[0], 1)))
            + criterion(expert, torch.zeros((demonstrations.shape[0], 1)))
            + beta * bottleneck_loss
        )

        vdb_optim.zero_grad()
        vdb_loss.backward(retain_graph=True)
        vdb_optim.step()

        vdb_loss_val = float(vdb_loss.item())

    with torch.no_grad():
        expert_acc = (vdb(demonstrations)[0] < 0.5).float().mean().item()
        learner_acc = (
            vdb(torch.cat([states, actions], dim=1))[0] > 0.5
        ).float().mean().item()

    return expert_acc, learner_acc, beta, vdb_loss_val


def get_gae(rewards, masks, values, args):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    masks = torch.tensor(masks, dtype=torch.float32)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0.0
    previous_value = 0.0
    running_advants = 0.0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + (args.gamma * running_returns * masks[t])
        returns[t] = running_returns

        running_delta = rewards[t] + (args.gamma * previous_value * masks[t]) - values.data[t]
        previous_value = values.data[t]

        running_advants = running_delta + (args.gamma * args.lamda * running_advants * masks[t])
        advants[t] = running_advants

    advants = (advants - advants.mean()) / (advants.std() + 1e-8)
    return returns, advants


def surrogate_loss(actor, advants, states, old_policy, actions, batch_index):
    mu, std = actor(states)
    new_policy = log_prob_density(actions, mu, std)
    old_policy = old_policy[batch_index]

    ratio = torch.exp(new_policy - old_policy)
    surr_loss = ratio * advants
    entropy = get_entropy(mu, std)

    return surr_loss, ratio, entropy


def train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args):
    memory = np.array(memory, dtype=object)
    states = np.vstack(memory[:, 0])
    actions = np.vstack(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])

    old_values = critic(torch.tensor(states, dtype=torch.float32))
    returns, advants = get_gae(rewards, masks, old_values, args)

    mu, std = actor(torch.tensor(states, dtype=torch.float32))
    old_policy = log_prob_density(torch.tensor(actions, dtype=torch.float32), mu, std).detach()

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    for _ in range(args.ppo_update_num):
        np.random.shuffle(arr)

        for i in range(n // args.batch_size):
            batch_index = arr[args.batch_size * i : args.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)

            inputs = torch.tensor(states, dtype=torch.float32)[batch_index]
            actions_samples = torch.tensor(actions, dtype=torch.float32)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            oldvalue_samples = old_values[batch_index].detach()

            # ----- Critic update -----
            values = critic(inputs)
            clipped_values = oldvalue_samples + torch.clamp(
                values - oldvalue_samples, -args.clip_param, args.clip_param
            )
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            # ----- Actor update -----
            loss, ratio, entropy = surrogate_loss(
                actor, advants_samples, inputs, old_policy, actions_samples, batch_index
            )
            clipped_ratio = torch.clamp(
                ratio, 1.0 - args.clip_param, 1.0 + args.clip_param
            )
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()
            total_actor_loss = actor_loss - 0.001 * entropy

            actor_optim.zero_grad()
            total_actor_loss.backward()
            actor_optim.step()


# =====================
# Rollouts & plotting
# =====================


def collect_samples(
    env: VAIL2DEnv,
    actor: Actor,
    vdb: VDB,
    total_sample_size: int,
    max_episode_steps: int,
    running_state: ZFilter | None = None,
):
    memory = deque()
    steps = 0
    scores = []
    irl_rewards = []

    while steps < total_sample_size:
        raw_state = env.reset()
        if running_state is not None:
            state = running_state(raw_state)
        else:
            state = raw_state.copy()

        score = 0.0

        for _ in range(max_episode_steps):
            mu, std = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            action = get_action(mu, std)[0]
            action = np.clip(action, -1.0, 1.0)

            next_raw_state, env_reward, done, _ = env.step(action)
            irl_reward = get_reward(vdb, state, action)

            mask = 0 if done else 1
            memory.append([state, action, irl_reward, mask])

            score += env_reward
            irl_rewards.append(irl_reward)

            steps += 1
            if done or steps >= total_sample_size:
                break

            if running_state is not None:
                state = running_state(next_raw_state)
            else:
                state = next_raw_state.copy()

        scores.append(score)

    return (
        memory,
        float(np.mean(scores)),
        float(np.mean(irl_rewards)) if irl_rewards else 0.0,
        float(np.std(irl_rewards)) if irl_rewards else 0.0,
    )


def generate_policy_trajectories(
    actor: Actor,
    env: VAIL2DEnv,
    n_trajs: int = 20,
    running_state: ZFilter | None = None,
):
    trajectories = []
    actor.eval()

    for _ in range(n_trajs):
        raw_state = env.reset()
        traj = []

        for _ in range(MAX_STEP_COUNT):
            if running_state is not None:
                state = running_state(raw_state, update=False)
            else:
                state = raw_state.copy()

            mu, std = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            action = get_action(mu, std)[0]
            action = np.clip(action, -1.0, 1.0)

            next_raw_state, _reward, done, _ = env.step(action)
            traj.append((raw_state.copy(), action.copy(), next_raw_state.copy()))
            raw_state = next_raw_state

            if done:
                break

        trajectories.append(traj)

    actor.train()
    return trajectories


# =====================
# Main
# =====================


def parse_args():
    parser = argparse.ArgumentParser(description="VAIL on 2D grid (single file)")
    parser.add_argument("--path_type", type=str, default="straight", choices=["s", "c", "straight"],
                        help="expert trajectory type: s, c, or straight")
    parser.add_argument("--n_expert_trajs", type=int, default=30,
                        help="number of expert trajectories")
    parser.add_argument("--normalize_state", action="store_true", default=True,
                        help="use running state normalization (default: True)")
    parser.add_argument("--no_normalize_state", action="store_false", dest="normalize_state",
                        help="disable running state normalization")

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.98)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--z_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--l2_rate", type=float, default=1e-3)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--alpha_beta", type=float, default=1e-4)
    parser.add_argument("--i_c", type=float, default=0.5)
    parser.add_argument("--vdb_update_num", type=int, default=3)
    parser.add_argument("--ppo_update_num", type=int, default=10)
    parser.add_argument("--total_sample_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--suspend_accu_exp", type=float, default=0.8)
    parser.add_argument("--suspend_accu_gen", type=float, default=0.8)
    parser.add_argument("--max_iter_num", type=int, default=200)
    parser.add_argument("--seed", type=int, default=500)

    parser.add_argument("--save_every", type=int, default=10,
                        help="save plots/metrics every N iterations")
    parser.add_argument("--save_model_every", type=int, default=100,
                        help="save checkpoint every N iterations")
    parser.add_argument("--load_model", type=str, default=None,
                        help="path to load a saved checkpoint")
    parser.add_argument("--metrics_dir", type=str, default="metrics_out")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    enable_wall = args.path_type == "c"
    env = VAIL2DEnv(enable_wall=enable_wall, seed=args.seed)

    state_dim = 2
    action_dim = 2

    running_state = ZFilter((state_dim,), clip=5.0) if args.normalize_state else None

    actor = Actor(state_dim, action_dim, args)
    critic = Critic(state_dim, args)
    vdb = VDB(state_dim + action_dim, args)

    actor_optim = optim.Adam(actor.parameters(), lr=1e-3)
    critic_optim = optim.Adam(critic.parameters(), lr=1e-4, weight_decay=args.l2_rate)
    vdb_optim = optim.Adam(vdb.parameters(), lr=1e-2)

    if args.load_model is not None:
        ckpt = torch.load(args.load_model)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        vdb.load_state_dict(ckpt["vdb"])

        if running_state is not None and "z_filter_n" in ckpt:
            running_state.rs.n = ckpt["z_filter_n"]
            running_state.rs.mean = ckpt["z_filter_m"]
            running_state.rs.sum_square = ckpt["z_filter_s"]

        print(f"Loaded checkpoint: {args.load_model}")

    rng = np.random.RandomState(args.seed)
    if args.path_type == "s":
        expert_trajs = generate_s_shaped_expert_trajectories(n_trajs=args.n_expert_trajs, rng=rng)
    elif args.path_type == "c":
        expert_trajs = generate_curved_expert_trajectories(n_trajs=args.n_expert_trajs, rng=rng)
    else:
        expert_trajs = generate_straight_expert_trajectories(n_trajs=args.n_expert_trajs, rng=rng)

    expert_states, expert_actions = flatten_expert_trajectories(expert_trajs)

    metrics = MetricsLogger()
    beta = 0.0
    train_discrim_flag = True

    for iteration in range(args.max_iter_num):
        actor.eval(), critic.eval(), vdb.eval()

        memory, score_avg, irl_mean, irl_std = collect_samples(
            env,
            actor,
            vdb,
            total_sample_size=args.total_sample_size,
            max_episode_steps=MAX_STEP_COUNT,
            running_state=running_state,
        )

        print(f"{iteration}:: episode score is {score_avg:.2f}")

        actor.train(), critic.train(), vdb.train()

        expert_acc = learner_acc = vdb_loss_val = None
        if train_discrim_flag:
            expert_acc, learner_acc, beta, vdb_loss_val = train_vdb(
                vdb,
                memory,
                vdb_optim,
                expert_states,
                expert_actions,
                beta,
                args,
                running_state=running_state,
            )
            print(f"Expert: {expert_acc * 100:.2f}% | Learner: {learner_acc * 100:.2f}%")
            if expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen:
                train_discrim_flag = False

        train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args)

        # ---- Metrics ----
        metrics.log("score", score_avg)
        metrics.log("irl_reward_mean", irl_mean)
        metrics.log("irl_reward_std", irl_std)
        metrics.log("beta", beta)
        metrics.log("expert_acc", expert_acc if expert_acc is not None else float("nan"))
        metrics.log("learner_acc", learner_acc if learner_acc is not None else float("nan"))
        metrics.log("vdb_loss", vdb_loss_val if vdb_loss_val is not None else float("nan"))

        # ---- Periodic plot & metrics snapshot ----
        if (iteration + 1) % args.save_every == 0:
            gen_trajs = generate_policy_trajectories(
                actor, env, n_trajs=20, running_state=running_state
            )
            plot_expert_vs_generator(
                expert_trajs,
                gen_trajs,
                GOAL,
                [WALL_X, WALL_Y_MIN, WALL_Y_MAX] if enable_wall else None,
            )
            metrics.save_all(Path(args.metrics_dir), show=False)

        # ---- Checkpoint ----
        if args.save_model_every > 0 and (iteration + 1) % args.save_model_every == 0:
            model_path = Path("save_model")
            model_path.mkdir(parents=True, exist_ok=True)
            ckpt_path = model_path / f"ckpt_iter_{iteration+1}.pth.tar"
            save_checkpoint(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "vdb": vdb.state_dict(),
                    "z_filter_n": running_state.rs.n if running_state is not None else 0,
                    "z_filter_m": running_state.rs.mean if running_state is not None else None,
                    "z_filter_s": running_state.rs.sum_square if running_state is not None else None,
                    "args": vars(args),
                    "score": score_avg,
                },
                filename=str(ckpt_path),
            )

    metrics.save_all(Path(args.metrics_dir), show=True)


if __name__ == "__main__":
    main()
