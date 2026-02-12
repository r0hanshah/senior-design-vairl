"""
VAIRL built on the imitation AIRL trainer, adapted to the 2D grid world.
Keeps the same terminal outputs/metrics as vairl2.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    import gymnasium as gym
    _GYMNASIUM = True
except ImportError:  # fallback for older gym
    import gym
    _GYMNASIUM = False

try:
    from imitation.algorithms.adversarial.airl import AIRL
    from imitation.data import rollout
    from imitation.data import types as imitation_types
    from imitation.rewards.reward_nets import RewardNet
    from imitation.util.networks import RunningNorm
    from imitation.util.util import make_vec_env
except ImportError as e:  # pragma: no cover - explicit runtime error for missing dep
    raise ImportError(
        "imitation is required for vairl3.py. Install it (e.g., `pip install imitation`)."
    ) from e

from sb3_contrib import TRPO

from metrics import MetricsLogger
from display import plot_expert_vs_generator


# Environment
GRID_MIN = -10
GRID_MAX = 10

RANDOM_SPAWN_MIN_X = -3
RANDOM_SPAWN_MIN_Y = -3

RANDOM_SPAWN_MAX_X = -2
RANDOM_SPAWN_MAX_Y = -2

GOAL = np.array([5, 5], dtype=np.float32)

# WALL
WALL_X = 2.5
WALL_Y_MIN = -10
WALL_Y_MAX = 6.0

# Movement
MAX_STEP_COUNT = 40
MAX_STEP_SIZE = 0.5

# Training
NUM_EPISODES = 500
ROLLOUT_STEPS = 2048

GENERATOR_LEARNING_RATE = 1e-4
DISCRIMINATOR_LEARNING_RATE = 1e-4

# VDB/VAIRL
Z_SIZE = 6
E_HIDDEN_SIZE = 128
I_C = 0.5
BETA_STEP_SIZE = 1e-3
GAMMA = 0.99

# How often to save metrics snapshots
SAVE_EVERY = 10

SEED = 42
ENV_ID = "Vairl2D-v0"
EXPERT_NOISE_STD = 1e-4


def hits_wall(state, next_state):
    if (state[0] < WALL_X <= next_state[0]) or (next_state[0] < WALL_X <= state[0]):
        t = (WALL_X - state[0]) / (next_state[0] - state[0] + 1e-8)
        y_cross = state[1] + t * (next_state[1] - state[1])

        if WALL_Y_MIN <= y_cross <= WALL_Y_MAX:
            return True

    return False


class Vairl2DEnv(gym.Env):
    """
    Simple 2D grid environment for imitation.
    """

    def __init__(self, enable_wall: bool = False):
        super().__init__()

        self.enable_wall = enable_wall

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.state = None
        self.t = 0
        self.rng = np.random.default_rng(SEED)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.state = np.array(
            [
                self.rng.uniform(RANDOM_SPAWN_MIN_X, RANDOM_SPAWN_MAX_X),
                self.rng.uniform(RANDOM_SPAWN_MIN_Y, RANDOM_SPAWN_MAX_Y),
            ],
            dtype=np.float32,
        )
        self.t = 0

        if _GYMNASIUM:
            return self.state.copy(), {}
        return self.state.copy()

    def step(self, action):
        self.t += 1

        action = np.asarray(action, dtype=np.float32)
        next_state = self.state + action

        env_reward = 0.0
        if self.enable_wall and hits_wall(self.state, next_state):
            next_state = self.state.copy()
            env_reward = -1.0

        self.state = next_state
        dist = np.linalg.norm(next_state - GOAL)
        terminated = dist < 0.5
        truncated = self.t >= MAX_STEP_COUNT

        if (terminated or truncated) and dist <= 0.5:
            env_reward += 1.0

        if _GYMNASIUM:
            return next_state.copy(), float(env_reward), bool(terminated), bool(truncated), {}
        done = bool(terminated or truncated)
        return next_state.copy(), float(env_reward), done, {}


def register_env() -> None:
    try:
        if _GYMNASIUM:
            from gymnasium.envs.registration import register
        else:
            from gym.envs.registration import register

        register(
            id=ENV_ID,
            entry_point=Vairl2DEnv,
            max_episode_steps=MAX_STEP_COUNT,
        )
    except Exception:
        # Already registered or gym registry unavailable; ignore
        pass


def step_state(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.clip(state + action, GRID_MIN, GRID_MAX)


def _maybe_add_noise(action: np.ndarray, noise_std: float) -> np.ndarray:
    if noise_std <= 0:
        return action
    return action + np.random.normal(scale=noise_std, size=action.shape)


def generate_s_shaped_expert_trajectories(
    n_steps: int = MAX_STEP_COUNT,
    amplitude: float = 1.0,
    frequency: float = 2.0,
    n_trajs: int = 30,
    noise_std: float = EXPERT_NOISE_STD,
):
    trajectories = []

    for _ in range(n_trajs):
        state = np.array(
            [
                np.random.uniform(RANDOM_SPAWN_MIN_X, RANDOM_SPAWN_MAX_X),
                np.random.uniform(RANDOM_SPAWN_MIN_Y, RANDOM_SPAWN_MAX_Y),
            ]
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

            action = MAX_STEP_SIZE * move_dir
            action = _maybe_add_noise(action, noise_std)
            next_state = step_state(state, action)

            traj.append((state.copy(), action.copy(), next_state.copy()))
            state = next_state

        trajectories.append(traj)

    return trajectories


def generate_curved_expert_trajectories(
    n_steps: int = MAX_STEP_COUNT,
    curvature: float = 10.0,
    n_trajs: int = 30,
    noise_std: float = EXPERT_NOISE_STD,
):
    trajectories = []

    for _ in range(n_trajs):
        state = np.array(
            [
                np.random.uniform(RANDOM_SPAWN_MIN_X, RANDOM_SPAWN_MAX_X),
                np.random.uniform(RANDOM_SPAWN_MIN_Y, RANDOM_SPAWN_MAX_Y),
            ]
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

            action = MAX_STEP_SIZE * move_dir
            action = _maybe_add_noise(action, noise_std)
            next_state = step_state(state, action)

            traj.append((state.copy(), action.copy(), next_state.copy()))
            state = next_state

        trajectories.append(traj)

    return trajectories


def generate_straight_expert_trajectories(
    n_trajs: int = 30,
    noise_std: float = EXPERT_NOISE_STD,
):
    trajectories = []

    for _ in range(n_trajs):
        state = np.array(
            [
                np.random.uniform(RANDOM_SPAWN_MIN_X, RANDOM_SPAWN_MAX_X),
                np.random.uniform(RANDOM_SPAWN_MIN_Y, RANDOM_SPAWN_MAX_Y),
            ]
        )
        traj = []

        for _ in range(MAX_STEP_COUNT):
            direction = GOAL - state
            direction /= (np.linalg.norm(direction) + 1e-8)

            action = MAX_STEP_SIZE * direction
            action = _maybe_add_noise(action, noise_std)
            next_state = step_state(state, action)

            traj.append((state.copy(), action.copy(), next_state.copy()))
            state = next_state

        trajectories.append(traj)

    return trajectories


def trajectories_to_imitation(trajectories: List[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]):
    trajs = []
    for traj in trajectories:
        obs = [step[0] for step in traj]
        obs.append(traj[-1][2])
        acts = [step[1] for step in traj]
        obs_arr = np.asarray(obs, dtype=np.float32)
        acts_arr = np.asarray(acts, dtype=np.float32)
        trajs.append(
            imitation_types.Trajectory(
                obs=obs_arr,
                acts=acts_arr,
                infos=None,
                terminal=True,
            )
        )
    return trajs


class Encoder(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, E_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(E_HIDDEN_SIZE, E_HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.mu = nn.Linear(E_HIDDEN_SIZE, Z_SIZE)
        self.logvar = nn.Linear(E_HIDDEN_SIZE, Z_SIZE)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), min=-10.0, max=10.0)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1)


class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z):
        return self.net(z)


class HNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z):
        return self.net(z)


class VAIRLRewardNet(RewardNet):
    """
    VAIRL reward network with variational encoders on g and h.
    f(s,a,s') = g(z_g(s,a)) + gamma*h(z_h(s')) - h(z_h(s))
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        normalize_input_layer=RunningNorm,
        gamma: float = GAMMA,
    ):
        try:
            super().__init__(observation_space, action_space)
        except TypeError:
            super().__init__()

        self.gamma = gamma
        self.obs_dim = int(np.prod(observation_space.shape))
        self.act_dim = int(np.prod(action_space.shape))

        self.obs_norm = normalize_input_layer(observation_space.shape) if normalize_input_layer else None
        self.act_norm = normalize_input_layer(action_space.shape) if normalize_input_layer else None

        self.enc_g = Encoder(self.obs_dim + self.act_dim)
        self.enc_h = Encoder(self.obs_dim)
        self.g = GNet()
        self.h = HNet()

    def _normalize(self, obs: torch.Tensor, acts: torch.Tensor):
        if self.obs_norm is not None:
            obs = self.obs_norm(obs)
        if self.act_norm is not None:
            acts = self.act_norm(acts)
        return obs, acts

    def forward_with_stats(
        self,
        obs: torch.Tensor,
        acts: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ):
        obs, acts = self._normalize(obs, acts)
        next_obs = self.obs_norm(next_obs) if self.obs_norm is not None else next_obs

        sa = torch.cat([obs, acts], dim=1)
        z_g, mu_g, logvar_g = self.enc_g(sa)
        z_h, mu_h, logvar_h = self.enc_h(obs)
        z_hn, mu_hn, logvar_hn = self.enc_h(next_obs)

        f = self.g(z_g) + self.gamma * self.h(z_hn) - self.h(z_h)
        stats = (mu_g, logvar_g, mu_h, logvar_h, mu_hn, logvar_hn)
        return f, stats

    def forward(self, obs, acts, next_obs, dones):
        f, _stats = self.forward_with_stats(obs, acts, next_obs, dones)
        return f.squeeze(-1)


def total_kl(stats: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    mu_g, logvar_g, mu_h, logvar_h, mu_hn, logvar_hn = stats
    return (
        kl_divergence(mu_g, logvar_g)
        + kl_divergence(mu_h, logvar_h)
        + kl_divergence(mu_hn, logvar_hn)
    ).mean()


class VAIRL(AIRL):
    """
    AIRL with a variational bottleneck on the reward network.
    """

    def __init__(
        self,
        *,
        demonstrations,
        demo_batch_size: int,
        venv,
        gen_algo,
        reward_net: VAIRLRewardNet,
        beta: float = 0.0,
        i_c: float = I_C,
        beta_step_size: float = BETA_STEP_SIZE,
        disc_lr: float = DISCRIMINATOR_LEARNING_RATE,
        **kwargs,
    ):
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            reward_net=reward_net,
            **kwargs,
        )

        self.vairl_reward_net = reward_net
        self.beta = float(beta)
        self.i_c = float(i_c)
        self.beta_step_size = float(beta_step_size)

        self.device = getattr(gen_algo, "device", torch.device("cpu"))
        self.vairl_reward_net.to(self.device)

        self._vairl_disc_opt = optim.Adam(self.vairl_reward_net.parameters(), lr=disc_lr)

    def train_disc(
        self,
        *,
        expert_samples: Dict[str, np.ndarray] | None = None,
        gen_samples: Dict[str, np.ndarray] | None = None,
    ) -> Dict[str, float]:
        if expert_samples is None or gen_samples is None:
            raise ValueError("VAIRL.train_disc requires explicit expert and gen samples")

        def to_torch(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device)

        obs_e = to_torch(expert_samples["obs"])
        acts_e = to_torch(expert_samples["acts"])
        next_obs_e = to_torch(expert_samples["next_obs"])
        dones_e = to_torch(expert_samples["dones"]).unsqueeze(1)

        obs_g = to_torch(gen_samples["obs"])
        acts_g = to_torch(gen_samples["acts"])
        next_obs_g = to_torch(gen_samples["next_obs"])
        dones_g = to_torch(gen_samples["dones"]).unsqueeze(1)

        f_e, stats_e = self.vairl_reward_net.forward_with_stats(obs_e, acts_e, next_obs_e, dones_e)
        f_g, stats_g = self.vairl_reward_net.forward_with_stats(obs_g, acts_g, next_obs_g, dones_g)

        with torch.no_grad():
            _, log_prob_e, _ = self.gen_algo.policy.evaluate_actions(obs_e, acts_e)
            _, log_prob_g, _ = self.gen_algo.policy.evaluate_actions(obs_g, acts_g)

        if log_prob_e.ndim == 1:
            log_prob_e = log_prob_e.unsqueeze(1)
        if log_prob_g.ndim == 1:
            log_prob_g = log_prob_g.unsqueeze(1)

        logits_e = f_e - log_prob_e
        logits_g = f_g - log_prob_g

        labels_e = torch.ones_like(logits_e)
        labels_g = torch.zeros_like(logits_g)

        loss_disc = F.binary_cross_entropy_with_logits(logits_e, labels_e)
        loss_disc = loss_disc + F.binary_cross_entropy_with_logits(logits_g, labels_g)

        kl = 0.5 * (total_kl(stats_e) + total_kl(stats_g))
        bottleneck = kl - self.i_c

        loss = loss_disc + self.beta * bottleneck

        self._vairl_disc_opt.zero_grad()
        loss.backward()
        self._vairl_disc_opt.step()

        self.beta = max(0.0, self.beta + self.beta_step_size * float(bottleneck.item()))

        return {
            "loss": float(loss.item()),
            "kl": float(kl.item()),
            "beta": float(self.beta),
        }


def trajectories_to_batch(trajs: List[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]):
    obs_list = []
    acts_list = []
    next_obs_list = []
    dones_list = []

    for traj in trajs:
        for i, (s, a, s_next) in enumerate(traj):
            obs_list.append(s)
            acts_list.append(a)
            next_obs_list.append(s_next)
            dones_list.append(1.0 if i == len(traj) - 1 else 0.0)

    return {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "acts": np.asarray(acts_list, dtype=np.float32),
        "next_obs": np.asarray(next_obs_list, dtype=np.float32),
        "dones": np.asarray(dones_list, dtype=np.float32),
    }


def generate_generator_trajectories(
    generator: TRPO,
    reward_net: VAIRLRewardNet,
    env: Vairl2DEnv,
    n_trajs: int = 20,
):
    trajectories = []
    rewards = []

    for _ in range(n_trajs):
        if _GYMNASIUM:
            state, _ = env.reset()
        else:
            state = env.reset()

        traj = []
        reward_accumulated = 0.0

        for _ in range(MAX_STEP_COUNT):
            action, _ = generator.predict(state, deterministic=False)
            action = np.asarray(action, dtype=np.float32)

            if _GYMNASIUM:
                next_state, _env_reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            else:
                next_state, _env_reward, done, _ = env.step(action)

            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                a_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
                s_next_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                d_t = torch.tensor([float(done)], dtype=torch.float32).unsqueeze(1)
                reward = reward_net(s_t, a_t, s_next_t, d_t).item()

            dist = np.linalg.norm(next_state - GOAL)
            if done and dist <= 0.5:
                reward += 1.0

            reward_accumulated += reward
            traj.append((state.copy(), action.copy(), next_state.copy()))

            state = next_state

            if done:
                if _GYMNASIUM:
                    state, _ = env.reset()
                else:
                    state = env.reset()

        rewards.append(reward_accumulated)
        trajectories.append(traj)

    print(f"Reward mean  : {np.mean(rewards):.3f}")
    print(f"Reward std   : {np.std(rewards):.3f}")

    return trajectories, float(np.mean(rewards)), float(np.std(rewards))


def main():
    train_discriminator_flag = True

    path_type = input(
        "\nChoose expert path type:\n- s: S-Curve\n- c: C-Curve\n- else: Straight\n\nChoice: "
    )
    enable_wall = path_type == "c"

    out_dir = Path("metrics_out")
    metrics = MetricsLogger()

    register_env()

    if path_type == "s":
        exp_data = generate_s_shaped_expert_trajectories()
    elif path_type == "c":
        exp_data = generate_curved_expert_trajectories()
    else:
        exp_data = generate_straight_expert_trajectories()

    exp_batch = trajectories_to_batch(exp_data)
    expert_trajs = trajectories_to_imitation(exp_data)
    expert_transitions = rollout.flatten_trajectories(expert_trajs)
    demo_batch_size = int(min(2048, len(expert_transitions)))

    # Training env (will be reward-wrapped by AIRL)
    train_venv = make_vec_env(
        ENV_ID,
        rng=np.random.default_rng(SEED),
        n_envs=1,
        env_make_kwargs={"enable_wall": enable_wall},
    )

    generator = TRPO(
        policy="MlpPolicy",
        env=train_venv,
        learning_rate=GENERATOR_LEARNING_RATE,
        batch_size=2048,
        gamma=0.99,
        gae_lambda=0.95,
        target_kl=0.01,
        verbose=True,
        seed=SEED,
    )

    reward_net = VAIRLRewardNet(
        observation_space=train_venv.observation_space,
        action_space=train_venv.action_space,
        normalize_input_layer=RunningNorm,
        gamma=GAMMA,
    )

    vairl = VAIRL(
        demonstrations=expert_transitions,
        demo_batch_size=demo_batch_size,
        venv=train_venv,
        gen_algo=generator,
        reward_net=reward_net,
        beta=0.0,
        i_c=I_C,
        beta_step_size=BETA_STEP_SIZE,
        disc_lr=DISCRIMINATOR_LEARNING_RATE,
        allow_variable_horizon=True,
    )

    eval_env = Vairl2DEnv(enable_wall=enable_wall)

    for episode in range(NUM_EPISODES):
        print(f"EPISODE {episode}")

        reward_net.eval()
        gen_data, reward_mean, reward_std = generate_generator_trajectories(
            generator, reward_net, eval_env
        )

        gen_batch = trajectories_to_batch(gen_data)

        # TRAIN GENERATOR (use AIRL wrapper to manage buffering)
        vairl.train_gen(total_timesteps=ROLLOUT_STEPS)

        logger = generator.logger.name_to_value

        # Keep exact naming / print format
        for k in [
            "train/kl_divergence_loss",
            "train/is_line_search_success",
            "train/std",
            "train/policy_objective",
            "train/explained_variance",
        ]:
            print(k, logger.get(k))

        # TRAIN DISCRIMINATOR
        reward_net.train()
        if train_discriminator_flag:
            disc_stats = vairl.train_disc(
                expert_samples=exp_batch,
                gen_samples=gen_batch,
            )
        else:
            disc_stats = {"loss": 0.0, "kl": 0.0, "beta": float(vairl.beta)}

        loss = disc_stats["loss"]
        kl = disc_stats["kl"]
        beta = disc_stats["beta"]

        print(f"D loss       : {loss:.4f}")
        print(f"kl           : {kl:.3f}")
        print(f"beta         : {beta:.3f}")

        # ---- Metrics capture (no output changes) ----
        def _safe_log(key: str) -> float:
            val = logger.get(key)
            return float(val) if val is not None else 0.0

        metrics.log("train/kl_divergence_loss", _safe_log("train/kl_divergence_loss"))
        metrics.log("train/is_line_search_success", _safe_log("train/is_line_search_success"))
        metrics.log("train/std", _safe_log("train/std"))
        metrics.log("train/policy_objective", _safe_log("train/policy_objective"))
        metrics.log("train/explained_variance", _safe_log("train/explained_variance"))

        metrics.log("Reward mean", reward_mean)
        metrics.log("Reward std", reward_std)

        metrics.log("D loss", loss)
        metrics.log("KL(z|x)", kl)
        metrics.log("KL - I_c", (kl - I_C))
        metrics.log("beta", float(beta))

        if (episode + 1) % SAVE_EVERY == 0:
            plot_expert_vs_generator(
                exp_data,
                gen_data,
                GOAL,
                [WALL_X, WALL_Y_MIN, WALL_Y_MAX] if enable_wall else None,
            )
            metrics.save_all(out_dir, show=False)

    metrics.save_all(out_dir, show=True)


if __name__ == "__main__":
    main()
