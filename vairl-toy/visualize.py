import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def plot_trajectories(expert_trajs, policy_trajs):
    plt.figure(figsize=(6, 6))

    for traj in expert_trajs:
        traj = np.array(traj)
        traj = np.array(traj)
        plt.scatter(traj[0][0], traj[0][1], color='green') # start
        plt.scatter(traj[-1][0], traj[-1][1], color='red') # end
        plt.plot(traj[:, 0], traj[:, 1], 'g--', alpha=0.5)

    for i, traj in enumerate(policy_trajs):
        traj = np.array(traj)
        plt.scatter(traj[0][0], traj[0][1], color='green') # start
        plt.scatter(traj[-1][0], traj[-1][1], color='red') # end
        plt.plot(traj[:, 0], traj[:, 1], color=(i/len(policy_trajs), 0, 0))

    plt.scatter(0, 0, c='blue', s=100, label="Goal")
    plt.legend(["Expert", "Policy", "Goal"])
    plt.axis("equal")
    plt.title("Expert vs Learned Trajectories")
    plt.show()

def plot_reward(reward_model, device="cpu"):
    xs = np.linspace(-5, 5, 100)
    ys = np.linspace(-5, 5, 100)

    R = np.zeros((len(xs), len(ys)))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            s = torch.tensor([[x, y]], dtype=torch.float32).to(device)
            with torch.no_grad():
                r = reward_model(s)[0].item()
            R[j, i] = r

    plt.figure(figsize=(6, 5))
    plt.imshow(R, extent=[-5, 5, -5, 5], origin="lower")
    plt.colorbar(label="Reward")
    plt.scatter(0, 0, c='red', s=100)
    plt.title("Learned Reward Landscape")
    plt.show()

def plot_latent(reward_model, trajectories):
    zs = []

    for traj in trajectories:
        for s in traj:
            s = torch.tensor([s], dtype=torch.float32)
            with torch.no_grad():
                z, _, _ = reward_model.forward(s)
            zs.append(z.cpu().numpy())

    zs = np.array(zs)

    plt.figure(figsize=(5, 5))
    plt.scatter(zs[:, 0], zs[:, 1], alpha=0.5)
    plt.title("Latent Variable Distribution")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.show()


def animate_traj(traj):
    traj = np.array(traj)

    fig, ax = plt.subplots()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    point, = ax.plot([], [], 'ro')

    def update(i):
        point.set_data(traj[i, 0], traj[i, 1])
        return point,

    ani = animation.FuncAnimation(fig, update, frames=len(traj))
    plt.show()


