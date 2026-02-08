import matplotlib.pyplot as plt

def plot_expert_vs_generator(expert_trajs, generator_trajs, goal=None, wall=None, max_trajs=10):
    """
    Overlay expert and generator trajectories in 2D.
    """

    plt.figure(figsize=(6, 6))

    # Plot expert trajectories
    for traj in expert_trajs[:max_trajs]:
        xs = [step[0][0] for step in traj]
        ys = [step[0][1] for step in traj]
        xs.append(traj[-1][2][0])
        ys.append(traj[-1][2][1])
        plt.plot(xs, ys, 'g-', alpha=0.6, label='Expert' if 'Expert' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Plot generator trajectories
    for traj in generator_trajs[:max_trajs]:
        xs = [step[0][0] for step in traj]
        ys = [step[0][1] for step in traj]
        xs.append(traj[-1][2][0])
        ys.append(traj[-1][2][1])
        plt.plot(xs, ys, 'r--', alpha=0.6, label='Generator' if 'Generator' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    # Plot wall if provided
    if wall:
        plt.plot([wall[0], wall[0]], [wall[1], wall[2]], 'b-', alpha=0.6, label='Wall')

    # Plot goal if provided
    if goal is not None:
        plt.scatter(goal[0], goal[1], c='blue', s=100, marker='*', label='Goal')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Expert vs Generator Trajectories")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

