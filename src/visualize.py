import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(trajectory, title="Trajectory"):
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.scatter(trajectory[0, 0], trajectory[0, 1], color="green", label="Start")
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color="red", label="End")
    plt.axis("equal")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_three_trajectories(trajectory1, trajectory2, trajectory3, title="Trajectories"):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].plot(trajectory1[:,0], trajectory1[:,1])
    axs[0].scatter(trajectory1[:,0], trajectory1[:,1], s=5)
    axs[0].set_title("Linear")
    axs[0].axis("equal")

    axs[1].plot(trajectory2[:,0], trajectory2[:,1])
    axs[1].scatter(trajectory2[:,0], trajectory2[:,1], s=5)
    axs[1].set_title("Curved")
    axs[1].axis("equal")

    axs[2].plot(trajectory3[:,0], trajectory3[:,1])
    axs[2].scatter(trajectory3[:,0], trajectory3[:,1], s=5)
    axs[2].set_title("Noisy")
    axs[2].axis("equal")

    plt.tight_layout()
    plt.show()

def plot_trajectories_grid(trajectories, titles=None, n_rows=1, n_cols=1):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

    # Si solo hay un subplot, lo convertimos en array para iterar igual
    axs = np.array(axs).reshape(-1)

    for i in range(n_rows * n_cols):
        ax = axs[i]

        if i < len(trajectories):
            traj = trajectories[i]
            ax.plot(traj[:,0], traj[:,1])
            ax.scatter(traj[:,0], traj[:,1], s=5)
            ax.axis("equal")

            if titles and i < len(titles):
                ax.set_title(titles[i])
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()