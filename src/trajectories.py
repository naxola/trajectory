import numpy as np

def linear_trajectory(start, end, num_points):
    x = np.linspace(start[0], end[0], num_points)
    y = np.linspace(start[1], end[1], num_points)
    return np.column_stack((x, y))

def curved_trajectory(start, end, num_points, amplitude):
    x = np.linspace(start[0], end[0], num_points)
    y = np.linspace(start[1], end[1], num_points) + amplitude * np.sin(np.linspace(0, np.pi, num_points))
    return np.column_stack((x, y))

def noisy_trajectory(trajectory, noise_level):
    traj = trajectory.copy()
    traj[:,1] += np.random.normal(0, noise_level, len(traj))
    return traj


def bezier_n(points, steps=100):
    points = np.array(points)
    n = len(points) - 1
    t = np.linspace(0, 1, steps)

    curve = np.zeros((steps, 2))

    for i, p in enumerate(points):
        binom = np.math.comb(n, i)
        curve += binom * ((1 - t)**(n - i))[:, None] * (t**i)[:, None] * p

    return curve