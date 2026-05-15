from src.trajectories import linear_trajectory, curved_trajectory, noisy_trajectory, bezier_n
from src.visualize import plot_trajectories_grid

start = (0,0)
end = (10, 5)

control_points = [
    (0, 0),
    (2, 3),
    (5, 1),
    (10, 5)
]


trajectory1 = linear_trajectory(start, end, 100)

trajectory2 = curved_trajectory(start, end, 100, 2)

trajectory3 = noisy_trajectory(trajectory2, 0.1)

bezier_traj = bezier_n(control_points)

plot_trajectories_grid([trajectory1, trajectory2, trajectory3, bezier_traj], ["Linear", "Curved", "Noisy", "Bezier"], 2, 2)