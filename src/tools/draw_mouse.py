import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import time


def draw(event):
    x, y = event.x, event.y
    if canvas.old_coords:
        x1, y1 = canvas.old_coords
        canvas.create_line(x, y, x1, y1, fill='#f11', width=4, dash=(4, 10))
    canvas.old_coords = x, y
    canvas.trajectory.append([x, canvas.height-y])  # mirror y given that it starts from the upper side of the image

    if canvas.start is False:
        canvas.start = True
        canvas.time = time.time()
    else:
        dt = time.time() - canvas.time
        canvas.sampling_times.append(dt)
        canvas.time = time.time()


def reset_coords(event):
    canvas.old_coords = None
    canvas.trajectories.append(np.stack(canvas.trajectory).astype(float))
    canvas.trajectory = []
    print('Mean sampling time:', np.mean(canvas.sampling_times))
    canvas.sampling_times = []
    canvas.start = False


def get_velocity(trajectory, alpha):
    trajectory_off_plus = np.append(trajectory, np.array([[0, 0]]), axis=0)
    trajectory_off_minus = np.append(np.array([[0, 0]]), trajectory, axis=0)
    velocity = trajectory_off_minus - trajectory_off_plus
    velocity = - velocity[1:] * alpha
    velocity[-1] = [0, 0]
    return velocity


root = tk.Tk()
height = 400
width = 600
canvas = tk.Canvas(root, width=width, height=height)
canvas.height = 400
canvas.pack()
canvas.old_coords = None
canvas.trajectory = []
canvas.trajectories = []
canvas.start = False
canvas.sampling_times = []

name = 'multi_v2'
root.bind('<B1-Motion>', draw)
root.bind('<ButtonRelease-1>', reset_coords)
root.mainloop()

for i in range(len(canvas.trajectories)):
    trajectory = canvas.trajectories[i]
    trajectory = trajectory.reshape(1, trajectory.shape[0], trajectory.shape[1])
    np.save(name + '_' + str(i), trajectory)
    print('Trajectory %i shape:' % i, trajectory.shape)