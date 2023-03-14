from tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from initializer import initialize_framework
import numpy as np

# Parameters file information
params_name = '1st_order_evaluate_2D'
results_base_directory = './'
simulation_length = 2000

# Load parameters
Params = getattr(importlib.import_module('params.' + params_name), 'Params')
params = Params(results_base_directory)

# Initialize framework
agent, _ = initialize_framework(params, params_name, verbose=False)

# Initialize dynamical system
x_t_init = np.array([[0.4, 0.8]])  # initial state
dynamical_system = agent.dynamical_system(initial_states=torch.FloatTensor(x_t_init).cuda())

# Initialize trajectory plotter
fig, ax = plt.subplots()
fig.show()
trajectory_plotter = TrajectoryPlotter(fig, x0=x_t_init.T, pause_time=1e-5, goal=agent.goals[0])

# Simulate dynamical system and plot
for i in range(simulation_length):
    # Do transition
    x_t, dx_t, _, _ = dynamical_system.transition(space='world')

    # Update plot
    trajectory_plotter.update(x_t.T.cpu().detach().numpy(), dx_t.T.cpu().detach().numpy())