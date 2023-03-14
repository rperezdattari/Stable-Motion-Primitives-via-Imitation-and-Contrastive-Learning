"""
Authors:
    Lekan Molu <patlekno@icloud.com>
    Rodrigo Perez-Dattari <r.j.perezdattari@tudelft.nl>
    Initial version of this code extracted from:
    https://github.com/robotsorcerer/LyapunovLearner/blob/master/scripts/visualization/traj_plotter.py
"""

import time
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec


def buffered_axis_limits(amin, amax, buffer_factor=1.0):
    """
    Increases the range (amin, amax) by buffer_factor on each side
    and then rounds to precision of 1/10th min or max.
    Used for generating good plotting limits.
    For example (0, 100) with buffer factor 1.1 is buffered to (-10, 110)
    and then rounded to the nearest 10.
    """
    diff = amax - amin
    amin -= (buffer_factor-1)*diff
    amax += (buffer_factor-1)*diff
    magnitude = np.floor(np.log10(np.amax(np.abs((amin, amax)) + 1e-100)))
    precision = np.power(10, magnitude-1)
    amin = np.floor(amin/precision) * precision
    amax = np.ceil(amax/precision) * precision
    return (amin, amax)


class TrajectoryPlotter():
    def __init__(self, fig, fontdict=None, pause_time=1e-3, labels=None, x0=None, goal=None):
        """
        Class TrajectoryPlotter:
        This class expects to be constantly given values to plot in realtime.
        """
        self._fig = fig
        self._gs = gridspec.GridSpec(1, 1, self._fig)
        plt.ion()
        self._ax = plt.subplot(self._gs[0])

        self._labels = labels
        self._init = False
        self.Xinit = x0
        self._fontdict  = fontdict
        self._labelsize = 16
        self.goal = goal
        self.pause_time = pause_time

        plt.rcParams['toolbar'] = 'None'
        for key in plt.rcParams:
            if key.startswith('keymap.'):
                plt.rcParams[key] = ''

        if np.any(self.Xinit):
            self.init(x0.shape[-1])

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def init(self, data_len):
        """
        Initialize plots based off the length of the data array.
        """
        self._data_len = data_len
        self._xi = np.empty((0, data_len))
        self._xi_dot = np.empty((0, data_len))

        D = self.Xinit.shape[-1]

        # Plot it
        colors = ['blue', 'magenta', 'purple']
        self._plots = [np.nan for _ in range(self.Xinit.shape[-1])]
        for i in range(D):
            self._plots[i] = [self._ax.plot(self.Xinit[0, i], self.Xinit[1, i], 'o', color=colors[i], markersize=10)[0]]
            self._plots[i] += [self._ax.plot([], [], color=colors[i], linewidth=2.5)[0]]

        # Show attractor
        self.targ, = self._ax.plot(self.goal[0], self.goal[1], 'g*', markersize=15, linewidth=3, label='Target')
        self._ax.set_xlabel('$x$', fontdict=self._fontdict)
        self._ax.set_ylabel('$y$', fontdict=self._fontdict)

        self._ax.xaxis.set_tick_params(labelsize=self._labelsize)
        self._ax.yaxis.set_tick_params(labelsize=self._labelsize)

        self._ax.grid('on')
        self._ax.legend(loc='best')
        self._ax.set_title('Dynamical System', fontdict=self._fontdict)

        self._init = True

    def update(self, xi):
        """
        Update the plots with new data x. Assumes x is a one-dimensional array.
        """
        D = xi.shape[-1]  # number of conditions
        if not self._init:
            self.init(xi.shape[0])

        assert xi.shape[1] == self._data_len, f'xi of shape {xi.shape}has to be of shape {self._data_len}'

        self._xi = np.append(self._xi, xi[0].reshape(1, D), axis=0)
        self._xi_dot = np.append(self._xi_dot, xi[1].reshape(1, D), axis=0)

        # idx=0
        xlims, ylims = [np.nan for _ in range(len(self._plots))], [np.nan for _ in range(len(self._plots))]

        for idx, traj_plotter in enumerate(self._plots):
            traj_plotter[-1].set_data(self._xi[:, idx], self._xi_dot[:, idx]) #

            x_min, x_max = np.amin(self._xi[:, idx]), np.amax(self._xi[:, idx])
            y_min, y_max = np.amin(self._xi_dot[:, idx]), np.amax(self._xi_dot[:, idx])

            xlims[idx] = (x_min, x_max)
            ylims[idx] = (y_min, y_max)

            # idx+=1

        x_min = min(0, min([tup[0] for tup in xlims])); y_min = min(0, min([tup[0] for tup in ylims]))
        x_max = max(0, max([tup[1] for tup in xlims])); y_max = max(0, max([tup[1] for tup in ylims]))

        self._ax.set_xlim(buffered_axis_limits(x_min, x_max, buffer_factor=1.05))
        self._ax.set_ylim(buffered_axis_limits(y_min, y_max, buffer_factor=1.05))

        self.draw()
        time.sleep(self.pause_time)

    def draw(self):
        for plots in self._plots:
            for each_plot in plots:
                self._ax.draw_artist(each_plot)

        self._fig.canvas.flush_events()


if __name__ == "__main__":
    fig, ax = plt.subplots()
    fig.show()
    x = np.array([[0, 0]]).T
    traj_plotter = TrajectoryPlotter(fig, x0=x, goal=[0, 0])
    traj_plotter.init(data_len=1)
    for i in range(1000):
        x = np.array([[np.sin(i*0.007), np.sin(i*0.014)]]).T
        traj_plotter.update(x)
