from evaluation.evaluate import Evaluate
import numpy as np
import matplotlib.pyplot as plt
from agent.utils.dynamical_system_operations import denormalize_state


class EvaluateND(Evaluate):
    """
    Class for evaluating n-dimensional dynamical systems
    """
    def __init__(self, learner, data, params, verbose=True):
        super().__init__(learner, data, params, verbose=verbose)

    def compute_quali_eval(self, sim_results, attractor, primitive_id, iteration):
        """
        Computes qualitative results
        """

        # Get time demonstrations
        delta_t_history = np.array(self.delta_t_eval)
        time_demonstrations = self.compute_time(delta_t_history)

        # Get time simulated trajectories
        delta_t_eval = np.mean(self.delta_t_eval[0])  # assumes delta is the same among demonstrations
        time_simulations = np.repeat(np.arange(0, sim_results['visited states grid'].shape[0] * delta_t_eval, delta_t_eval).reshape(-1, 1),
                                     self.density**self.dim_workspace, axis=1)

        # Plot
        save_path = self.learner.save_path + 'images/' + 'primitive_%i_iter_%i' % (primitive_id, iteration) + '.pdf'
        self.plot_nd_motion(sim_results, time_demonstrations, time_simulations, primitive_id, save_path=save_path)
        return True

    def compute_time(self, delta_t_history):
        """
        Computes time for plotting simulated trajectories
        """
        time_history = []

        for i in range(len(delta_t_history)):
            # Sum every delta to get the time of the trajectory
            time = 0
            time_history_demo = []
            for delta_t in delta_t_history[i]:
                time = time + delta_t
                time_history_demo.append(time)
            time_history.append(time_history_demo)
        return time_history

    def plot_nd_motion(self, sim_results, time_demonstrations, time_simulations, primitive_id, save_path):
        """
        Plots n-dimensional motion per dimension
        """
        plt.rcdefaults()
        plt.rcParams.update({'font.size': 20})

        # Create subplots
        n_joints = self.dim_workspace
        fig, axs = plt.subplots(n_joints, figsize=(10, 25))

        # Add title of subplots
        fig.suptitle('Dynamical System')

        # Plot every joint simulated trajectories
        denorm_visited_states_grid = denormalize_state(sim_results['visited states grid'], self.x_min, self.x_max)
        for i in range(n_joints):
            axs[i].set_title('Joint' + ' ' + str(i + 1))
            axs[i].set_xlabel('time [s]')
            axs[i].set_ylabel('angle [rad]')
            axs[i].grid()
            axs[i].margins(x=0)
            axs[i].plot(time_simulations, denorm_visited_states_grid[:, :, i], color='blue', linewidth=1.0,
                        alpha=0.15)

        # Plot every joint demonstrations
        denorm_visited_states_demos = denormalize_state(sim_results['visited states demos'], self.x_min, self.x_max)
        for i in range(n_joints):
            for j in range(self.n_trajectories):
                if self.primitive_ids[j] == primitive_id:
                    axs[i].plot(time_demonstrations[j], self.demonstrations_eval[j][i], color='black', linewidth=8)
                    axs[i].scatter(time_demonstrations[j][-1], self.demonstrations_eval[j][i][-1], color='red', edgecolors='black', zorder=10000, s=180)

        for i in range(n_joints):
            for j in range(np.sum(self.primitive_ids == primitive_id)):
                axs[i].plot(time_simulations[:, 0], denorm_visited_states_demos[:, j, i], color='red', linestyle='--', linewidth=2)

        fig.tight_layout()

        # Save
        print('Saving image to %s...' % save_path)
        fig.savefig(save_path)

    def compute_diffeo_quali_eval(self, sim_results, sim_results_latent, primitive_id, iteration):
        # Not implemented
        return False
