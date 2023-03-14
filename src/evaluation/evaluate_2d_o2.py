import matplotlib.pyplot as plt
from evaluation.evaluate import Evaluate
from agent.utils.dynamical_system_operations import denormalize_state


class Evaluate2DO2(Evaluate):
    """
    Class for evaluating second-order two-dimensional dynamical systems
    """
    def __init__(self, learner, data, params, verbose=True):
        super().__init__(learner, data, params, verbose=verbose)

    def compute_quali_eval(self, sim_results, attractor, primitive_id, iteration):
        """
        Computes qualitative results
        """
        save_path = self.learner.save_path + 'images/' + 'primitive_%i_iter_%i' % (primitive_id, iteration) + '.pdf'
        self.plot_dynamical_system(sim_results, attractor, save_path)
        return True

    def compute_diffeo_quali_eval(self, sim_results, sim_results_latent, primitive_id, iteration):
        # Not implemented
        return False

    def plot_dynamical_system(self, sim_results, attractor, save_path):
        """
        Plots demonstrations and simulated trajectories from grid initial states
        """
        # Update plot params
        plt.rcParams.update({'font.size': 14,
                             'figure.figsize': (8, 8)})

        ax = plt.gca()
        ax.grid(linestyle='--')

        # Get rainbow cmap
        cm = plt.get_cmap('gist_rainbow')
        num_colors = self.density ** self.dim_workspace
        ax.set_prop_cycle('color', [cm(1. * i / num_colors) for i in range(num_colors)])

        # Plot demonstrations (we need loop because they can have different lengths)
        for i in range(self.n_trajectories):
            plt.plot(self.demonstrations_eval[i][0], self.demonstrations_eval[i][1], color='lightgray', alpha=1.0,
                     linewidth=6)

        # Plot simulated trajectories
        plt.plot(denormalize_state(sim_results['visited states grid'][:, :, 0], self.x_min[0], self.x_max[0]),
                 denormalize_state(sim_results['visited states grid'][:, :, 1], self.x_min[1], self.x_max[1]),
                 linewidth=4,
                 zorder=11)

        # Plot attractors
        plt.scatter(attractor[:, 0], attractor[:, 1], linewidth=8, color='blue', zorder=12, edgecolors='black')

        # Plot details/info
        ax.set_xlabel('x (px)')
        ax.set_ylabel('y (px)')
        ax.set_title('Dynamical System')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')

        # Save
        print('Saving image to %s...' % save_path)
        plt.savefig(save_path, bbox_inches='tight')

        # Close
        plt.clf()
        plt.close()
