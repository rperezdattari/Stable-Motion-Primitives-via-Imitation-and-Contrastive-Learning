import numpy as np
import torch
from agent.neural_network import NeuralNetwork
from agent.utils.ranking_losses import ContrastiveLoss, TripletLoss
from agent.dynamical_system import DynamicalSystem
from agent.utils.dynamical_system_operations import normalize_state


class ContrastiveImitation:
    """
    Computes CONDOR losses and optimizes Neural Network
    """
    def __init__(self, data, params):
        # Params file parameters
        self.dim_workspace = params.workspace_dimensions
        self.dynamical_system_order = params.dynamical_system_order
        self.dim_state = self.dim_workspace * self.dynamical_system_order
        self.imitation_window_size = params.imitation_window_size
        self.batch_size = params.batch_size
        self.save_path = params.results_path
        self.multi_motion = params.multi_motion
        self.stabilization_loss = params.stabilization_loss
        self.generalization_window_size = params.stabilization_window_size
        self.imitation_loss_weight = params.imitation_loss_weight
        self.stabilization_loss_weight = params.stabilization_loss_weight
        self.load_model = params.load_model
        self.results_path = params.results_path
        self.resample_length = params.trajectories_resample_length
        self.interpolation_sigma = params.interpolation_sigma
        self.delta_t = 1  # used for training, can be anything

        # Parameters data processor
        self.primitive_ids = np.array(data['demonstrations primitive id'])
        self.n_primitives = data['n primitives']
        self.goals_tensor = torch.FloatTensor(data['goals training']).cuda()
        self.demonstrations_train = data['demonstrations train']
        self.n_demonstrations = data['n demonstrations']
        self.min_vel = torch.from_numpy(data['vel min train'].reshape([1, self.dim_workspace])).float().cuda()
        self.max_vel = torch.from_numpy(data['vel max train'].reshape([1, self.dim_workspace])).float().cuda()
        min_acc = torch.from_numpy(data['acc min train'].reshape([1, self.dim_workspace])).float().cuda()
        max_acc = torch.from_numpy(data['acc max train'].reshape([1, self.dim_workspace])).float().cuda()

        # Dynamical-system-only params
        self.params_dynamical_system = {'saturate transition': params.saturate_out_of_boundaries_transitions,
                                        'x min': data['x min'],
                                        'x max': data['x max'],
                                        'vel min train': self.min_vel,
                                        'vel max train': self.max_vel,
                                        'acc min train': min_acc,
                                        'acc max train': max_acc}

        # Initialize Neural Network losses
        self.mse_loss = torch.nn.MSELoss()
        self.triplet_loss = TripletLoss(margin=params.triplet_margin, swap=True)
        self.contrastive_loss = ContrastiveLoss(margin=params.contrastive_margin)

        # Initialize Neural Network
        self.model = NeuralNetwork(dim_state=self.dim_state,
                                   dynamical_system_order=self.dynamical_system_order,
                                   n_primitives=self.n_primitives,
                                   multi_motion=self.multi_motion,
                                   latent_gain_lower_limit=params.latent_gain_lower_limit,
                                   latent_gain_upper_limit=params.latent_gain_upper_limit,
                                   latent_gain=params.latent_gain,
                                   latent_space_dim=params.latent_space_dim,
                                   neurons_hidden_layers=params.neurons_hidden_layers,
                                   adaptive_gains=params.adaptive_gains).cuda()

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=params.learning_rate,
                                           weight_decay=params.weight_decay)

        # Load Neural Network if requested
        if self.load_model:
            self.model.load_state_dict(torch.load(self.results_path + 'model'), strict=False)

        # Initialize latent goals
        self.model.update_goals_latent_space(self.goals_tensor)

    def init_dynamical_system(self, initial_states, primitive_type=None, delta_t=1):
        """
        Creates dynamical system using the parameters/variables of the learning policy
        """
        # If no primitive type, assume single-model learning
        if primitive_type is None:
            primitive_type = torch.FloatTensor([1])

        # Create dynamical system
        dynamical_system = DynamicalSystem(x_init=initial_states,
                                           model=self.model,
                                           primitive_type=primitive_type,
                                           order=self.dynamical_system_order,
                                           min_state_derivative=[self.params_dynamical_system['vel min train'],
                                                                 self.params_dynamical_system['acc min train']],
                                           max_state_derivative=[self.params_dynamical_system['vel max train'],
                                                                 self.params_dynamical_system['acc max train']],
                                           saturate_transition=self.params_dynamical_system['saturate transition'],
                                           dim_state=self.dim_state,
                                           delta_t=delta_t,
                                           x_min=self.params_dynamical_system['x min'],
                                           x_max=self.params_dynamical_system['x max'])

        return dynamical_system

    def imitation_cost(self, state_sample, primitive_type_sample):
        """
        Imitation cost
        """
        # Create dynamical system
        dynamical_system = self.init_dynamical_system(initial_states=state_sample[:, :, 0],
                                                      primitive_type=primitive_type_sample)

        # Compute imitation error for transition window
        imitation_error_accumulated = 0

        for i in range(self.imitation_window_size - 1):
            # Compute transition
            x_t_d = dynamical_system.transition(space='task')['desired state']

            # Compute and accumulate error
            imitation_error_accumulated += self.mse_loss(x_t_d[:, :self.dim_workspace], state_sample[:, :self.dim_workspace, i + 1].cuda())
        imitation_error_accumulated = imitation_error_accumulated / (self.imitation_window_size - 1)

        return imitation_error_accumulated * self.imitation_loss_weight

    def contrastive_matching(self, state_sample, primitive_type_sample):
        """
        Transition matching cost
        """
        # Create dynamical systems
        dynamical_system_task = self.init_dynamical_system(initial_states=state_sample,
                                                           primitive_type=primitive_type_sample)

        dynamical_system_latent = self.init_dynamical_system(initial_states=state_sample,
                                                             primitive_type=primitive_type_sample)

        # Compute cost over trajectory
        contrastive_matching_cost = 0
        batch_size = state_sample.shape[0]
        y_t_task = None

        for i in range(self.generalization_window_size):
            # Do transition
            y_t_task_prev = y_t_task
            y_t_task = dynamical_system_task.transition(space='task')['latent state']
            _, y_t_latent = dynamical_system_latent.transition_latent_system()

            if i > 0:  # we need at least one iteration to have a previous point to push the current one away from
                # Transition matching cost
                if self.stabilization_loss == 'contrastive':
                    # Anchor
                    anchor_samples = torch.cat((y_t_task, y_t_task))

                    # Positive/Negative samples
                    contrastive_samples = torch.cat((y_t_latent, y_t_task_prev))

                    # Contrastive label
                    contrastive_label_pos = torch.ones(batch_size).cuda()
                    contrastive_label_neg = torch.zeros(batch_size).cuda()
                    contrastive_label = torch.cat((contrastive_label_pos, contrastive_label_neg))

                    # Compute cost
                    contrastive_matching_cost += self.contrastive_loss(anchor_samples, contrastive_samples, contrastive_label)

                elif self.stabilization_loss == 'triplet':
                    contrastive_matching_cost += self.triplet_loss(y_t_task, y_t_latent, y_t_task_prev)
        contrastive_matching_cost = contrastive_matching_cost / (self.generalization_window_size - 1)

        return contrastive_matching_cost * self.stabilization_loss_weight

    def demo_sample(self):
        """
        Samples a batch of windows from the demonstrations
        """

        # Select demonstrations randomly
        selected_demos = np.random.choice(range(self.n_demonstrations), self.batch_size)

        # Get random points inside trajectories
        i_samples = np.random.randint(0, self.resample_length, self.batch_size, dtype=int)

        # Get sampled positions from training data
        position_sample = self.demonstrations_train[selected_demos, i_samples]
        position_sample = torch.FloatTensor(position_sample).cuda()

        # Create empty state
        state_sample = torch.empty([self.batch_size, self.dim_state, self.imitation_window_size]).cuda()

        # Fill first elements of the state with position
        state_sample[:, :self.dim_workspace, :] = position_sample[:, :, (self.dynamical_system_order - 1):]

        # Fill rest of the elements with velocities for second order systems
        if self.dynamical_system_order == 2:
            velocity = (position_sample[:, :, 1:] - position_sample[:, :, :-1]) / self.delta_t
            velocity_norm = normalize_state(velocity,
                                            x_min=self.min_vel.reshape(1, self.dim_workspace, 1),
                                            x_max=self.max_vel.reshape(1, self.dim_workspace, 1))
            state_sample[:, self.dim_workspace:, :] = velocity_norm

        # Finally, get primitive ids of sampled batch (necessary when multi-motion learning)
        primitive_type_sample = self.primitive_ids[selected_demos]
        primitive_type_sample = torch.FloatTensor(primitive_type_sample).cuda()

        return state_sample, primitive_type_sample

    def space_sample(self):
        """
        Samples a batch of windows from the state space
        """
        with torch.no_grad():
            # Sample state
            state_sample_gen = torch.Tensor(self.batch_size, self.dim_state).uniform_(-1, 1).cuda()

            # Choose sampling methods
            if not self.multi_motion:
                primitive_type_sample_gen = torch.randint(0, self.n_primitives, (self.batch_size,)).cuda()
            else:
                # If multi-motion learning also sample in interpolation space
                # sigma of the samples are in the demonstration spaces
                encodings = torch.eye(self.n_primitives).cuda()
                primitive_type_sample_gen_demo = encodings[torch.randint(0, self.n_primitives, (round(self.batch_size * self.interpolation_sigma),)).cuda()]

                # 1 - sigma  of the samples are in the interpolation space
                primitive_type_sample_gen_inter = torch.rand(round(self.batch_size * (1 - self.interpolation_sigma)), self.n_primitives).cuda()

                # Concatenate both samples
                primitive_type_sample_gen = torch.cat((primitive_type_sample_gen_demo, primitive_type_sample_gen_inter), dim=0)

        return state_sample_gen, primitive_type_sample_gen

    def compute_loss(self, state_sample_IL, primitive_type_sample_IL, state_sample_gen, primitive_type_sample_gen):
        """
        Computes total cost
        """
        loss_list = []  # list of losses
        losses_names = []

        # Learning from demonstrations outer loop
        if self.imitation_loss_weight != 0:
            imitation_cost = self.imitation_cost(state_sample_IL, primitive_type_sample_IL)
            loss_list.append(imitation_cost)
            losses_names.append('Imitation')

        # Transition matching
        if self.stabilization_loss_weight != 0:
            contrastive_matching_cost = self.contrastive_matching(state_sample_gen, primitive_type_sample_gen)
            loss_list.append(contrastive_matching_cost)
            losses_names.append('Stability')

        # Sum losses
        loss = 0
        for i in range(len(loss_list)):
            loss += loss_list[i]

        return loss, loss_list, losses_names

    def update_model(self, loss):
        """
        Updates Neural Network with computed cost
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update goal in latent space
        self.model.update_goals_latent_space(self.goals_tensor)

    def train_step(self):
        """
        Samples data and trains Neural Network
        """
        # Sample from space
        state_sample_gen, primitive_type_sample_gen = self.space_sample()

        # Sample from trajectory
        state_sample_IL, primitive_type_sample_IL = self.demo_sample()

        # Get loss from CONDOR
        loss, loss_list, losses_names = self.compute_loss(state_sample_IL,
                                                          primitive_type_sample_IL,
                                                          state_sample_gen,
                                                          primitive_type_sample_gen)

        # Update model
        self.update_model(loss)

        return loss, loss_list, losses_names