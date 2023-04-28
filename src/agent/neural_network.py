import torch
import numpy as np


class NeuralNetwork(torch.nn.Module):
    """
    Neural Network model
    """
    def __init__(self, dim_state, dynamical_system_order, n_primitives, multi_motion, latent_gain_lower_limit,
                 latent_gain_upper_limit, latent_gain, latent_space_dim, neurons_hidden_layers, adaptive_gains):
        super(NeuralNetwork, self).__init__()
        
        # Initialize Network parameters
        self.n_input = dim_state
        n_output = dim_state // dynamical_system_order
        latent_input_size = latent_space_dim
        self.multi_motion = multi_motion
        self.n_primitives = n_primitives
        self.latent_space_dim = latent_space_dim
        self.dynamical_system_order = dynamical_system_order
        self.latent_gain = latent_gain
        self.latent_gain_lower_limit = latent_gain_lower_limit
        self.latent_gain_upper_limit = latent_gain_upper_limit
        self.adaptive_gains = adaptive_gains

        # Select activation function
        self.activation = torch.nn.GELU()
        self.sigmoid = torch.nn.Sigmoid()

        # Initialize goals list
        self.goals_latent_space = list(np.zeros(n_primitives))

        # Primitives encodings
        self.primitives_encodings = torch.eye(n_primitives).cuda()

        # Initialize encoder layers: psi
        if multi_motion:
            self.encoder1 = torch.nn.Linear(self.n_input + n_primitives, neurons_hidden_layers)
        else:
            self.encoder1 = torch.nn.Linear(self.n_input, neurons_hidden_layers)
        self.norm_e_1 = torch.nn.LayerNorm(neurons_hidden_layers)
        self.encoder2 = torch.nn.Linear(neurons_hidden_layers, neurons_hidden_layers)
        self.norm_e_2 = torch.nn.LayerNorm(neurons_hidden_layers)
        self.encoder3 = torch.nn.Linear(neurons_hidden_layers, neurons_hidden_layers)
        self.norm_e_3 = torch.nn.LayerNorm(neurons_hidden_layers)

        # Norm output latent space
        self.norm_latent = torch.nn.LayerNorm(latent_input_size)

        # Initialize dynamical system decoder layers: phi
        self.decoder1_dx = torch.nn.Linear(latent_input_size, neurons_hidden_layers)
        self.norm_de_dx1 = torch.nn.LayerNorm(neurons_hidden_layers)
        self.decoder2_dx = torch.nn.Linear(neurons_hidden_layers, neurons_hidden_layers)
        self.norm_de_dx2 = torch.nn.LayerNorm(neurons_hidden_layers)
        self.decoder3_dx = torch.nn.Linear(neurons_hidden_layers, n_output)

        # Latent space
        self.gain_nn_1 = torch.nn.Linear(latent_input_size, self.latent_space_dim)

        self.norm_latent_gain_input = torch.nn.LayerNorm(latent_input_size)
        self.norm_gain_1 = torch.nn.LayerNorm(self.latent_space_dim)
        self.gain_nn_2 = torch.nn.Linear(self.latent_space_dim, latent_input_size)

        self.norm_de_dx0 = []
        for i in range(self.dynamical_system_order):
            self.norm_de_dx0.append(torch.nn.LayerNorm(self.latent_space_dim))

        self.norm_de_dx0 = torch.nn.ModuleList(self.norm_de_dx0)
        self.norm_de_dx0_0 = torch.nn.LayerNorm(self.latent_space_dim)
        self.norm_de_dx0_1 = torch.nn.LayerNorm(self.latent_space_dim)

    def update_goals_latent_space(self, goals):
        """
        Maps task space goal to latent space goal
        """
        for i in range(self.n_primitives):
            primitive_type = torch.FloatTensor([i]).cuda()
            input = torch.zeros([1, self.n_input])  # add zeros as velocity goal for second order DS
            input[:, :goals[i].shape[0]] = goals[i]
            self.goals_latent_space[i] = self.encoder(input, primitive_type)

    def get_goals_latent_space_batch(self, primitive_type):
        """
        Creates a batch with latent space goals computed in 'update_goals_latent_space'
        """
        goals_latent_space_batch = torch.zeros(primitive_type.shape[0], self.latent_space_dim).cuda()
        for i in range(self.n_primitives):
            goals_latent_space_batch[primitive_type == i] = self.goals_latent_space[i]

        return goals_latent_space_batch

    def get_encoding_batch(self, primitive_type):
        """
        When multi-model learning, encodes primitive id into one-hot code
        """
        encoding_batch = torch.zeros(primitive_type.shape[0], self.n_primitives).cuda()
        for i in range(self.n_primitives):
            encoding_batch[primitive_type == i] = self.primitives_encodings[i]

        return encoding_batch

    def encoder(self, x_t, primitive_type):
        """
        Maps task space state to latent space state (psi)
        """
        # Get batch encodings
        if primitive_type.ndim == 1:  # if primitive type needs to be encoded
            encoding = self.get_encoding_batch(primitive_type)
        else:  # we assume the code is ready
            encoding = primitive_type

        # Encoder layer 1
        if self.multi_motion:
            input_encoded = torch.cat((x_t.cuda(), encoding), dim=1)
            e_1 = self.activation(self.norm_e_1(self.encoder1(input_encoded)))
        else:
            e_1 = self.activation(self.norm_e_1(self.encoder1(x_t.cuda())))

        # Encoder layer 2
        e_2 = self.activation(self.norm_e_2(self.encoder2(e_1)))

        # Encoder layer 3
        e_3 = self.activation(self.encoder3(e_2))
        return e_3

    def decoder_dx(self, y_t):
        """
        Maps latent space state to task space derivative (phi)
        """
        # Normalize y_t
        y_t_norm = self.norm_de_dx0[0](y_t)

        # Decoder dx layer 1
        de_1 = self.activation(self.norm_de_dx1(self.decoder1_dx(y_t_norm)))

        # Decoder dx layer 2
        de_2 = self.activation(self.norm_de_dx2(self.decoder2_dx(de_1)))

        # Decoder dx layer 3
        de_3 = self.decoder3_dx(de_2)
        return de_3

    def gains_latent_dynamical_system(self, y_t_norm):
        """
        Computes gains latent dynamical system f^{L}
        """
        if self.adaptive_gains:
            input = y_t_norm
            latent_gain_1 = self.activation(self.norm_gain_1(self.gain_nn_1(input)))
            gains = self.sigmoid(self.gain_nn_2(latent_gain_1))

            # Keep gains between the set limits
            gains = gains * (self.latent_gain_upper_limit - self.latent_gain_lower_limit) + self.latent_gain_lower_limit
        else:
            gains = self.latent_gain
        return gains

    def latent_dynamical_system(self, y_t, primitive_type):
        """
        Stable latent dynamical system
        """
        if primitive_type.ndim > 1:  # if primitive is already encoded, decode TODO: this should be modified to work with changing goal position
            primitive_type = torch.argmax(primitive_type, dim=1)  # one hot encoding to integers

        # Get latent goals batch
        y_goal = self.get_goals_latent_space_batch(primitive_type)

        # With bad hyperparams y value can explode when simulating the system, creating nans -> clamp to avoid issues when hyperparam tuning
        y_t = torch.clamp(y_t, min=-3e18, max=3e18)

        # Normalize y_t
        y_t_norm = self.norm_latent_gain_input(y_t)

        # Get gain latent dynamical system
        alpha = self.gains_latent_dynamical_system(y_t_norm)

        # First order dynamical system in latent space
        dy_t = alpha * (y_goal.cuda() - y_t.cuda())

        return dy_t