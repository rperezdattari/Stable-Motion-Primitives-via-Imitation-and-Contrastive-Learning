import numpy as np
from scipy.interpolate import splprep, splev
from agent.utils.dynamical_system_operations import normalize_state
from data_preprocessing.data_loader import load_demonstrations


class DataPreprocessor:
    def __init__(self, params, verbose=True):
        """
        Class for loading and preprocessing demonstrations
        """
        self.trajectories_resample_length = params.trajectories_resample_length
        self.state_increment = params.state_increment
        self.dim_workspace = params.workspace_dimensions
        self.dynamical_system_order = params.dynamical_system_order
        self.dim_state = self.dim_workspace * self.dynamical_system_order
        self.workspace_boundaries_type = params.workspace_boundaries_type
        self.workspace_boundaries = np.array(params.workspace_boundaries)
        self.eval_length = params.evaluation_samples_length
        self.dataset_name = params.dataset_name
        self.selected_primitives_id = params.selected_primitives_ids
        self.spline_sample_type = params.spline_sample_type

        self.delta_t = 1  # this value is for learning, so it can be anything
        self.imitation_window_size = params.imitation_window_size  # window size used for imitation cost
        self.verbose = verbose

    def run(self):
        """
        Computes relevant features from the raw demonstrations
        """
        # Load demonstrations and associated data
        loaded_data = load_demonstrations(self.dataset_name, self.selected_primitives_id)

        # Get features from demonstrations demonstrations
        features_demos = self.get_features_demos(loaded_data)

        # Generate training data
        demonstrations_train = self.generate_training_data(loaded_data, features_demos)

        # Get min/max derivatives of training demonstrations
        limits_derivatives = self.get_limits_derivatives(demonstrations_train)

        # Create preprocess output dictionary
        preprocess_output = {'demonstrations train': demonstrations_train}
        preprocess_output.update(loaded_data)
        preprocess_output.update(features_demos)
        preprocess_output.update(limits_derivatives)

        return preprocess_output

    def get_features_demos(self, loaded_data):
        """
        Computes useful features from demonstrations
        """
        demonstrations_raw = loaded_data['demonstrations raw']
        primitive_ids = loaded_data['demonstrations primitive id']

        # Get workspace boundaries
        x_min, x_max = self.get_workspace_boundaries(demonstrations_raw)

        # Get goal states positions
        goals = self.get_goals(demonstrations_raw, primitive_ids)

        # Normalize goals
        goals_training = normalize_state(goals, x_min, x_max)

        # Get number of demonstrated trajectories
        n_trajectories = len(demonstrations_raw)

        # Get trajectories length and eval indexes
        max_trajectory_length, trajectories_length, eval_indexes = self.get_trajectories_length(demonstrations_raw,
                                                                                                n_trajectories)

        # Collect info
        features_demos = {'x min': x_min,
                          'x max': x_max,
                          'goals': goals,
                          'goals training': goals_training,
                          'max demonstration length': max_trajectory_length,
                          'demonstrations length': trajectories_length,
                          'eval indexes': eval_indexes,
                          'n demonstrations': n_trajectories}

        return features_demos

    def get_workspace_boundaries(self, demonstrations_raw):
        """
        Computes workspace boundaries
        """
        if self.workspace_boundaries_type == 'from data':
            # Compute boundaries based on data
            max_single_trajectory = []
            min_single_trajectory = []

            # Get max for every trajectory in each dimension
            for j in range(len(demonstrations_raw)):
                max_single_trajectory.append(np.array(demonstrations_raw[j]).max(axis=1))
                min_single_trajectory.append(np.array(demonstrations_raw[j]).min(axis=1))

            # Get the max and min values along all of the trajectories
            x_max_orig = np.array(max_single_trajectory).max(axis=0)
            x_min_orig = np.array(min_single_trajectory).min(axis=0)

            # Add a tolerance
            x_max = x_max_orig + (x_max_orig - x_min_orig) * self.state_increment / 2
            x_min = x_min_orig - (x_max_orig - x_min_orig) * self.state_increment / 2

        elif self.workspace_boundaries_type == 'custom':
            # Use custom boundaries
            x_max = self.workspace_boundaries[:, 1]
            x_min = self.workspace_boundaries[:, 0]
        else:
            raise NameError('Selected workspace boundaries type not valid. Try: from data, custom')

        return x_min, x_max

    def get_trajectories_length(self, demonstrations_raw, n_trajectories):
        """
        Computes length trajectories, longest trajectory and evaluation indexes for fast evaluation
        """
        trajectories_length, eval_indexes = [], []
        max_trajectory_length = 0

        # Iterate through each demonstration
        for j in range(n_trajectories):
            # Get trajectory length
            length_demo = len(demonstrations_raw[j][0])
            trajectories_length.append(length_demo)

            # Find largest trajectory in demonstrations
            if length_demo > max_trajectory_length:
                max_trajectory_length = length_demo

            # Obtain indexes used for fast evaluation
            if length_demo > self.eval_length:
                eval_interval = np.floor(length_demo / self.eval_length)
                eval_indexes.append(np.arange(0, length_demo, eval_interval, dtype=np.int32))
            else:
                eval_indexes.append(np.arange(0, length_demo, 1, dtype=np.int32))

        return max_trajectory_length, trajectories_length, eval_indexes

    def get_goals(self, demonstrations_raw, primitive_ids):
        """
        Computes goal demonstrations from data
        """
        # Iterate through primitives
        goals = []
        for i in np.unique(primitive_ids):
            ids_primitives = primitive_ids == i
            demonstrations_primitive_ids = np.array(np.where(ids_primitives))[0]
            # Iterate through trajectories of each primitive
            goals_primitive = []
            for j in demonstrations_primitive_ids:
                goals_primitive.append(np.array(demonstrations_raw[j])[:, -1])

            # Average goals and append
            goal_mean = np.mean(np.array(goals_primitive), axis=0)
            goals.append(goal_mean)

        return np.array(goals)

    def generate_training_data(self, loaded_data, features_demos):
        """
        Normalizes demonstrations, resamples demonstrations using spline to keep a constant distance between points,
        and creates imitation window for backpropagation through time
        """
        demonstrations_raw = loaded_data['demonstrations raw']
        n_trajectories = len(demonstrations_raw)
        resampled_positions, error_acc = [], []

        # Pad demonstrations
        demonstrations_raw_padded = []
        for i in range(features_demos['n demonstrations']):
            padding_length = features_demos['max demonstration length'] - features_demos['demonstrations length'][i]
            demonstrations_raw_padded.append(np.pad(demonstrations_raw[i], ((0, 0), (0, padding_length)), mode='edge'))
        demonstrations_raw_padded = np.array(demonstrations_raw_padded)

        # Iterate through each demonstration
        for j in range(n_trajectories):
            if self.verbose:
                print('Data preprocessing, demonstration %i / %i' % (j + 1, n_trajectories))

            # Get current trajectory
            demo = np.array(demonstrations_raw_padded[j]).T
            length_demo = demo.shape[0]

            # Normalize demos
            demo_norm = normalize_state(demo, x_min=features_demos['x min'], x_max=features_demos['x max'])

            if self.spline_sample_type == 'evenly spaced':
                # Create phase array that spatially parametrizes demo in one dimension
                curve_phase = 0
                curve_phases, delta_phases = [curve_phase], []

                for i in range(length_demo - 1):  # iterate through every point in trajectory and assign a phase value
                    # Compute phase increment based on distance of consecutive points
                    delta_phase = np.linalg.norm(demo_norm[i + 1, :] - demo_norm[i, :])

                    if delta_phase == 0:
                        # If points in trajectory have zero phase difference, splprep throws error -> add small margin
                        delta_phase += 1e-15

                    # Increment phase
                    curve_phase += delta_phase

                    # Store phase and delta of current point in curve
                    curve_phases.append(curve_phase)
                    delta_phases.append(delta_phase)

                delta_phases.append(0)  # zero delta for last point
                curve_phases = np.array(curve_phases)
                delta_phases = np.array(delta_phases)
                max_phase = curve_phases[-1]

            elif self.spline_sample_type == 'from data' or self.spline_sample_type == 'from data resample':
                curve_phases = np.arange(0, length_demo * self.delta_t, self.delta_t)
                delta_phases = np.ones(length_demo) * self.delta_t  # TODO: could be extended to variable delta t
                max_phase = np.max(curve_phases)
            else:
                raise NameError('Spline sample type not valid, check params file for options.')

            # Create input for spline: demonstrations and corresponding phases
            spline_input = []
            for i in range(self.dim_workspace):
                spline_input.append(demo_norm[:, i])
            spline_input.append(curve_phases)
            spline_input.append(delta_phases)

            # Fit spline
            spline_parameters, _ = splprep(spline_input, s=0, k=1, u=curve_phases)  # s = 0 -> no smoothing; k = 1 -> linear interpolation

            # Create initial phases u with spatially equidistant points
            if self.spline_sample_type == 'evenly spaced' or self.spline_sample_type == 'from data resample':
                u = np.linspace(0, max_phase, self.trajectories_resample_length)
            elif self.spline_sample_type == 'from data':
                u = curve_phases
            else:
                raise NameError('Spline sample type not valid, check params file for options.')

            # Iterate using imitation window size to get position labels for backpropagation through time
            window = []
            for _ in range(self.imitation_window_size + (self.dynamical_system_order - 1)):
                # Compute demo positions based on current phase value
                spline_values = splev(u, spline_parameters)
                position_window = spline_values[:self.dim_workspace]

                # Append position to window trajectory
                window.append(position_window)

                # Find time/phase for next point in imitation window
                delta_phase = spline_values[-1]
                next_t = u + delta_phase
                u = np.clip(next_t, a_min=0, a_max=max_phase)  # update phase

                # Accumulate error for debugging
                predicted_phase = splev(u, spline_parameters)[-2]
                error_acc.append(np.mean(np.abs(predicted_phase - u)))

            resampled_positions.append(window)

        if self.verbose:
            print('Mean error spline resampling:', np.mean(error_acc))

        # Change axes order to one more intuitive
        # 0: trajectories; 1: states trajectory; 2: state dimensions; 3: imitation window position
        resampled_positions = np.transpose(np.array(resampled_positions), (0, 3, 2, 1))
        return resampled_positions

    def get_limits_derivatives(self, demos):
        """
        Computes velocity and acceleration of the training demonstrations
        """
        # Get velocities from normalized resampled demonstrations
        velocity = (demos[:, :, :, 1:] - demos[:, :, :, :-1]) / self.delta_t

        # Get accelerations from velocities
        acceleration = (velocity[:, :, :, 1:] - velocity[:, :, :, :-1]) / self.delta_t

        # Compute max velocities
        min_velocity = np.min(velocity, axis=(0, 1, 3))
        max_velocity = np.max(velocity, axis=(0, 1, 3))

        # Compute max acceleration
        if self.dynamical_system_order == 1:
            min_acceleration = None  # acceleration not used in first-order systems
            max_acceleration = None
        elif self.dynamical_system_order == 2:
            min_acceleration = np.min(acceleration, axis=(0, 1, 3))
            max_acceleration = np.max(acceleration, axis=(0, 1, 3))
        else:
            raise ValueError('Selected dynamical system order not valid, options: 1, 2.')

        # If second order, since the velocity is part of the state, we extend its limits
        if self.dynamical_system_order == 2:
            max_velocity_state = max_velocity + (max_velocity - min_velocity) * self.state_increment / 2
            min_velocity_state = min_velocity - (max_velocity - min_velocity) * self.state_increment / 2
            max_velocity = max_velocity_state
            min_velocity = min_velocity_state

        # Collect
        limits = {'vel min train': min_velocity,
                  'vel max train': max_velocity,
                  'acc min train': min_acceleration,
                  'acc max train': max_acceleration}
        return limits
