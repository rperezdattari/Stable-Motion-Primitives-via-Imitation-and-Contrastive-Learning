from dataclasses import dataclass
import numpy as np


@dataclass
class Params:
    """ General parameters """
    dataset_name: str = 'optitrack'  # options: LASA, LAIR, optitrack, interpolation, joint_space
    results_path: str = 'results/1st_order_3D/'
    multi_motion: bool = False  # true when learning multiple motions together
    selected_primitives_ids: str = '0'  # id number from dataset_keys.py, e.g., '2' or '4,0,6'
    workspace_dimensions: int = 3  # dimensionality of the data
    saturate_out_of_boundaries_transitions: bool = True  # True to enforce positively invariant set
    dynamical_system_order: int = 1  # options: 1, 2

    """ Latent Dynamical System parameters """
    adaptive_gains: bool = True  # adaptive gains if true
    latent_gain_lower_limit: float = 0  # adaptive gains lower limit (always zero in paper)
    latent_gain_upper_limit: float = 0.0997  # adaptive gains upper limit
    latent_gain: float = 0.008  # value of gains when fixed

    """ Neural Network """
    latent_space_dim: int = 300  # dimensionality latent space
    neurons_hidden_layers: int = 300  # number of neurons per layer
    batch_size: int = 250  # sampling batch size
    learning_rate: float = 0.00059  # AdamW learning rate
    weight_decay: float = 0.0001  # AdamW weight decay

    """ Contrastive Imitation """
    imitation_loss_weight: int = 1  # imitation loss weight
    stabilization_loss_weight: int = 0.93  # stability loss weight
    imitation_window_size: int = 15  # imitation window size
    stabilization_window_size: int = 2  # stability window size
    stabilization_loss: str = 'contrastive'  # options: contrastive, triplet
    contrastive_margin: float = 0.0333  # contrastive loss margin
    triplet_margin: float = 1e-4  # triplet loss margin
    interpolation_sigma: float = 0.8  # percentage of points sampled in demonstrations space when multi-model learning

    """ Training """
    train: bool = True  # true when training
    load_model: bool = False  # true to load previously trained model
    max_iterations: int = 41000  # maximum number of training iterations

    """ Preprocessing """
    spline_sample_type: str = 'from data'  # resample from spline type, options: from data, from data resample, evenly spaced
    workspace_boundaries_type: str = 'custom'  # options: from data, custom
    workspace_boundaries: np.ndarray = np.array([[-1.5, 1.5],
                                                 [0.0, 2.0],
                                                 [-0.5, 2.0]])  # list to provide boundaries when custom boundaries
    trajectories_resample_length: int = 2000  # amount of points resampled from splines when type spline_sample_type is 'from data resample' or 'evenly spaced'
    state_increment: float = 0.3  # when workspace_boundaries_type = from data, percentage to increment state-space size

    """ Evaluation """
    save_evaluation: bool = True  # true to save evaluation results
    evaluation_interval: int = 1000  # interval between training iterations to evaluate model
    quanti_eval: bool = True  # quantitative evaluation
    quali_eval: bool = True  # qualitative evaluation
    diffeo_quanti_eval: bool = False  # quantitative evaluation of diffeomorphism mismatch
    diffeo_quali_eval: bool = False  # qualitative evaluation of diffeomorphism mismatch
    ignore_n_spurious: bool = False  # when selecting best model, true to ignore amount of spurious attractors
    fixed_point_iteration_thr = 2  # distance threshold to consider that a point did not reach the goal
    density: int = 6  # density^workspace_dimension = amount of points sampled from state space for evaluation
    simulated_trajectory_length: int = 500  # integration length for evaluation
    evaluation_samples_length: int = 100  # integration steps skipped in quantitative evaluation for faster evaluation
    show_plotly: bool = True  # show evaluation during training

    """ Hyperparameter Optimization """
    gamma_objective_1 = 0.48  # weight 1 for hyperparameter evaluation
    gamma_objective_2 = 3.5  # weight 2 for hyperparameter evaluation
    optuna_n_trials = 1000  # maximum number of optuna trials

    """ Dataset training """
    length_dataset = 1  # number of primitives in dataset

    def __init__(self, results_base_directory):
        self.results_path = results_base_directory + self.results_path
