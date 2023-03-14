import optuna
import numpy as np
import os


def optuna_generate_optimizers(trial, params):
    params.learning_rate = trial.suggest_float('learning rate', low=1e-5, high=1e-3, log=True)
    params.stabilization_loss_weight = trial.suggest_float('stabilization loss weight', low=1e-1, high=10, log=False)
    params.imitation_window_size = trial.suggest_int('imitation window size', low=2, high=15)
    params.stabilization_window_size = trial.suggest_int('stabilization window size', low=2, high=15)
    params.contrastive_margin = trial.suggest_float('contrastive_margin', low=1e-5, high=1e-1, log=True)
    params.triplet_margin = params.contrastive_margin
    if params.adaptive_gains:
        params.latent_gain_upper_limit = trial.suggest_float('latent gain upper limit', low=1e-3, high=10, log=False)
    else:
        params.latent_gain = trial.suggest_float('latent gain', low=1e-4, high=1e-1)
    return params


def optuna_get_remaining_models_ids(params_name, length_dataset):
    file_dir = 'experiments/status_variables/' + params_name + '_remaining.npy'
    if os.path.isfile(file_dir):
        remaining_models_ids_numpy = np.load(file_dir)
    else:
        remaining_models_ids_numpy = np.arange(length_dataset)

    remaining_models_ids = [int(remaining_models_ids_numpy[i]) for i in range(len(remaining_models_ids_numpy))]
    return remaining_models_ids


def optuna_compute_objective(trial, params, mean_distance_to_goal, mean_RMSE, mean_RMSE_trajectory_comparison, iteration, prune):
    # Compute objective for optuna
    objective = mean_RMSE + params.gamma_objective_1 * mean_RMSE_trajectory_comparison + params.gamma_objective_2 * mean_distance_to_goal
    print('Hyper objective:', objective)
    if prune:
        # Report trial
        trial.report(objective, iteration)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return objective


def optuna_study(train, task, params, params_name):
    # Get high-level study parameters
    if task == 'train_dataset':
        study_name = 'train_dataset_' + params_name + '_' + params.dataset_name
        remaining_models_ids = optuna_get_remaining_models_ids(params_name, params.length_dataset)
        n_trials = len(remaining_models_ids)
    elif task == 'optimize':
        study_name = 'optuna_study_' + params_name + '_' + params.dataset_name
        n_trials = params.optuna_n_trials
    else:
        raise NameError('Selected task is not valid. Try: train_datasset, optimize')

    # Create/load study
    try:
        study = optuna.create_study(study_name=study_name,
                                    direction='minimize',
                                    storage='sqlite:///%s.db' % study_name,
                                    sampler=optuna.samplers.TPESampler())
    except optuna.exceptions.DuplicatedStudyError:
        print('Loading existing study...')
        study = optuna.load_study(study_name=study_name,
                                  storage='sqlite:///%s.db' % study_name)

    # Add models ids to queue for evaluation
    if task == 'train_dataset':
        for id_model in remaining_models_ids:
            study.enqueue_trial({'Primitive Id': id_model})

    # Run optimization
    study.optimize(train, n_trials=n_trials)
