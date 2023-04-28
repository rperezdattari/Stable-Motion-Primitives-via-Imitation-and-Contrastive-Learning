# Import libraries
from simple_parsing import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from hyperparam_optimization.optuna_functions import optuna_generate_optimizers, optuna_compute_objective, optuna_study, optuna_get_remaining_models_ids
from initializer import initialize_framework
import importlib
import optuna
import os

# Get arguments
parser = ArgumentParser()
parser.add_argument('--params', type=str, default='1st_order_2D', help='')
parser.add_argument('--train-dataset', type=bool, default=False, help='')
parser.add_argument('--hyperparameter-optimization', type=bool, default=False, help='')
parser.add_argument('--results-base-directory', type=str, default='./', help='')
args = parser.parse_args()

# Import parameters
Params = getattr(importlib.import_module('params.' + args.params), 'Params')


def train(trial):
    # Initialize objective
    objective = 1e16
    params = Params(args.results_base_directory)
    params.quanti_eval = True
    params.diffeo_quanti_eval = True

    # Generate optuna optimizers for hyperparemeter optimization
    if args.hyperparameter_optimization:
        params = optuna_generate_optimizers(trial, params)
    # Or not, if using optuna only for training dataset
    elif args.train_dataset:
        remaining_models_ids = optuna_get_remaining_models_ids(args.params, params.length_dataset)
        params.selected_primitives_ids = str(trial.suggest_categorical('Primitive Id', remaining_models_ids))

    params.results_path += params.selected_primitives_ids + '/'

    # Check if selected primitive was trained already
    if os.path.exists(params.results_path + 'images/') and args.train_dataset:
        print('Experiment exists already!')
        raise optuna.exceptions.TrialPruned()

    # Initialize training
    learner, evaluator, _ = initialize_framework(params, args.params)

    # Start tensorboard writer
    if params.save_evaluation and not args.hyperparameter_optimization:
        log_name = args.params + '_' + params.selected_primitives_ids
        writer = SummaryWriter(log_dir='results/tensorboard_runs/' + log_name)

    # Train
    for iteration in range(params.max_iterations + 1):
        # Evaluate model
        if iteration % params.evaluation_interval == 0:
            metrics_acc, metrics_stab = evaluator.run(iteration=iteration)

            if params.save_evaluation and not args.hyperparameter_optimization:
                evaluator.save_progress(params.results_path, iteration, learner.model, writer)

            print('Metrics sum:', metrics_acc['metrics sum'], '; Number of unsuccessful trajectories:', metrics_stab['n spurious'])

            if args.hyperparameter_optimization or args.train_dataset:
                if args.train_dataset:
                    prune = False
                else:
                    prune = True

                # Compute hyperparameter optimization objective
                objective = optuna_compute_objective(trial, params, metrics_stab['mean dist to goal'],
                                                     metrics_acc['RMSE'], metrics_stab['diffeo mismatch'], iteration,
                                                     prune=prune)

        # Run one train step of the learner
        loss, _, _ = learner.train_step()

        if iteration % 10 == 0:
            print(iteration, 'Total cost:', loss.item())

    return objective


if __name__ == "__main__":
    params = Params(args.results_base_directory)
    if args.hyperparameter_optimization:  # optuna hyperparameter optimization
        optuna_study(train, task='optimize', params=params, params_name=args.params)
    elif args.train_dataset:  # optuna train complete dataset
        optuna_study(train, task='train_dataset', params=params, params_name=args.params)
