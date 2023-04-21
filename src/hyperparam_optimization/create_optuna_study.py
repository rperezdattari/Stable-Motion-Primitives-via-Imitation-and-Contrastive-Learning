"""
Authors:
    Rodrigo Perez-Dattari <r.j.perezdattari@tudelft.nl>
"""
import optuna
import sys
sys.path.append('.')

# Get arguments
from simple_parsing import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--params', type=str, default='2nd_order_outer', help='')
parser.add_argument('--train-dataset', type=bool, default=False, help='')
parser.add_argument('--hyperparameter-optimization', type=bool, default=False, help='')
parser.add_argument('--results-base-directory', type=str, default='./', help='')
args = parser.parse_args()

# Import parameters
import importlib
Params = getattr(importlib.import_module('params.' + args.params), 'Params')
params = Params(args.results_base_directory)

if args.hyperparameter_optimization:
    task = 'optimize'
elif args.train_dataset:
    task = 'train_dataset'
else:
    raise NameError('Selected task does not exist.')

# Get high-level study parameters
if task == 'train_dataset':
    study_name = 'train_dataset_' + args.params + '_' + params.dataset_name
    sampler = optuna.samplers.RandomSampler()
elif task == 'optimize':
    study_name = 'optuna_study_' + args.params + '_' + params.dataset_name
    sampler = optuna.samplers.TPESampler()
else:
    raise NameError('Selected task is not valid. Try: train_datasset, optimize')

study = optuna.create_study(study_name=study_name,
                            direction='minimize',
                            storage='sqlite:///%s.db' % study_name,
                            sampler=sampler)

if task == 'optimize':  # add initial guess of parameters of hyperparameter optimization
    initial_guess = {'learning rate': params.learning_rate,
                     'imitation window size': params.imitation_window_size}

    if params.stabilization_loss_weight > 0:
        initial_guess.update({'stabilization loss weight': params.stabilization_loss_weight,
                              'stabilization window size': params.stabilization_window_size,
                              'contrastive_margin': params.contrastive_margin,
                              'latent gain upper limit': params.latent_gain_upper_limit})

    # Add guess to study
    study.enqueue_trial(initial_guess)
