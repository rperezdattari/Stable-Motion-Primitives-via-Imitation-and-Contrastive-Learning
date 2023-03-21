import os
import time
from dataclasses import fields
from data_preprocessing.data_preprocessor import DataPreprocessor
from agent.contrastive_imitation import ContrastiveImitation
from evaluation.evaluator_init import evaluator_init


def initialize_framework(params, name_params, verbose=True):
    """
    Runs data preprocessor and initializes learner and evaluator
    """
    if params.save_evaluation:
        # Create directories in computer required for saving results
        create_directories(params.results_path)

        # Save selected parameters in results directory
        save_parameters(params, name_params)

    # Load and preprocess demonstrations
    data = DataPreprocessor(params=params, verbose=verbose).run()

    # Initialize learning agent
    learner = ContrastiveImitation(data=data, params=params)

    # Initialize learning process evaluation
    evaluator = evaluator_init(learner=learner, data=data, params=params, verbose=verbose)

    if verbose:
        # Print general configuration information
        print('\n<<< CONDOR framework: stable and flexible movement primitives >>> \n')
        print('Parameters:', name_params)
        print('Dynamical System order:', params.dynamical_system_order)
        print('Dataset:', params.dataset_name)
        print('Demos ID:', params.selected_primitives_ids)
        print('Results path:', params.results_path)
        print('\n')
        time.sleep(3)

    return learner, evaluator, data


def create_directories(results_path):
    """
    Creates the requested directory and subfolders
    """
    try:
        if not os.path.exists(results_path + 'images/'):
            os.makedirs(results_path + 'images/')
            os.makedirs(results_path + 'stats/')
    except FileExistsError:
        pass


def save_parameters(params, params_name):
    """
    Saves selected parameters in params.txt file
    """
    with open(params.results_path + 'params.txt', 'w') as f:
        f.write(params_name + '\n \n')
        f.write(' '.join(['%s = %s \n' % (field.name, field.default) for field in fields(params)]))
