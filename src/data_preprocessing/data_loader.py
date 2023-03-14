from datasets.dataset_keys import LASA, LAIR, optitrack, interpolation, joint_space
import os
import pickle
import numpy as np
import scipy.io as sio


def load_demonstrations(dataset_name, selected_primitives_ids):
    """
    Loads demonstrations
    """
    # Get names of primitives in dataset
    dataset_primitives_names = get_dataset_primitives_names(dataset_name)

    # Get names of selected primitives for training
    primitives_names, primitives_save_name = select_primitives(dataset_primitives_names, selected_primitives_ids)

    # Get number of selected primitives
    n_primitives = len(primitives_names)

    # Get loading path
    dataset_path = 'datasets/' + dataset_name + '/'

    # Get data loader
    data_loader = get_data_loader(dataset_name)

    # Load
    demonstrations, demonstrations_primitive_id, delta_t_eval = data_loader(dataset_path, primitives_names)

    # Out dictionary
    loaded_info = {'demonstrations raw': demonstrations,
                   'demonstrations primitive id': demonstrations_primitive_id,
                   'n primitives': n_primitives,
                   'delta t eval': delta_t_eval}
    return loaded_info


def get_dataset_primitives_names(dataset_name):
    """
    Chooses primitives keys
    """
    if dataset_name == 'LASA':
        dataset_primitives_names = LASA
    elif dataset_name == 'LAIR':
        dataset_primitives_names = LAIR
    elif dataset_name == 'optitrack':
        dataset_primitives_names = optitrack
    elif dataset_name == 'interpolation':
        dataset_primitives_names = interpolation
    elif dataset_name == 'joint_space':
        dataset_primitives_names = joint_space
    else:
        raise NameError('Dataset %s does not exist' % dataset_name)

    return dataset_primitives_names


def select_primitives(dataset, selected_primitives_ids):
    """
    Gets selected primitives
    """
    selected_primitives_names = []
    selected_primitives_save_name = ''
    selected_primitives_ids = list(map(int, selected_primitives_ids.split(',')))  # map from string to list
    for id in selected_primitives_ids:
        selected_primitives_names.append(dataset[id])
        selected_primitives_save_name += str(id) + '_'

    return selected_primitives_names, selected_primitives_save_name[:-1]


def get_data_loader(dataset_name):
    """
    Chooses data loader depending on the data type
    """
    if dataset_name == 'LASA':
        data_loader = load_LASA
    elif dataset_name == 'LAIR' or dataset_name == 'optitrack' or dataset_name == 'interpolation':
        data_loader = load_numpy_file
    elif dataset_name == 'joint_space':
        data_loader = load_from_dict
    else:
        raise NameError('Dataset %s does not exist' % dataset_name)

    return data_loader


def load_LASA(dataset_dir, demonstrations_names):
    """
    Load LASA matlab models
    """
    s_x, s_y, demos, primitive_id, dt = [], [], [], [], []
    for i in range(len(demonstrations_names)):
        mat_file = sio.loadmat(dataset_dir + demonstrations_names[i])
        data = mat_file['demos']

        for j in range(data.shape[1]):  # iterate through demonstrations
            s_x = data[0, j]['pos'][0, 0][0]
            s_y = data[0, j]['pos'][0, 0][1]
            s = [s_x, s_y]
            demos.append(s)
            dt.append(data[0, j]['dt'][0, 0][0, 0])
            primitive_id.append(i)

    return demos, primitive_id, dt


def load_numpy_file(dataset_dir, demonstrations_names):
    """
    Loads demonstrations in numpy files
    """
    demos, primitive_id = [], []
    for i in range(len(demonstrations_names)):
        demos_primitive = os.listdir(dataset_dir + demonstrations_names[i])

        for demo_primitive in demos_primitive:
            data = np.load(dataset_dir + demonstrations_names[i] + '/' + demo_primitive)
            if data.shape[0] == 1:
                # if extra dimension in demo, remove
                data = data[0]
            demos.append(data.T)
            primitive_id.append(i)

    dt = 1
    return demos, primitive_id, dt


def load_from_dict(dataset_dir, demonstrations_names):
    """
    Loads demonstrations in dictionaries
    """
    demos, primitive_id, dt = [], [], []

    # Iterate in each primitive (multi model learning)
    for i in range(len(demonstrations_names)):
        demos_primitive = os.listdir(dataset_dir + demonstrations_names[i])

        # Iterate over each demo in primitive
        for demo_primitive in demos_primitive:
            filename = dataset_dir + demonstrations_names[i] + '/' + demo_primitive
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            demos.append(data['q'].T)
            dt.append(data['delta_t'])
            primitive_id.append(i)

    return demos, primitive_id, dt
