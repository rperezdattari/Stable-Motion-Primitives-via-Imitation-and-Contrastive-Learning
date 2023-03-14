from evaluation.evaluate_2d_o1 import Evaluate2DO1
from evaluation.evaluate_2d_o2 import Evaluate2DO2
from evaluation.evaluate_3d import Evaluate3D
from evaluation.evaluate_nd import EvaluateND


def evaluator_init(learner, data, params, verbose=True):
    """
    Selects and initializes evaluation class
    """
    if params.workspace_dimensions == 2 and params.dynamical_system_order == 1:
        return Evaluate2DO1(learner, data, params, verbose)
    elif params.workspace_dimensions == 2 and params.dynamical_system_order == 2:
        return Evaluate2DO2(learner, data, params, verbose)
    elif params.workspace_dimensions == 3:
        return Evaluate3D(learner, data, params, verbose)
    else:
        return EvaluateND(learner, data, params, verbose)
