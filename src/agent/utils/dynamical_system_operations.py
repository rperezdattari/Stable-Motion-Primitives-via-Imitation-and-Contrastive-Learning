"""""""""""""""""""""""""""""""""""""""""""""""
        Dynamical System useful function
"""""""""""""""""""""""""""""""""""""""""""""""


def normalize_state(state, x_min, x_max):
    """
    Normalize state
    """
    state = (((state - x_min) / (x_max - x_min)) - 0.5) * 2

    return state


def get_derivative_normalized_state(dx, x_min, x_max):
    """
    Computes derivative of normalized state from derivative of unnormalized state
    """
    dx_x_norm = (2 * dx) / (x_max - x_min)
    return dx_x_norm


def denormalize_state(state, x_min, x_max):
    """
    Denormalize state
    """
    state = ((state / 2) + 0.5) * (x_max - x_min) + x_min
    return state


def denormalize_derivative(dx_t, max_state_derivative):
    """
    Denormalize state derivative
    """
    dx_t_denormalized = dx_t * max_state_derivative
    return dx_t_denormalized


def normalize_derivative(dx_t, max_state_derivative):
    """
    Normalize state derivative
    """
    dx_t_normalized = dx_t / max_state_derivative
    return dx_t_normalized


def euler_integration(x_t, dx_t, delta_t):
    """
    Euler integration and get x_{t+1}
    """
    x_t_next = x_t + dx_t * delta_t
    return x_t_next


def euler_diff(x_t_next, x_t, delta_t):
    """
    Euler differentiation and get dx_t
    """
    dx = (x_t_next - x_t) / delta_t
    return dx