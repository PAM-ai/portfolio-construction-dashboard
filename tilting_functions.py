import numpy as np
from scipy.stats import norm

def tilt_weights(w, z, exponent, tilt_func):
    """
    Applies a tilting function to portfolio weights based on factor scores.

    Parameters:
        w (np.array): Initial portfolio weights.
        z (np.array): Standardized factor scores (shape: assets × factors).
        exponent (np.array): Exponent vector applied to the tilting function.
        tilt_func (str): Tilting method ('ncdf' or 'exp').

    Returns:
        np.array: Adjusted portfolio weights normalized to sum to 1.
    """
    w[np.isnan(w)] = 0  # Replace NaNs with 0 
    w0 = w.copy()

    for i in range(z.shape[1]):
        if tilt_func == "ncdf":
            w0 *= norm.cdf(z[:, i]) ** exponent[i]
        elif tilt_func == "exp":
            w0 *= np.exp(z[:, i] * exponent[i])
        else:
            raise ValueError("Tilt function must be 'ncdf' or 'exp'.")

    return w0 / np.nansum(w0)  # Normalize weights

def tilt_equations(exponent, w0, z, tz, tilt_func, target):
    """
    Defines the system of equations for solving the tilt exponent.

    Parameters:
        exponent (np.array): Exponent vector.
        w0 (np.array): Initial portfolio weights.
        z (np.array): Standardized factor scores.
        tz (np.array): Factor exposure matrix.
        tilt_func (str): Tilting function ('ncdf' or 'exp').
        target (np.array): Target values for factor exposures.

    Returns:
        list: Differences between achieved and target factor exposures.
    """
    weight = tilt_weights(w0, z, exponent, tilt_func)
    return [np.nansum(weight * tz[:, i]) - target[i] for i in range(z.shape[1])]

def tilt_weights_with_constraints(w, z, exponent, tilt_func, w_min, w_max):
    """
    Computes tilted weights while enforcing portfolio constraints.

    Parameters:
        w (np.array): Initial portfolio weights.
        z (np.array): Standardized factor scores.
        exponent (np.array): Exponent vector applied to the tilting function.
        tilt_func (str): Tilting method ('ncdf' or 'exp').
        w_min (np.array or float): Minimum allowed weight for each asset.
        w_max (np.array or float): Maximum allowed weight for each asset.

    Returns:
        np.array: Adjusted and constrained portfolio weights.
    """
    w = np.nan_to_num(w)  # Replace NaNs with 0
    w0 = w.copy()

    for i in range(z.shape[1]):
        if tilt_func == "ncdf":
            w0 *= norm.cdf(z[:, i]) ** exponent[i]
        elif tilt_func == "exp":
            w0 *= np.exp(z[:, i] * exponent[i])
        else:
            raise ValueError("Tilt function must be 'ncdf' or 'exp'.")

    w0 /= np.nansum(w0)  # Normalize weights

    # Apply constraints to ensure weights remain within [w_min, w_max]
    w0 = np.clip(w0, w_min, w_max)
    w0 /= np.sum(w0)
    return  w0


def tilt_equations_with_constraints(exponent, w0, z, tz, tilt_func, target, w_min, w_max):
    """
    Defines the system of equations for solving the tilt exponent under weight constraints.

    Parameters:
        exponent (np.array): Exponent vector.
        w0 (np.array): Initial portfolio weights.
        z (np.array): Standardized factor scores.
        tz (np.array): Factor exposure matrix.
        tilt_func (str): Tilting function ('ncdf' or 'exp').
        target (np.array): Target values for factor exposures.
        w_min (np.array or float): Minimum allowed weight for each asset.
        w_max (np.array or float): Maximum allowed weight for each asset.

    Returns:
        list: Differences between achieved and target factor exposures.
    """
    weight = tilt_weights_with_constraints(w0, z, exponent, tilt_func, w_min, w_max)
    return [np.nansum(weight * tz[:, i]) - target[i] for i in range(z.shape[1])]
