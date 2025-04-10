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

def tilt_weights_with_constraints(w, z, exponent, tilt_func, config):
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
        print(i)
        if tilt_func == "ncdf":
            w0 *= norm.cdf(z[:, i]) ** exponent[i]
        elif tilt_func == "exp":
            w0 *= np.exp(z[:, i] * exponent[i])
        else:
            raise ValueError("Tilt function must be 'ncdf' or 'exp'.")

    w0 /= np.nansum(w0)  # Normalize weights

    # Apply constraints to ensure weights remain within [w_min, w_max]
    w0 = w0 / np.sum(w0)
    w0 = project_weights(w0, config)
    return w0

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

import numpy as np
from scipy.optimize import minimize

def fast_project_weights(w, lower, upper, tol=1e-12, max_iter=1000):
    """
    Projects weights onto the intersection of box constraints and the simplex (sum to 1).
    Uses a dual bisection method.
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    
    # Bisection bounds for dual variable (lambda)
    lambda_low = np.min(w - upper)
    lambda_high = np.max(w - lower)

    for _ in range(max_iter):
        lambda_mid = (lambda_low + lambda_high) / 2.0
        projected = np.clip(w - lambda_mid, lower, upper)
        total = np.sum(projected)

        if abs(total - 1.0) < tol:
            return projected
        elif total < 1.0:
            lambda_high = lambda_mid
        else:
            lambda_low = lambda_mid

    # If we reach here, return last projection anyway (approx)
    print("Warning: fast projection did not fully converge")
    return projected

def project_weights(weights, config):
    capacity_ratio = config["Capacity Ratio"]
    max_weight = config["Max Weight"]
    stock_bound = config["Stock Bound"]

    # Compute bounds per stock
    upper_bounds = np.minimum(
        capacity_ratio * weights,
        np.minimum(weights + stock_bound, max_weight)
    )
    print(max(upper_bounds/weights))
    lower_bounds = np.zeros_like(weights)

    return fast_project_weights(weights, lower_bounds, upper_bounds)
