import numpy as np
from scipy.stats import norm

# Define the tilt function
def tiltweights(w, z, exponent, tilt_func):
    w[np.isnan(w)] = 0  # Replace NaNs with 0
    w0 = w.copy()

    for i in range(z.shape[1]):
        if tilt_func == 'ncdf':
            w0 *= norm.cdf(z[:, i]) ** exponent[i]
        elif tilt_func == 'exp':
            w0 *= np.exp(z[:, i] * exponent[i])
        else:
            raise ValueError("Tilt function is not defined")

    return w0 / np.nansum(w0)  # Normalize weights

# Define the system of equations
def tiltEquations(exponent, w0, z, tz, tilt_func, target):
    weight = tiltweights(w0, z, exponent, tilt_func)
    return [np.nansum(weight * tz[:, i]) - target[i] for i in range(z.shape[1])]

def tiltweights_with_constraints(w, z, exponent, tilt_func, w_min, w_max):
    w[np.isnan(w)] = 0  # Replace NaNs with 0
    w0 = w.copy()

    for i in range(z.shape[1]):
        if tilt_func == 'ncdf':
            w0 *= norm.cdf(z[:, i]) ** exponent[i]
        elif tilt_func == 'exp':
            w0 *= np.exp(z[:, i] * exponent[i])
        else:
            raise ValueError("Tilt function is not defined")

    w0 /= np.nansum(w0)  # Normalize weights

    # Apply weight constraints
    return np.clip(w0, w_min, w_max)  # Ensures all weights stay within bounds

def tiltEquations_with_constraints(exponent, w0, z, tz, tilt_func, target):
    weight = tiltweights_with_constraints(w0, z, exponent, tilt_func)
    return [np.nansum(weight * tz[:, i]) - target[i] for i in range(z.shape[1])]