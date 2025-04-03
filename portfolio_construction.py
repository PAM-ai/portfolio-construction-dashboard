import pandas as pd
import numpy as np
from scipy import optimize
import tilting_functions as tilts
from importlib import reload

reload(tilts)

def tilt_residuals_with_constraints(exponent, w0, z, tz, tilt_func, lower_bounds, upper_bounds, w_min, w_max):
    """
    Computes the residuals for the optimization problem when applying tilt constraints.

    Parameters:
        exponent (np.array): Exponent vector applied to the tilting function.
        w0 (np.array): Initial portfolio weights.
        z (np.array): Standardized factor scores.
        tz (np.array): Factor exposure matrix.
        tilt_func (str): Type of tilting function ('ncdf' or 'exp').
        lower_bounds (np.array): Lower bounds for factor exposure constraints.
        upper_bounds (np.array): Upper bounds for factor exposure constraints.
        w_min (float): Minimum allowed weight for assets.
        w_max (float): Maximum allowed weight for assets.

    Returns:
        np.array: Residuals indicating constraint violations.
    """
    weight = tilts.tilt_weights_with_constraints(w0, z, exponent, tilt_func, w_min, w_max)
    actual_values = weight @ tz  # Compute achieved factor exposures
    residuals = np.maximum(0, lower_bounds - actual_values) + np.maximum(0, actual_values - upper_bounds)
    
    return residuals

def get_targets(review_data, sustainable_factors, targets, annual_trajectory_rate, reviews_per_year):
    """
    Computes target factor exposures over time, accounting for benchmark evolution and trajectory adjustments.

    Parameters:
        review_data (pd.DataFrame): Historical review data containing stock weights and factor exposures.
        sustainable_factors (list): List of factors used for sustainable tilting.
        targets (list): Target percentage reductions for each factor.
        annual_trajectory_rate (list): Annual reduction rates for each factor.
        reviews_per_year (int): Number of reviews per year.

    Returns:
        pd.DataFrame: Dataframe containing base intensity, review intensity, and target levels for each factor.
    """
    review_dates = sorted(review_data["Review Date"].unique())
    targets_df = pd.DataFrame({"Review Date": review_dates})

    for i, factor in enumerate(sustainable_factors):
        base_date = review_dates[0]
        review_base = review_data[review_data["Review Date"] == base_date]
        base_factor_value = np.dot(review_base["Weight"], review_base[factor])

        fixed_target = targets[i]
        rate = annual_trajectory_rate[i]
        trajectory_rate = ((1 + rate) ** (1 / reviews_per_year)) - 1

        bmk_intensities, target_levels, target_values = [], [], []
        
        for period, date in enumerate(review_dates):
            review_subset = review_data[review_data["Review Date"] == date]
            bmk_intensity = np.dot(review_subset["Weight"], review_subset[factor])
            target_level = ((1 + trajectory_rate) ** period) * (1 + fixed_target)
            target_value = base_factor_value * target_level if rate != 0 else bmk_intensity * target_level

            bmk_intensities.append(bmk_intensity)
            target_levels.append(target_level)
            target_values.append(target_value)

        targets_df[f"BaseIntensity_{factor}"] = base_factor_value
        targets_df[f"ReviewIntensity_{factor}"] = bmk_intensities
        targets_df[f"TargetLevel_{factor}"] = target_levels
        targets_df[f"TargetValue_{factor}"] = target_values

    return targets_df

def get_weights(review_dates, review_data, sustainable_factors, excluded_subsectors, targets_df, tilt_func, config):
    """
    Computes optimized portfolio weights that satisfy sustainable constraints.

    Parameters:
        review_dates (list): List of dates when portfolio reviews occur.
        review_data (pd.DataFrame): Dataframe containing stock weights and factor scores.
        sustainable_factors (list): List of factors used for tilting.
        excluded_subsectors (list): List of subsectors to be excluded.
        targets_df (pd.DataFrame): Dataframe with computed factor targets.
        tilt_func (str): Type of tilting function to apply ('ncdf' or 'exp').
        config (dict): Configuration dictionary containing weight constraints.

    Returns:
        pd.DataFrame: Adjusted portfolio weights.
        pd.DataFrame: Dataframe with achieved factor levels compared to targets.
    """
    # Filter out excluded subsectors
    review_data = review_data[~review_data["Sub-Sector"].isin(excluded_subsectors)].copy()
    review_data["Review Date"] = pd.to_datetime(review_data["Review Date"])

    tilted_weights = []
    benchmark_dict, target_dict, reached_dict = {}, {}, {}

    for factor in sustainable_factors:
        benchmark_dict[factor], target_dict[factor], reached_dict[factor] = [], [], []

    for period, date in enumerate(review_dates):
        print(f"Review date: {date}")

        # Subset data for the given review date
        review_subset = review_data[review_data["Review Date"] == date]
        targets_subset = targets_df[targets_df["Review Date"] == date]

        # Extract weights and factor scores
        weights = review_subset["Weight"].values
        tscores = review_subset[sustainable_factors].values
        zscores = (tscores - tscores.mean(axis=0)) / tscores.std(axis=0)

        # Compute benchmark exposures and targets
        bmk_intensity = np.dot(weights, tscores)
        upper_bounds = np.array([targets_subset[f"TargetValue_{factor}"].iloc[0] for factor in sustainable_factors])
        lower_bounds = np.zeros(len(sustainable_factors))

        # Store benchmark and target values
        for i, factor in enumerate(sustainable_factors):
            benchmark_dict[factor].append(bmk_intensity[i])
            target_dict[factor].append(upper_bounds[i])

        # Define weight constraints
        capacity_ratio = config["Capacity Ratio"]
        max_weight = config["Max Weight"]
        stock_bound = config["Stock Bound"]

        weights_upper_bounds = np.minimum(capacity_ratio * weights, np.minimum(weights + stock_bound, max_weight))
        weights_lower_bounds = np.zeros(len(weights))

        # Solve for exponent vector
        exponent_init = np.zeros(zscores.shape[1])

        exponent_solution = optimize.least_squares(
            tilt_residuals_with_constraints,
            exponent_init,
            args=(weights, zscores, tscores, tilt_func, lower_bounds, upper_bounds, weights_lower_bounds, weights_upper_bounds),
            method="trf",
            bounds=(-100, 100)
        )

        if exponent_solution.success:
            exponent = exponent_solution.x
            tilted_weights_date = tilts.tilt_weights_with_constraints(weights, zscores, exponent, tilt_func, weights_lower_bounds, weights_upper_bounds)

            print(f"Solved exponent: {exponent}")
            print(f"Bmk: {bmk_intensity} - Target: {upper_bounds} - Achieved: {np.dot(tilted_weights_date, tscores)}")
            print(f"Max Capacity Ratio: {max(tilted_weights_date / weights)}")
            print(f"Max Weight: {max(tilted_weights_date)}")

            tilted_weights.extend(tilted_weights_date)

            achieved_intensity = np.dot(tilted_weights_date, tscores)
            for i, factor in enumerate(sustainable_factors):
                reached_dict[factor].append(achieved_intensity[i])
        else:
            print("Solution did not converge:", exponent_solution.message)
            for factor in sustainable_factors:
                reached_dict[factor].append(np.nan)

    # Create final DataFrame
    achieved_targets_df = pd.DataFrame({"Review Date": review_dates})
    for factor in sustainable_factors:
        achieved_targets_df[f"Benchmark {factor}"] = benchmark_dict[factor]
        achieved_targets_df[f"Target {factor}"] = target_dict[factor]
        achieved_targets_df[f"Achieved {factor}"] = reached_dict[factor]

    index_weights = review_data.copy()
    index_weights["IndexWeights"] = tilted_weights
    index_weights = index_weights[["Review Date", "Symbol", "IndexWeights"]]

    return index_weights, achieved_targets_df

