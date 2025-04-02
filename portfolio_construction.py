import pandas as pd
import numpy as np
from scipy import optimize
import tilting_functions as tilts
from importlib import reload

reload(tilts)

def tiltResiduals_with_constraints(exponent, w0, z, tz, tilt_func, lower_bounds, upper_bounds, w_min, w_max):
    weight = tilts.tiltweights_with_constraints(w0, z, exponent, tilt_func, w_min, w_max)
    actual_values = weight @ tz
    residuals = np.maximum(0, lower_bounds - actual_values) + np.maximum(0, actual_values - upper_bounds)
    return residuals

def get_targets(review_data, sustainable_factors, targets, annual_trajectory_rate, reviews_per_year):

    review_dates = list(np.sort(review_data["Review Date"].unique()))
    targets_df = pd.DataFrame(review_dates, columns=["Review Date"])

    for i, factor in enumerate(sustainable_factors):
        
        base_date = review_dates[0]
        review_base = review_data[review_data["Review Date"]==base_date]
        base_factor_value = review_base["Weight"].values @ review_base[factor].values
        
        fixed_target = targets[i]
        rate = annual_trajectory_rate[i]
        trajectory_rate = ((1 + rate)**(1/reviews_per_year)) - 1

        bmk_intensities = []
        target_levels = []
        target_values = []
        
        for period, date in enumerate(review_dates):
    
            review_subset = review_data[review_data["Review Date"]==date]
            bmk_intensity = review_subset["Weight"].values @ review_subset[factor].values
            target_level = ((1 + trajectory_rate)**(period)) * (1 + fixed_target)
            target_value = target_level * base_factor_value if rate !=0 else target_level * bmk_intensity

            # Append values to lists
            bmk_intensities.append(bmk_intensity)
            target_levels.append(target_level)
            target_values.append(target_value)
            
        targets_df[f"BaseIntensity{factor}"] = base_factor_value
        targets_df[f"ReviewIntensity{factor}"] = bmk_intensities
        targets_df[f"TargetLevel{factor}"] = target_levels
        targets_df[f"TargetValue{factor}"] = target_values
        
    return targets_df


def get_weights(review_dates, review_data, sustainable_factors, excluded_subsectors, targets_df, tilt_func, config):
    
    review_data = review_data[~review_data["Sub-Sector"].isin(excluded_subsectors)]
    review_data["Review Date"] = pd.to_datetime(review_data["Review Date"])
    
    tilted_weights = []
    benchmark_dict = {factor: [] for factor in sustainable_factors}
    target_dict = {factor: [] for factor in sustainable_factors}
    reached_dict = {factor: [] for factor in sustainable_factors}

    for period, date in enumerate(review_dates):
        
        print(f"Review date: {date}")
        
        # Select subset that matches the review date
        review_subset = review_data[review_data["Review Date"] == date]
        targets_subset = targets_df[targets_df["Review Date"] == date]

        # Create arrays 
        weights = np.array(review_subset["Weight"])
        tscores = review_subset[sustainable_factors].values
        zscores = (tscores - tscores.mean(axis=0)) / tscores.std(axis=0)
        
        # Compute benchmark and upper bounds
        bmk_intensity = weights @ tscores  # Array of benchmark intensities per factor
        upper_bounds = np.array([targets_subset[f"TargetValue{factor}"].iloc[0] for factor in sustainable_factors] ) # Array of target values per factor
        lower_bounds = np.zeros(tscores.shape[1])  # Define lower bounds correctly

        # Store benchmark and target values per factor
        for i, factor in enumerate(sustainable_factors):
            benchmark_dict[factor].append(bmk_intensity[i])
            target_dict[factor].append(upper_bounds[i])

        # Weights constraints
        capacity_ratio = config.get("Capacity Ratio")
        max_weight = config.get("Max Weight")
        stock_bound = config.get("Stock Bound")

        weights_upper_bounds = np.minimum(capacity_ratio * weights, np.minimum(weights + stock_bound, max_weight))
        weights_lower_bounds = np.zeros(weights.shape[0])

        # Solve for exponent vectors and get tilted weights
        exponent_init = np.zeros(zscores.shape[1])  # Initial guess

        exponent_solution = optimize.least_squares(
            tiltResiduals_with_constraints,
            exponent_init,
            args=(weights, zscores, tscores, tilt_func, lower_bounds, upper_bounds, weights_lower_bounds, weights_upper_bounds),
            method="trf",
            bounds=(-100, 100) 
        )
        
        if exponent_solution.success:
            exponent = exponent_solution.x
            tilted_weights_date = tilts.tiltweights_with_constraints(weights, zscores, exponent, tilt_func, weights_lower_bounds, weights_upper_bounds)

            print("Solved exponent:", exponent)
            print(f"Bmk: {weights @ tscores} - Target: {upper_bounds} - Results: {tilted_weights_date @ tscores}")
            print(f"Maximum Capacity Ratio: {max(tilted_weights_date / weights)}")
            print(f"Maximum Weights: {max(tilted_weights_date)}")
            
            # Append tilted weights to the list
            tilted_weights.extend(tilted_weights_date)

            # Store achieved values per factor
            achieved_intensity = tilted_weights_date @ tscores  # Array of achieved values per factor
            for i, factor in enumerate(sustainable_factors):
                reached_dict[factor].append(achieved_intensity[i])
        else:
            print("Solution did not converge:", exponent_solution.message)
            for factor in sustainable_factors:
                reached_dict[factor].append(np.nan)  # Store NaN if no solution

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

