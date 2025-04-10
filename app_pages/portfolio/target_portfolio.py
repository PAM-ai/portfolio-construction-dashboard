import pandas as pd
import numpy as np
from scipy import optimize
import app_pages.portfolio.tilting_functions as tilts

def capping(weights, lower_bounds, upper_bounds, max_iter=100, tolerance=1e-12):
    """
    Adjusts weights to comply with lower and upper bounds while maintaining sum-to-one constraint.
    
    Parameters:
        weights (np.array): Initial weights to adjust
        lower_bounds (np.array): Lower bounds for each weight
        upper_bounds (np.array): Upper bounds for each weight
        max_iter (int): Maximum number of iterations
        tolerance (float): Tolerance for bound satisfaction
    
    Returns:
        np.array: Adjusted weights that satisfy the bounds, or None if no solution found
    """
    i = 0
    w0 = weights.copy()
    
    # Check if lower bounds exceed upper bounds
    if (lower_bounds > upper_bounds).any():
        print("Error: Lower bounds exceed upper bounds")
        return None
    
    while (np.less(w0 - lower_bounds, np.ones(len(w0)) * (-tolerance)).any() or 
           np.less(upper_bounds - w0, np.ones(len(w0)) * (-tolerance)).any()):
        
        # Identify weights that violate constraints
        cap = np.logical_or(w0 < lower_bounds, w0 > upper_bounds)
        
        # Cap weights at their bounds
        w0[w0 < lower_bounds] = lower_bounds[w0 < lower_bounds]
        w0[w0 > upper_bounds] = upper_bounds[w0 > upper_bounds]
        
        # Rescale uncapped weights to maintain sum-to-one
        if cap.all():
            # If all weights are capped, renormalize everything
            w0 = w0 / np.sum(w0)
        else:
            # Only rescale uncapped weights
            w0[~cap] = (w0[~cap] * 
                       (1 - np.sum(w0[cap])) / 
                       np.sum(w0[~cap]))
        
        i += 1
        if i >= max_iter:
            print(f"Warning: Maximum iterations ({max_iter}) reached")
            break
    
    return w0

def verify_and_adjust_exposures(weights, factor_scores, target_exposures, tilt_func, original_weights, 
                               config, max_attempts=3, tolerance=1e-4):
    """
    Verify that capped weights still meet target exposures, and adjust if needed.
    For reduction targets, exposures below the target are considered acceptable.
    
    Parameters:
        weights (np.array): Capped portfolio weights
        factor_scores (np.array): Factor scores for each asset
        target_exposures (np.array): Target exposures for each factor
        tilt_func (str): Tilting function to use ('exp' or 'ncdf')
        original_weights (np.array): Original benchmark weights
        config (dict): Configuration with capacity ratio, max weight, and stock bound
        max_attempts (int): Maximum number of adjustment attempts
        tolerance (float): Tolerance for exposure satisfaction
    
    Returns:
        np.array: Weights that satisfy both constraints and exposures
    """
    
    # Calculate current exposures
    current_exposures = np.dot(weights, factor_scores)
    
    # Check if exposures are acceptable:
    # For reduction targets, below target is fine (more reduction than required)
    # Otherwise, must be within tolerance
    exposure_diff = current_exposures - target_exposures
    exposure_abs_diff = np.abs(exposure_diff)
    
    # For reduction targets, we consider exposures acceptable if:
    # 1. They're within tolerance of the target, OR
    # 2. They're below the target (which means exceeding the reduction goal)
    is_satisfied = np.logical_or(
        exposure_abs_diff <= tolerance,  # Within tolerance
        exposure_diff < 0                # Below target (for reduction targets)
    )
    
    if np.all(is_satisfied):
        return weights
    
    # Only consider differences for exposures that aren't satisfied
    unsatisfied_idx = ~is_satisfied
    
    # Define weight bounds
    capacity_ratio = config["Capacity Ratio"]
    max_weight = config["Max Weight"]
    stock_bound = config["Stock Bound"]
    
    upper_bounds = np.minimum(
        capacity_ratio * original_weights,
        np.minimum(original_weights + stock_bound, max_weight * np.ones_like(original_weights))
    )
    lower_bounds = np.zeros_like(original_weights)
    
    # Standardize factor scores for tilting
    norm_factor_scores = factor_scores.copy()
    if norm_factor_scores.ndim == 1:
        norm_factor_scores = norm_factor_scores.reshape(-1, 1)
    
    # If multiple factors, convert to 2D array
    if len(target_exposures) > 1 and norm_factor_scores.ndim == 1:
        norm_factor_scores = norm_factor_scores.reshape(-1, 1)
    
    # Standardize scores if they aren't already
    if norm_factor_scores.ndim > 1:
        for i in range(norm_factor_scores.shape[1]):
            col = norm_factor_scores[:, i]
            norm_factor_scores[:, i] = (col - np.mean(col)) / np.std(col)
    else:
        norm_factor_scores = (norm_factor_scores - np.mean(norm_factor_scores)) / np.std(norm_factor_scores)
    
    # Try to adjust exposures while respecting constraints
    best_weights = weights.copy()
    min_error_score = np.sum(exposure_abs_diff[unsatisfied_idx])
    
    for attempt in range(max_attempts):
        # Find new tilts starting from current weights
        exponent_init = np.zeros(len(target_exposures))
        
        result = optimize.root(
            tilts.tilt_equations,
            exponent_init,
            args=(weights, norm_factor_scores, factor_scores, tilt_func, target_exposures),
            method="hybr",
            tol=1e-6
        )
        
        if not result.success:
            continue
        
        # Apply new tilts
        exponent = result.x
        tilted_weights = tilts.tilt_weights(weights, norm_factor_scores, exponent, tilt_func)
        
        # Cap weights again
        capped_weights = capping(tilted_weights, lower_bounds, upper_bounds)
        if capped_weights is None:
            continue
        
        # Check exposures
        new_exposures = np.dot(capped_weights, factor_scores)
        new_diff = new_exposures - target_exposures
        new_abs_diff = np.abs(new_diff)
        
        # Check if exposures are acceptable with the same criteria as above
        new_is_satisfied = np.logical_or(
            new_abs_diff <= tolerance,  # Within tolerance
            new_diff < 0                # Below target
        )
        
        # Only consider unsatisfied exposures for error calculation
        unsatisfied_exposures = ~new_is_satisfied
        if unsatisfied_exposures.any():
            new_error_score = np.sum(new_abs_diff[unsatisfied_exposures])
        else:
            new_error_score = 0
        
        
        
        # Keep the best result
        if new_error_score < min_error_score or np.all(new_is_satisfied):
            best_weights = capped_weights
            min_error_score = new_error_score
            
            # If all exposures are satisfied, we're done
            if np.all(new_is_satisfied):
                
                return best_weights
    
    # Check if best weights satisfy our criteria
    final_exposures = np.dot(best_weights, factor_scores)
    final_diff = final_exposures - target_exposures
    final_is_satisfied = np.logical_or(
        np.abs(final_diff) <= tolerance,
        final_diff < 0
    )

    return best_weights

def solve_with_capping(weights, review_subset, targets_subset, sustainable_factors, tilt_func, config, xtol=1e-6):
    """
    Solve for portfolio weights that satisfy sustainable constraints, then apply weight constraints
    and verify exposures are still met.
    
    Parameters:
        weights (np.array): Initial portfolio weights
        review_subset (pd.DataFrame): Subset of data for the review date
        targets_subset (pd.DataFrame): Target values for sustainable factors
        sustainable_factors (list or str): List of sustainable factors (or single factor as string)
        tilt_func (str): Tilting function to use ('exp' or 'ncdf')
        config (dict): Configuration with capacity ratio, max weight, stock bound
        xtol (float): Solver tolerance
    
    Returns:
        np.array: Portfolio weights that satisfy both factor targets and weight constraints
    """
    
    # Ensure factors are always in list form
    if isinstance(sustainable_factors, str):
        sustainable_factors = [sustainable_factors]

    # Extract and clean factor scores
    tscores = review_subset[sustainable_factors].values
    if tscores.ndim == 1:
        tscores = tscores.reshape(-1, 1)

    # Standardize factor scores for optimization
    zscores = tscores.copy()
    if zscores.ndim > 1:
        for i in range(zscores.shape[1]):
            zscores[:, i] = (zscores[:, i] - np.mean(zscores[:, i])) / np.std(zscores[:, i])
    else:
        zscores = (zscores - np.mean(zscores)) / np.std(zscores)

    # Extract target values
    target_columns = [f"TargetValue_{factor}" for factor in sustainable_factors]
    targets = targets_subset[target_columns].values.flatten()

    # Phase 1: Find weights that satisfy factor targets
    exponent_init = np.zeros(zscores.shape[1] if zscores.ndim > 1 else 1)
    
    result = optimize.root(
        tilts.tilt_equations,
        exponent_init,
        args=(weights, zscores, tscores, tilt_func, targets),
        method="hybr",
        tol=xtol
    )
    
    if not result.success:
        print(f"Solver did not converge: {result.message}")
        return None
    
    # Get tilted weights that satisfy factor targets
    exponent = result.x
    tilted_weights = tilts.tilt_weights(weights, zscores, exponent, tilt_func)
    
    # Verify factor targets are hit
    factor_exposures = np.dot(tilted_weights, tscores)
    
    # Phase 2: Apply weight constraints
    capacity_ratio = config["Capacity Ratio"]
    max_weight = config["Max Weight"]
    stock_bound = config["Stock Bound"]
    
    upper_bounds = np.minimum(
        capacity_ratio * weights,
        np.minimum(weights + stock_bound, np.ones_like(weights) * max_weight)
    )
    lower_bounds = np.zeros_like(weights)
    
    # Apply weight capping
    capped_weights = capping(tilted_weights, lower_bounds, upper_bounds)
    if capped_weights is None:
        print("Failed to apply weight constraints. Using uncapped weights.")
        return tilted_weights
    
    # Phase 3: Verify and adjust exposures if needed
    final_weights = verify_and_adjust_exposures(
        capped_weights, 
        tscores, 
        targets, 
        tilt_func, 
        weights, 
        config
    )
    
    return final_weights

def get_weights(review_dates, review_data, sustainable_factors, excluded_subsectors, targets_df, tilt_func, config, xtol=1e-6):
    """
    Computes optimized portfolio weights that satisfy sustainable constraints.

    Parameters:
        review_dates (list): List of dates for portfolio reviews.
        review_data (pd.DataFrame): DataFrame with stock weights and factor scores.
        sustainable_factors (list): List of sustainability factors.
        excluded_subsectors (list): Sub-sectors to exclude from the portfolio.
        targets_df (pd.DataFrame): DataFrame with target exposures.
        tilt_func (str): Tilting function to use ('ncdf' or 'exp').
        config (dict): Configuration with capacity ratio, max weight, and stock bound.
        xtol (float): Solver tolerance.

    Returns:
        pd.DataFrame: Final adjusted portfolio weights.
    """

    # Filter out excluded subsectors
    review_data = review_data[~review_data["Sub-Sector"].isin(excluded_subsectors)].copy()
    review_data["Review Date"] = pd.to_datetime(review_data["Review Date"])

    tilted_weights = []
    reached_dict = {factor: [] for factor in sustainable_factors} # Initialize reached_dict for each factor to store achieved values

    for date in review_dates:
        print(f"\n--- Review date: {date} ---")

        # Prepare data for the current review date
        review_subset = review_data[review_data["Review Date"] == date]
        targets_subset = targets_df[targets_df["Review Date"] == date]
        weights = review_subset["Weight"].values

        # Constraints (for future extension)
        capacity_ratio = config["Capacity Ratio"]
        max_weight = config["Max Weight"]
        stock_bound = config["Stock Bound"]

        weights_upper_bounds = np.minimum(capacity_ratio * weights, np.minimum(weights + stock_bound, max_weight))
        weights_lower_bounds = np.zeros(len(weights))

        # Initialize with the largest reduction factor
        active_factors = [targets_subset["LargestReductionItem"].iloc[0]]
        solved = False
        attempts = 0
        max_attempts = len(sustainable_factors)  # Safe stop condition

        while not solved and attempts < max_attempts:
            print(f"\nAttempt {attempts + 1} with factors: {active_factors}")

            tilted_weights_date = solve_with_capping(
                weights,
                review_subset,
                targets_subset,
                active_factors,
                tilt_func,
                config,
                xtol
            )

            if tilted_weights_date is None:
                print(f"Solver failed on attempt {attempts + 1}. Exiting loop.")
                break  # Exit early if solver fails

            # Check achieved exposures
            achieved_exposure = tilted_weights_date @ review_subset[sustainable_factors].values
            bmk_intensity = np.array([targets_subset[f"ReviewIntensity_{factor}"].iloc[0] for factor in sustainable_factors])
            upper_bounds = np.array([targets_subset[f"TargetValue_{factor}"].iloc[0] for factor in sustainable_factors])
            lower_bounds = np.zeros(len(sustainable_factors))

            # Identify new violations
            violations = []
            for idx, factor in enumerate(sustainable_factors):
                exposure = achieved_exposure[idx]
                if exposure > upper_bounds[idx] or exposure < lower_bounds[idx]:
                    if factor not in active_factors:
                        violations.append(factor)

            if not violations:
                for i, factor in enumerate(sustainable_factors):
                    reached_dict[factor].append(achieved_exposure[i])

                # Break the loop if no violations are found    
                solved = True
            else:
                active_factors.extend(violations)
                attempts += 1

        if tilted_weights_date is not None:
            tilted_weights.extend(tilted_weights_date)
        else:
            print(f"Skipping date {date} due to solver failure.")

    

    # Target and achieved exposures
    achieved_targets = targets_df.copy()
    for factor in sustainable_factors:
        achieved_targets[f"Achieved{factor}"] = reached_dict[factor]
    
    # Prepare final output weights DataFrame
    index_weights = review_data.copy()
    index_weights["IndexWeights"] = tilted_weights
    index_weights = index_weights[["Review Date", "Symbol", "IndexWeights"]]

    return index_weights, achieved_targets

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
        targets_df[f"AnnualRate_{factor}"] = rate
        targets_df[f"TargetLevelFromBaseBenchmark{factor}"] = np.where(targets_df[f"AnnualRate_{factor}"] != 0, targets_df[f"TargetValue_{factor}"] / targets_df[f"BaseIntensity_{factor}"], targets_df[f"TargetLevel_{factor}"])

    # Find the column name with the minimum value for each row
    targets_df['LargestReductionItem'] = targets_df[[f"TargetLevelFromBaseBenchmark{factor}" for factor in sustainable_factors]].idxmin(axis=1)
    targets_df["LargestReductionItem"] = targets_df["LargestReductionItem"].str.replace("TargetLevelFromBaseBenchmark", "")

    return targets_df

def get_sector_targets(review_data, exclusions, config, relax_value=0):
    """
    Calculate sector-specific lower and upper targets.
    
    Parameters:
    - review_data: DataFrame, historical review data containing stock weights and factor exposures.
    - exclusions: list, sectors to exclude from the portfolio.
    - config: dict, configuration parameters for the portfolio.
    
    Returns:
    - sector_targets: DataFrame, with sector lower and upper bounds.
    """
    review_dates = sorted(review_data["Review Date"].unique())
    sectors_targets = pd.DataFrame({"Review Date": review_dates})
    
    # Filter out excluded subsectors
    filtered_data = review_data[~review_data["Sub-Sector"].isin(exclusions)].copy()
    
    sectors = filtered_data["Sector"].unique()
    
    sector_p = config.get("SectorP") 
    sector_q = config.get("SectorQ")
    
    for sector in sectors:
        sector_weights = []
        lower_bounds = []
        upper_bounds = []
        
        for date in review_dates:
            review_subset = filtered_data[filtered_data["Review Date"] == date]
            sector_weight = review_subset[review_subset["Sector"] == sector]["Weight"].sum()
            lower_bound = max(0, sector_weight * (1 - sector_p) - sector_q - relax_value)
            upper_bound = min(1, sector_weight * (1 + sector_p) + sector_q + relax_value)
            
            sector_weights.append(sector_weight)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        
        # Create columns
        sectors_targets[f"Sector_{sector}"] = sector_weights
        sectors_targets[f"TargetValue_{sector}_lower"] = lower_bounds
        sectors_targets[f"TargetValue_{sector}_upper"] = upper_bounds
    
    return sectors_targets
