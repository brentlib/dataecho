import pandas as pd
import numpy as np

def rake_weighting(
    survey_data: pd.DataFrame,
    id_col: str,
    target_margins: dict,
    max_iterations: int = 100,
    tolerance: float = 0.0001,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Perform survey raking manually to adjust survey weights to match known population margins.

    Args:
        survey_data (pd.DataFrame): Survey data containing the variables to weight on
        id_col (str): Column name to use as the id column
        target_margins (dict): Dictionary where keys are dimension names and values are
                               dictionaries of category proportions that sum to 1
                               Example: {
                                   'age': {'18-34': 0.3, '35-54': 0.4, '55+': 0.3},
                                   'gender': {'Male': 0.48, 'Female': 0.52}
                               }
        max_iterations (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        verbose (bool): If True, prints progress and final stats.

    Returns:
        pd.DataFrame: DataFrame with id_col and computed weights
    """
    # Check to make sure the id_col is in the survey_data
    if id_col not in survey_data.columns:
        raise ValueError(f"id_col: {id_col} not found in survey_data")
    
    # Check to make sure the id_col is unique
    if survey_data[id_col].duplicated().any():
        raise ValueError(f"id_col: {id_col} is not unique")

    # Check for missing data - filter out rows with missing data in the target_margins
    for dim_name, targets in target_margins.items():
        if dim_name not in survey_data.columns:
            raise ValueError(f"dim_name: {dim_name} not found in survey_data")

    # Validate input margins
    for dim_name, targets in target_margins.items():
        sum_value = round(sum(targets.values()), 2)
        if not np.isclose(sum_value, 1.0):
            raise ValueError(f"Target margins for dimension {dim_name} do not sum to 1.")

    # Filter out rows with missing data in any of the target dimensions
    starting_cases = len(survey_data)
    complete_cases = survey_data[list(target_margins.keys())].notna().all(axis=1)
    survey_data = survey_data[complete_cases].copy()
    total_cases = len(survey_data)

    if verbose:
        dropped_cases = starting_cases - total_cases
        print(f"Dropped {dropped_cases} cases due to missing data")
        print(f"Complete cases for weighting: {total_cases}\n")

    # Initialize weights to 1
    weights = np.ones(len(survey_data))

    target_totals = {}
    for dim, targets in target_margins.items():
        # Calculate raw values
        raw_totals = {cat: total_cases * prop for cat, prop in targets.items()}
        
        # Round down initially
        rounded_totals = {cat: int(val) for cat, val in raw_totals.items()}
        
        # Calculate how many points we need to distribute to reach total_cases
        remaining = total_cases - sum(rounded_totals.values())
        
        # Distribute remaining points based on decimal parts
        decimals = {cat: val - int(val) for cat, val in raw_totals.items()}
        categories = sorted(decimals.keys(), key=lambda x: decimals[x], reverse=True)
        
        # Add one to each category until we've distributed all remaining points
        for i in range(remaining):
            rounded_totals[categories[i]] += 1
            
        target_totals[dim] = rounded_totals

    convergence_history = []

    # Perform iterative proportional fitting
    for iteration in range(max_iterations):
        if verbose:
            print(f"Iteration {iteration + 1}")
        old_weights = weights.copy()  # Store previous weights for comparison

        for dim_name, targets in target_totals.items():
            if dim_name not in survey_data.columns:
                raise ValueError(f"Dimension {dim_name} not found in survey data")

            # Create binary indicators for each category
            indicators = pd.get_dummies(survey_data[dim_name], drop_first=False)

            # Calculate current weighted totals
            current_totals = indicators.T @ weights

            # Round small numbers to prevent floating point issues
            current_totals = pd.Series(
                np.where(current_totals < 1e-10, 1e-10, current_totals),
                index=current_totals.index
            )

            # Compute and apply adjustment factors
            for cat in targets:
                factor = targets[cat] / current_totals[cat]
                weights *= 1 + (indicators[cat].values * (factor - 1))

        # Calculate relative weight changes
        weight_changes = np.abs(1 - weights/old_weights)
        max_weight_change = np.max(weight_changes)
        convergence_history.append(max_weight_change)
        
        if max_weight_change < tolerance:
            if verbose:
                print(f"Converged in {iteration + 1} iterations (final delta: {max_weight_change:.6f})")
            break
    else:
        print(f"Did not converge within {max_iterations} iterations.")
        if verbose:
            print(f"Recent convergence history:")
            for i in range(max(0, len(convergence_history)-5), len(convergence_history)):
                print(f"Iteration {i+1}: {convergence_history[i]:.6f}")

    if weights.sum() != total_cases:
        print(f"Warning: Total weight ({weights.sum():.4f}) does not match total cases ({total_cases:.4f})")

    # Final weight statistics
    if verbose:
        min_weight = weights.min()
        max_weight = weights.max()
        avg_weight = weights.mean()

        print("\nFinal weight statistics:")
        print(f"Min weight: {min_weight:.4f}")
        print(f"Max weight: {max_weight:.4f}")
        print(f"Mean weight: {avg_weight:.4f}\n")

        # print warnings with an orange background
        if min_weight < 0.2:
            print("\033[93mWarning: Check your weights. Some weights are less than 0.2. Skewed weights might affect results. You might consider trimming your weights.\033[0m")
        if max_weight > 5.0:
            print("\033[93mWarning: Check your weights. Some weights are greater than 5.0. Skewed weights might affect results. You might consider trimming your weights.\033[0m")
        if avg_weight != 1.0:
            print("\033[93mWarning: Check your weights. The average weight is not 1.0. This typically indicates a problem with the data.\033[0m")

    return pd.DataFrame({
        id_col: survey_data[id_col],
        "weight": weights
    })