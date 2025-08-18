"""
Data Generation Module

This module handles generation of approximate integer test data and domain rescaling
for polynomial approximation analysis.
"""

import numpy as np
from typing import List, Tuple
import config


def generate_approximate_integers(target_value: float, epsilon: float, 
                                num_points: int, random_seed: int = None) -> np.ndarray:
    """
    Generate approximate integer values around a target value, ensuring bounds [0, MAX_VAL].
    
    Args:
        target_value: The target integer value to approximate
        epsilon: The tolerance range (±epsilon around target)
        num_points: Number of points to generate
        random_seed: Optional seed for reproducibility
        
    Returns:
        Array of approximate values constrained to [0, MAX_VAL]
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if config.EXACTLY_EPSILON:
        # Generate exactly 2 points: one at +epsilon, one at -epsilon
        # Handle boundary cases
        if target_value == 0:
            # For target 0, only +epsilon (can't go negative)
            raw_values = np.array([target_value + epsilon])
        elif target_value == config.MAX_VAL:
            # For target MAX_VAL, only -epsilon (can't exceed MAX_VAL)
            raw_values = np.array([target_value - epsilon])
        else:
            # Normal case: one at +epsilon, one at -epsilon
            raw_values = np.array([target_value + epsilon, target_value - epsilon])
    else:
        # Original behavior: generate uniform random values in [-epsilon, +epsilon]
        random_offsets = np.random.uniform(-epsilon, epsilon, num_points)
        raw_values = target_value + random_offsets
        
        # Ensure values stay within [0, MAX_VAL] bounds
        # For boundary values (0 and MAX_VAL), only generate noise in valid direction
        if target_value == 0:
            # For target 0, only allow positive noise (keep values >= 0)
            random_offsets = np.random.uniform(0, epsilon, num_points)
            raw_values = target_value + random_offsets
        elif target_value == config.MAX_VAL:
            # For target MAX_VAL, only allow negative noise (keep values <= MAX_VAL)
            random_offsets = np.random.uniform(-epsilon, 0, num_points)
            raw_values = target_value + random_offsets
    
    # Clamp all values to [0, MAX_VAL] as final safety check
    return np.clip(raw_values, 0, config.MAX_VAL)


def rescale_to_unit_interval(values: np.ndarray, min_val: float = None, 
                           max_val: float = None) -> np.ndarray:
    """
    Rescale values from [0, 8] to [-1, 1] using x/4 - 1 transformation.
    
    Args:
        values: Array of values to rescale (should be in [0, 8])
        min_val: Ignored (kept for API compatibility)
        max_val: Ignored (kept for API compatibility)
        
    Returns:
        Array of values rescaled to [-1, 1] using x/4 - 1
    """
    # Use the specified transformation: x/4 - 1
    # This maps [0, 8] -> [-1, 1]
    return values / 4.0 - 1.0


def rescale_from_unit_interval(values: np.ndarray, min_val: float = None,
                             max_val: float = None) -> np.ndarray:
    """
    Rescale values from [-1, 1] back to [0, 8] using (x + 1) * 4 transformation.
    
    Args:
        values: Array of values in [-1, 1] to rescale
        min_val: Ignored (kept for API compatibility)
        max_val: Ignored (kept for API compatibility)
        
    Returns:
        Array of values rescaled to [0, 8] using (x + 1) * 4
    """
    # Inverse of x/4 - 1 is (x + 1) * 4
    # This maps [-1, 1] -> [0, 8]
    return (values + 1.0) * 4.0


def generate_test_points(epsilon: float, random_seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate comprehensive test dataset for desired and non-desired values.
    
    Args:
        epsilon: The tolerance range for approximate integers
        random_seed: Optional seed for reproducibility
        
    Returns:
        Tuple of (test_points, labels) where:
        - test_points: All generated approximate integer values
        - labels: Binary labels (1.0 for desired, 0.0 for non-desired)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate desired values (around DESIRED_VALUE)
    desired_points = generate_approximate_integers(
        config.DESIRED_VALUE, epsilon, config.POINTS_PER_VALUE, random_seed
    )
    desired_labels = np.ones(len(desired_points))
    
    # Generate non-desired values (around all other integers in domain)
    non_desired_points = []
    non_desired_labels = []
    
    # Get all integer values in domain except the desired one
    integers_in_domain = range(int(config.MIN_VAL), int(config.MAX_VAL) + 1)
    non_desired_integers = [i for i in integers_in_domain if i != int(config.DESIRED_VALUE)]
    
    for integer_val in non_desired_integers:
        points = generate_approximate_integers(
            float(integer_val), epsilon, config.POINTS_PER_VALUE, random_seed
        )
        non_desired_points.extend(points)
        non_desired_labels.extend([0.0] * len(points))
    
    # Combine all points and labels
    all_points = np.concatenate([desired_points, np.array(non_desired_points)])
    all_labels = np.concatenate([desired_labels, np.array(non_desired_labels)])
    
    # Apply domain rescaling if configured
    if config.USE_RESCALED:
        # Use simple x/4 - 1 rescaling (no epsilon padding needed)
        all_points = rescale_to_unit_interval(all_points)
    
    return all_points, all_labels


def generate_epsilon_range() -> np.ndarray:
    """
    Generate array of epsilon values for testing.
    
    Returns:
        Array of epsilon values from MIN_EPSILON to MAX_EPSILON
    """
    return np.linspace(config.MIN_EPSILON, config.MAX_EPSILON, config.NUM_EPSILON_VALUES)


def generate_evaluation_grid(epsilon: float, num_points: int = 1000) -> np.ndarray:
    """
    Generate a fine grid of points for function evaluation and plotting.
    
    Args:
        epsilon: Current epsilon value (for domain padding)
        num_points: Number of evaluation points
        
    Returns:
        Array of evaluation points across the domain
    """
    if config.USE_RESCALED:
        # For rescaled domain, use [-1, 1]
        return np.linspace(-1.0, 1.0, num_points)
    else:
        # For original domain, use [MIN_VAL - epsilon, MAX_VAL + epsilon]
        domain_min = config.MIN_VAL - epsilon
        domain_max = config.MAX_VAL + epsilon
        return np.linspace(domain_min, domain_max, num_points)


def get_desired_value_in_domain(epsilon: float = 0.0) -> float:
    """
    Get the desired value location in the current domain.
    
    Args:
        epsilon: Current epsilon (for domain calculation)
        
    Returns:
        Desired value location in current domain (original or rescaled)
    """
    if config.USE_RESCALED:
        # Rescale desired value to [-1, 1] domain using x/4 - 1
        return rescale_to_unit_interval(np.array([config.DESIRED_VALUE]))[0]
    else:
        return config.DESIRED_VALUE


def validate_test_data(points: np.ndarray, labels: np.ndarray) -> bool:
    """
    Validate generated test data for consistency.
    
    Args:
        points: Generated test points
        labels: Corresponding labels
        
    Returns:
        True if data is valid, False otherwise
    """
    if len(points) != len(labels):
        print(f"Error: Points ({len(points)}) and labels ({len(labels)}) length mismatch")
        return False
    
    if len(points) == 0:
        print("Error: No test points generated")
        return False
    
    # Check label values are binary
    unique_labels = np.unique(labels)
    if not np.array_equal(np.sort(unique_labels), np.array([0.0, 1.0])):
        print(f"Error: Invalid label values: {unique_labels}, expected [0.0, 1.0]")
        return False
    
    # Check domain bounds
    if config.USE_RESCALED:
        if np.min(points) < -1.1 or np.max(points) > 1.1:  # Allow small tolerance
            print(f"Error: Rescaled points outside [-1,1]: [{np.min(points):.3f}, {np.max(points):.3f}]")
            return False
    else:
        if np.min(points) < 0 or np.max(points) > config.MAX_VAL:
            print(f"Error: Points outside [0,{config.MAX_VAL}]: [{np.min(points):.3f}, {np.max(points):.3f}]")
            return False
    
    # Count desired vs non-desired points
    desired_count = np.sum(labels == 1.0)
    non_desired_count = np.sum(labels == 0.0)
    expected_desired = config.POINTS_PER_VALUE
    expected_non_desired = config.POINTS_PER_VALUE * (int(config.MAX_VAL) - int(config.MIN_VAL))
    
    print(f"Generated data: {desired_count} desired, {non_desired_count} non-desired points")
    print(f"Expected: {expected_desired} desired, {expected_non_desired} non-desired points")
    
    return True


def print_data_summary(points: np.ndarray, labels: np.ndarray, epsilon: float):
    """
    Print summary statistics of generated test data.
    
    Args:
        points: Generated test points
        labels: Corresponding labels
        epsilon: Epsilon value used
    """
    print(f"\n=== Data Generation Summary (ε={epsilon:.6f}) ===")
    print(f"Total points: {len(points)}")
    print(f"Domain: {'[-1,1]' if config.USE_RESCALED else f'[{config.MIN_VAL},{config.MAX_VAL}]'}")
    print(f"Point range: [{np.min(points):.6f}, {np.max(points):.6f}]")
    
    desired_count = np.sum(labels == 1.0)
    non_desired_count = np.sum(labels == 0.0)
    print(f"Desired points (label=1): {desired_count}")
    print(f"Non-desired points (label=0): {non_desired_count}")
    
    if config.USE_RESCALED:
        desired_location = get_desired_value_in_domain(epsilon)
        print(f"Desired value location: {config.DESIRED_VALUE} → {desired_location:.6f} (rescaled)")
    else:
        print(f"Desired value location: {config.DESIRED_VALUE}")


if __name__ == "__main__":
    print("=== DATA GENERATION TEST ===")
    
    # Test configuration
    demo_points = 10  # Show 10 points per value (for random mode)
    test_epsilons = [0.001, 0.01, 0.05]  # 3 different epsilon values
    desired_value = 2.0  # Using desired value of 2
    
    # Test both EXACTLY_EPSILON modes
    for exactly_epsilon in [False, True]:
        config.EXACTLY_EPSILON = exactly_epsilon
        mode_name = "EXACTLY EPSILON" if exactly_epsilon else "RANDOM EPSILON"
        print(f"\n{'='*50}")
        print(f"TESTING {mode_name} MODE")
        print(f"{'='*50}")
        
        for i, epsilon in enumerate(test_epsilons):
            print(f"\n--- EPSILON {epsilon} ---")
            
            # Generate desired value points (around 2.0)
            desired_points = generate_approximate_integers(desired_value, epsilon, demo_points, 42+i)
            print(f"Desired ({desired_value}) ± {epsilon}: {desired_points}")
            print(f"  Count: {len(desired_points)} points")
            if len(desired_points) > 0:
                distances = np.abs(desired_points - desired_value)
                print(f"  Distances from target: {distances}")
                if exactly_epsilon:
                    expected_distances = [epsilon] * len(desired_points)
                    if np.allclose(distances, expected_distances):
                        print(f"  ✓ All distances exactly {epsilon}")
                    else:
                        print(f"  ✗ Expected all distances to be {epsilon}")
            
            # Generate points around key values: desired (2.0), boundaries (0, 8), and near-desired (1)
            test_values = [0, 1, 2.0, 8]  # Include desired value and boundary cases
            for test_val in test_values:
                if test_val == 2.0:
                    continue  # Already tested above
                elif test_val in [0, 8]:
                    label = "Boundary"
                else:
                    label = "Non-desired"
                
                test_points = generate_approximate_integers(test_val, epsilon, demo_points, 42+i+int(test_val*10))
                print(f"{label} ({test_val}) ± {epsilon}: {test_points}")
                print(f"  Count: {len(test_points)} points")
                
                if len(test_points) > 0:
                    distances = np.abs(test_points - test_val)
                    print(f"  Distances from target: {distances}")
                    
                    # For boundary cases in exactly epsilon mode, check constraints
                    if exactly_epsilon:
                        if test_val == 0:
                            if np.all(test_points >= 0) and np.allclose(distances, epsilon):
                                print(f"  ✓ Boundary case (0): all points >= 0 and exactly {epsilon} away")
                        elif test_val == 8:
                            if np.all(test_points <= 8) and np.allclose(distances, epsilon):
                                print(f"  ✓ Boundary case (8): all points <= 8 and exactly {epsilon} away")
                        else:
                            expected_distances = [epsilon] * len(test_points)
                            if np.allclose(distances, expected_distances):
                                print(f"  ✓ All distances exactly {epsilon}")
            
            # Test full dataset generation for this epsilon
            print(f"\n  Full dataset test:")
            test_points, test_labels = generate_test_points(epsilon, 42+i)
            desired_count = np.sum(test_labels == 1.0)
            non_desired_count = np.sum(test_labels == 0.0)
            total_count = len(test_points)
            
            print(f"    Total points: {total_count}")
            print(f"    Desired points: {desired_count}")
            print(f"    Non-desired points: {non_desired_count}")
            
            if exactly_epsilon:
                # In exactly epsilon mode, we expect minimal points
                # Desired: 2 points (or 1 if boundary)
                # Non-desired: 2 points each for non-boundary values, 1 each for boundary values
                expected_desired = 2 if desired_value not in [0, 8] else 1
                # Non-desired integers: 0,1,3,4,5,6,7,8 (8 integers, desired=2 excluded)
                non_desired_integers = [i for i in range(9) if i != int(desired_value)]
                expected_non_desired = sum(1 if i in [0, 8] else 2 for i in non_desired_integers)
                expected_total = expected_desired + expected_non_desired
                
                print(f"    Expected: {expected_total} total ({expected_desired} desired + {expected_non_desired} non-desired)")
                if total_count == expected_total:
                    print(f"    ✓ Point count matches expected for exactly epsilon mode")
                else:
                    print(f"    ✗ Point count mismatch")
        
        # Show rescaling example for the first epsilon
        if test_epsilons:
            epsilon = test_epsilons[0]
            print(f"\n--- RESCALING EXAMPLE (ε={epsilon}) ---")
            
            # Generate some test points
            desired_points = generate_approximate_integers(desired_value, epsilon, demo_points, 42)
            rescaled_desired = rescale_to_unit_interval(desired_points)
            rescaled_desired_center = rescale_to_unit_interval(np.array([desired_value]))[0]
            
            print(f"Domain: [0, 8] → [-1, 1] using x/4 - 1")
            print(f"Desired value: {desired_value} → {rescaled_desired_center:.6f}")
            print(f"Original points: {desired_points}")
            print(f"Rescaled points: {rescaled_desired}")
    
    # Reset to original value
    config.EXACTLY_EPSILON = False
    
    print(f"\n--- RESCALING VERIFICATION (x/4 - 1) ---")
    # Test round-trip rescaling accuracy with new transformation
    
    test_values = np.array([0.0, 2.0, 4.0, 8.0])  # Key values in [0, 8]
    rescaled = rescale_to_unit_interval(test_values)
    recovered = rescale_from_unit_interval(rescaled)
    
    print("Round-trip test (x/4 - 1):")
    for orig, resc, recov in zip(test_values, rescaled, recovered):
        error = abs(orig - recov)
        print(f"  {orig:.6f} → {resc:.6f} → {recov:.6f} (error: {error:.2e})")
    
    max_error = np.max(np.abs(test_values - recovered))
    print(f"Max error: {max_error:.2e} {'✓ PASS' if max_error < 1e-10 else '✗ FAIL'}")
    
    # Verify specific transformation values
    print("\nKey transformation points:")
    print(f"  0 → {0/4 - 1} (should be -1.0)")
    print(f"  4 → {4/4 - 1} (should be 0.0)")
    print(f"  8 → {8/4 - 1} (should be 1.0)")