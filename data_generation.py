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
    Generate approximate integer values around a target value.
    
    Args:
        target_value: The target integer value to approximate
        epsilon: The tolerance range (±epsilon around target)
        num_points: Number of points to generate
        random_seed: Optional seed for reproducibility
        
    Returns:
        Array of approximate values in [target_value - epsilon, target_value + epsilon]
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate uniform random values in [-epsilon, +epsilon]
    random_offsets = np.random.uniform(-epsilon, epsilon, num_points)
    return target_value + random_offsets


def rescale_to_unit_interval(values: np.ndarray, min_val: float = None, 
                           max_val: float = None) -> np.ndarray:
    """
    Rescale values from [min_val, max_val] to [-1, 1] using linear transformation.
    
    Args:
        values: Array of values to rescale
        min_val: Minimum value of original domain (uses config.MIN_VAL if None)
        max_val: Maximum value of original domain (uses config.MAX_VAL if None)
        
    Returns:
        Array of values rescaled to [-1, 1]
    """
    if min_val is None:
        min_val = config.MIN_VAL
    if max_val is None:
        max_val = config.MAX_VAL
    
    # Linear transformation: [min_val, max_val] -> [-1, 1]
    # Formula: 2 * (x - min_val) / (max_val - min_val) - 1
    domain_width = max_val - min_val
    return 2.0 * (values - min_val) / domain_width - 1.0


def rescale_from_unit_interval(values: np.ndarray, min_val: float = None,
                             max_val: float = None) -> np.ndarray:
    """
    Rescale values from [-1, 1] back to [min_val, max_val].
    
    Args:
        values: Array of values in [-1, 1] to rescale
        min_val: Minimum value of target domain (uses config.MIN_VAL if None)
        max_val: Maximum value of target domain (uses config.MAX_VAL if None)
        
    Returns:
        Array of values rescaled to [min_val, max_val]
    """
    if min_val is None:
        min_val = config.MIN_VAL
    if max_val is None:
        max_val = config.MAX_VAL
    
    # Inverse linear transformation: [-1, 1] -> [min_val, max_val]
    # Formula: (x + 1) * (max_val - min_val) / 2 + min_val
    domain_width = max_val - min_val
    return (values + 1.0) * domain_width / 2.0 + min_val


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
        # Calculate domain bounds with epsilon padding
        domain_min = config.MIN_VAL - epsilon
        domain_max = config.MAX_VAL + epsilon
        all_points = rescale_to_unit_interval(all_points, domain_min, domain_max)
    
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
        # Calculate domain bounds with epsilon padding for rescaling
        domain_min = config.MIN_VAL - epsilon
        domain_max = config.MAX_VAL + epsilon
        # Rescale desired value to [-1, 1] domain
        return rescale_to_unit_interval(
            np.array([config.DESIRED_VALUE]), domain_min, domain_max
        )[0]
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
    demo_points = 10  # Show 10 points per value
    test_epsilons = [0.001, 0.01, 0.05]  # 3 different epsilon values
    desired_value = 2.0  # Using desired value of 2
    
    for i, epsilon in enumerate(test_epsilons):
        print(f"\n--- EPSILON {epsilon} ---")
        
        # Generate desired value points (around 2.0)
        desired_points = generate_approximate_integers(desired_value, epsilon, demo_points, 42+i)
        print(f"Desired ({desired_value}) ± {epsilon}: {desired_points}")
        
        # Generate non-desired value points (around 0, 1, 3)
        for non_desired in [0, 1, 3]:
            non_desired_points = generate_approximate_integers(float(non_desired), epsilon, demo_points, 42+i+non_desired)
            print(f"Non-desired ({non_desired}) ± {epsilon}: {non_desired_points}")
        
        # Show rescaling for this epsilon
        print(f"\nRescaling with epsilon {epsilon}:")
        extended_min = 0.0 - epsilon  # MIN_VAL - epsilon
        extended_max = 8.0 + epsilon  # MAX_VAL + epsilon
        
        # Rescale desired points
        rescaled_desired = rescale_to_unit_interval(desired_points, extended_min, extended_max)
        rescaled_epsilon = 2 * epsilon / (extended_max - extended_min)
        rescaled_desired_center = rescale_to_unit_interval(np.array([desired_value]), extended_min, extended_max)[0]
        
        print(f"  Extended domain: [{extended_min:.6f}, {extended_max:.6f}] → [-1, 1]")
        print(f"  Original epsilon: {epsilon} → Rescaled epsilon: {rescaled_epsilon:.6f}")
        print(f"  Desired value: {desired_value} → {rescaled_desired_center:.6f}")
        print(f"  Rescaled desired points: {rescaled_desired}")
    
    print(f"\n--- RESCALING VERIFICATION ---")
    # Test round-trip rescaling accuracy
    test_epsilon = 0.01
    extended_min = 0.0 - test_epsilon
    extended_max = 8.0 + test_epsilon
    
    test_values = np.array([extended_min, 0.0, 2.0, 8.0, extended_max])
    rescaled = rescale_to_unit_interval(test_values, extended_min, extended_max)
    recovered = rescale_from_unit_interval(rescaled, extended_min, extended_max)
    
    print("Round-trip test:")
    for orig, resc, recov in zip(test_values, rescaled, recovered):
        error = abs(orig - recov)
        print(f"  {orig:.6f} → {resc:.6f} → {recov:.6f} (error: {error:.2e})")
    
    max_error = np.max(np.abs(test_values - recovered))
    print(f"Max error: {max_error:.2e} {'✓ PASS' if max_error < 1e-10 else '✗ FAIL'}")