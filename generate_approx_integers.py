import numpy as np

def generate_approx_integers_excluding(n=80, max_value=8, excluded_value=2, epsilon=0.1):
    """
    Generate an array of N values that are approximately integers between 0 and max_value,
    excluding the excluded_value.
    
    Parameters:
    - n: Number of values to generate (default: 80)
    - max_value: Maximum integer value (default: 8)
    - excluded_value: Integer value to exclude (default: 2)
    - epsilon: Maximum deviation from integer (default: 0.1)
    
    Returns:
    - numpy array of approximate integers
    """
    # Create array of valid integer values (excluding the excluded_value)
    valid_integers = np.array([i for i in range(max_value + 1) if i != excluded_value])
    
    # Randomly select integers from valid options
    selected_integers = np.random.choice(valid_integers, size=n)
    
    # Generate random approximation amounts with random signs
    approximations = np.random.uniform(-epsilon, epsilon, size=n)
    
    return selected_integers + approximations

def generate_approx_desired_value(n, desired_value=2, epsilon=0.1):
    """
    Generate an array of N values that are at most epsilon away from the desired value.
    
    Parameters:
    - n: Number of values to generate
    - desired_value: The target value to approximate (default: 2)
    - epsilon: Maximum deviation from desired value (default: 0.1)
    
    Returns:
    - numpy array of values close to desired_value
    """
    # Generate random approximation amounts with random signs
    approximations = np.random.uniform(-epsilon, epsilon, size=n)
    
    return np.full(n, desired_value) + approximations

def generate_mixed_approx_integers(n_excluding=80, n_desired=10, max_value=8, 
                                 desired_value=2, epsilon=0.1):
    """
    Generate a mixed array with values excluding and including the desired value.
    
    Parameters:
    - n_excluding: Number of values excluding desired_value (default: 80)
    - n_desired: Number of values close to desired_value (default: 10)
    - max_value: Maximum integer value (default: 8)
    - desired_value: The value to exclude/include (default: 2)
    - epsilon: Maximum deviation from integer (default: 0.1)
    
    Returns:
    - numpy array combining both types of values
    """
    excluding_values = generate_approx_integers_excluding(
        n_excluding, max_value, desired_value, epsilon
    )
    desired_values = generate_approx_desired_value(n_desired, desired_value, epsilon)
    
    # Combine and shuffle the arrays
    combined = np.concatenate([excluding_values, desired_values])
    np.random.shuffle(combined)
    
    return combined

def rescale_to_unit_interval(values, max_value=8):
    """
    Rescale values from [0, max_value] to [-1, 1] using the formula:
    x_i = i / (max_value/2) - 1.0
    
    For max_value=8: x_i = i / 4.0 - 1.0
    
    Parameters:
    - values: numpy array of values in [0, max_value] range
    - max_value: maximum input value (default: 8)
    
    Returns:
    - numpy array scaled to [-1, 1]
    """
    scale_factor = max_value / 2.0  # 4.0 when max_value=8
    return values / scale_factor - 1.0

if __name__ == "__main__":
    # Test excluding desired value
    print("=== Values excluding desired value ===")
    excluding_result = generate_approx_integers_excluding()
    print(f"Generated {len(excluding_result)} values excluding 2:")
    print(f"Range: {excluding_result.min():.3f} to {excluding_result.max():.3f}")
    print(f"First 10: {excluding_result[:10]}")
    
    # Test values close to desired value
    print("\n=== Values close to desired value ===")
    desired_result = generate_approx_desired_value(10)
    print(f"Generated {len(desired_result)} values close to 2:")
    print(f"Range: {desired_result.min():.3f} to {desired_result.max():.3f}")
    print(f"All values: {desired_result}")
    
    # Test mixed array
    print("\n=== Mixed array ===")
    mixed_result = generate_mixed_approx_integers()
    print(f"Generated {len(mixed_result)} total values:")
    print(f"Range: {mixed_result.min():.3f} to {mixed_result.max():.3f}")
    
    # Count values close to desired value
    close_to_desired = np.abs(mixed_result - 2) < 0.5
    print(f"Values close to desired value (2): {np.sum(close_to_desired)}")
    print(f"First 15 values: {mixed_result[:15]}")
    
    # Test rescaling function
    print("\n=== Rescaling test ===")
    test_values = np.array([0, 2, 4, 8])
    rescaled = rescale_to_unit_interval(test_values)
    print(f"Original values: {test_values}")
    print(f"Rescaled to [-1,1]: {rescaled}")
    
    # Test with mixed result
    rescaled_mixed = rescale_to_unit_interval(mixed_result)
    print(f"Mixed result rescaled range: {rescaled_mixed.min():.3f} to {rescaled_mixed.max():.3f}")