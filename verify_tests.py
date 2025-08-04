#!/usr/bin/env python3
"""
Verification Tests for Polynomial Approximation System

This script runs various tests to verify that the system is working correctly:
1. Test rescaling function
2. Test function evaluation 
3. Test that epsilon variation actually affects results
4. Verify that different functions produce different results
"""

import numpy as np
import matplotlib.pyplot as plt
from indicator_functions import FunctionConfig, get_function_by_name
from generate_approx_integers import rescale_to_unit_interval, generate_mixed_approx_integers
from chebyshev_approximation import ChebyshevApproximator
import config

def test_rescaling():
    """Test that rescaling function works correctly."""
    print("=== Testing Rescaling Function ===")
    
    # Test key values
    test_values = np.array([0, 2, 4, 8])  # Important integer points
    rescaled = rescale_to_unit_interval(test_values, max_value=8)
    
    print("Original -> Rescaled:")
    for orig, resc in zip(test_values, rescaled):
        print(f"  {orig} -> {resc:.3f}")
    
    # Verify expected mappings
    expected = np.array([-1.0, -0.5, 0.0, 1.0])  # 0->-1, 2->-0.5, 4->0, 8->1
    
    if np.allclose(rescaled, expected, atol=1e-10):
        print("✓ Rescaling function works correctly!")
        return True
    else:
        print("✗ Rescaling function ERROR!")
        print(f"Expected: {expected}")
        print(f"Got: {rescaled}")
        return False

def test_function_evaluation():
    """Test that different functions produce different outputs."""
    print("\n=== Testing Function Evaluation ===")
    
    # Test points around the desired value (2)
    test_points = np.array([1.5, 2.0, 2.5, 0.0, 8.0])
    
    functions_to_test = ['impulse', 'plateau_sine', 'plateau_reg']
    results = {}
    
    for func_name in functions_to_test:
        config_obj = FunctionConfig(
            function_name=func_name,
            desired_value=2.0,
            use_rescaled=False  # Test original domain first
        )
        
        func = get_function_by_name(func_name, config_obj)
        outputs = []
        
        for point in test_points:
            output = func(point)
            outputs.append(output)
        
        results[func_name] = outputs
        print(f"{func_name}:")
        for point, output in zip(test_points, outputs):
            print(f"  f({point}) = {output:.4f}")
    
    # Check that functions produce different results
    impulse_outputs = np.array(results['impulse'])
    plateau_sine_outputs = np.array(results['plateau_sine'])
    plateau_reg_outputs = np.array(results['plateau_reg'])
    
    different_funcs = (
        not np.allclose(impulse_outputs, plateau_sine_outputs, atol=1e-3) and
        not np.allclose(impulse_outputs, plateau_reg_outputs, atol=1e-3) and
        not np.allclose(plateau_sine_outputs, plateau_reg_outputs, atol=1e-3)
    )
    
    if different_funcs:
        print("✓ Functions produce different outputs!")
        return True
    else:
        print("✗ Functions produce similar outputs - something is wrong!")
        return False

def test_epsilon_variation_effect():
    """Test that changing epsilon actually affects the generated test points."""
    print("\n=== Testing Epsilon Variation Effect ===")
    
    epsilon_values = [0.001, 0.01, 0.1]
    point_spreads = []
    
    for epsilon in epsilon_values:
        # Generate test points with this epsilon
        test_points = generate_mixed_approx_integers(
            n_excluding=400, n_desired=100, 
            max_value=8, desired_value=2.0, epsilon=epsilon
        )
        
        # Measure how much the points deviate from integers
        integer_deviations = []
        for point in test_points:
            closest_int = round(point)
            deviation = abs(point - closest_int)
            integer_deviations.append(deviation)
        
        max_deviation = max(integer_deviations)
        avg_deviation = np.mean(integer_deviations)
        
        point_spreads.append((max_deviation, avg_deviation))
        print(f"Epsilon {epsilon}: Max deviation = {max_deviation:.6f}, Avg deviation = {avg_deviation:.6f}")
    
    # Check if point spreads increase with epsilon
    max_devs = [spread[0] for spread in point_spreads]
    avg_devs = [spread[1] for spread in point_spreads]
    
    # Check if deviations are increasing
    max_dev_increasing = all(max_devs[i] <= max_devs[i+1] for i in range(len(max_devs)-1))
    avg_dev_increasing = all(avg_devs[i] <= avg_devs[i+1] for i in range(len(avg_devs)-1))
    
    if max_dev_increasing and avg_dev_increasing:
        print("✓ Epsilon variation affects point generation correctly!")
        return True
    else:
        print("✗ Epsilon variation doesn't properly affect point generation.")
        return False

def test_rescaled_vs_original():
    """Test that rescaled and original domains produce different function values."""
    print("\n=== Testing Rescaled vs Original Domain ===")
    
    # Test the same physical point (2.0) in both domains
    test_point_original = 2.0
    test_point_rescaled = rescale_to_unit_interval(np.array([2.0]), max_value=8)[0]
    
    results = {}
    
    for use_rescaled in [False, True]:
        config_obj = FunctionConfig(
            function_name='impulse',
            desired_value=2.0,
            use_rescaled=use_rescaled
        )
        
        func = get_function_by_name('impulse', config_obj)
        
        if use_rescaled:
            func_output = func(test_point_rescaled)
            domain_str = "rescaled"
            test_point = test_point_rescaled
        else:
            func_output = func(test_point_original)
            domain_str = "original"
            test_point = test_point_original
        
        results[domain_str] = func_output
        print(f"{domain_str.capitalize()} domain: f({test_point:.3f}) = {func_output:.6f}")
    
    # Both should produce similar outputs since they represent the same physical point
    output_diff = abs(results['rescaled'] - results['original'])
    
    if output_diff < 0.1:  # Should be similar since it's the same physical point
        print(f"✓ Rescaled and original domains produce similar outputs for same physical point! Difference: {output_diff:.6f}")
        return True
    else:
        print(f"✗ Rescaled and original domains produce very different outputs. Difference: {output_diff:.6f}")
        return False

def plot_function_comparison():
    """Create plots to visually verify functions work correctly."""
    print("\n=== Creating Function Comparison Plots ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    functions = ['impulse', 'plateau_sine', 'plateau_reg']
    domains = ['original', 'rescaled']
    
    for i, domain in enumerate(domains):
        use_rescaled = (domain == 'rescaled')
        
        if use_rescaled:
            x_vals = np.linspace(-1, 1, 200)
        else:
            x_vals = np.linspace(0, 8, 200)
        
        for j, func_name in enumerate(functions):
            config_obj = FunctionConfig(
                function_name=func_name,
                desired_value=2.0,
                use_rescaled=use_rescaled
            )
            
            func = get_function_by_name(func_name, config_obj)
            y_vals = [func(x) for x in x_vals]
            
            ax = axes[i, j]
            ax.plot(x_vals, y_vals, 'b-', linewidth=2)
            ax.set_title(f'{func_name} ({domain})')
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.grid(True, alpha=0.3)
            
            # Add vertical line at desired value
            if use_rescaled:
                desired_rescaled = rescale_to_unit_interval(np.array([2.0]), max_value=8)[0]
                ax.axvline(desired_rescaled, color='red', linestyle='--', alpha=0.7, label='desired')
            else:
                ax.axvline(2.0, color='red', linestyle='--', alpha=0.7, label='desired')
            
            ax.legend()
    
    plt.tight_layout()
    plot_path = f'{config.GRAPHS_BASE_PATH}/verification_functions.png'
    import os
    os.makedirs(config.GRAPHS_BASE_PATH, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Function comparison plot saved to: {plot_path}")

def main():
    """Run all verification tests."""
    print("=" * 80)
    print("POLYNOMIAL APPROXIMATION VERIFICATION TESTS")
    print("=" * 80)
    
    tests_passed = []
    
    # Run all tests
    tests_passed.append(test_rescaling())
    tests_passed.append(test_function_evaluation())
    tests_passed.append(test_epsilon_variation_effect())
    tests_passed.append(test_rescaled_vs_original())
    
    # Create visual verification
    plot_function_comparison()
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    passed_count = sum(tests_passed)
    total_count = len(tests_passed)
    
    if passed_count == total_count:
        print(f"✓ ALL TESTS PASSED ({passed_count}/{total_count})")
        print("The system appears to be working correctly!")
    else:
        print(f"✗ SOME TESTS FAILED ({passed_count}/{total_count})")
        print("There are issues that need to be addressed.")
    
    print(f"\nVisual verification plots saved to: {config.GRAPHS_BASE_PATH}/")
    print("Review the plots to visually confirm function behavior.")
    
    return passed_count == total_count

if __name__ == "__main__":
    main()