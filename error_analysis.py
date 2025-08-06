"""
Error Analysis Module

This module provides functions for calculating and analyzing approximation errors
between Chebyshev polynomial approximations and true function values, with 
specialized analysis for desired vs non-desired regions.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Callable
import config
import utils
from chebyshev_approximation import eval_chebyshev_function
from indicator_functions import get_indicator_function
from data_generation import generate_test_points, get_desired_value_in_domain


def calculate_approximation_errors(
    func: Callable[[float, float], float],
    test_points: np.ndarray, 
    epsilon: float,
    degree: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate approximation errors between Chebyshev polynomial and true function.
    
    Args:
        func: The indicator function to approximate
        test_points: Array of points to evaluate
        epsilon: Current epsilon value (passed to function)
        degree: Polynomial degree (uses 119 if None)
        
    Returns:
        Tuple of (true_values, approximation_values)
    """
    if degree is None:
        degree = 119
    
    # Calculate domain bounds
    if config.USE_RESCALED:
        domain_bounds = (-1.0, 1.0)
    else:
        domain_bounds = (config.MIN_VAL, config.MAX_VAL)
    
    # Evaluate true function at test points (vectorized for efficiency)
    true_values = utils.vectorized_function_evaluation(func, test_points, epsilon)
    
    # Create function wrapper for Chebyshev approximation
    def func_wrapper(x):
        return func(x, epsilon)
    
    # Evaluate Chebyshev approximation at test points
    approx_values = eval_chebyshev_function(
        func_wrapper, 
        test_points.tolist(), 
        domain_bounds[0], 
        domain_bounds[1], 
        degree
    )
    
    return true_values, np.array(approx_values)


def calculate_expected_errors(
    function_values: np.ndarray,
    expected_labels: np.ndarray
) -> np.ndarray:
    """
    Calculate errors between function output and expected binary labels (0/1).
    
    Args:
        function_values: Output values from indicator function
        expected_labels: Expected binary labels (1.0 for desired, 0.0 for non-desired)
        
    Returns:
        Array of absolute errors |function_value - expected_label|
    """
    return np.abs(function_values - expected_labels)


def analyze_by_region(
    errors: np.ndarray,
    labels: np.ndarray,
    test_points: np.ndarray = None
) -> Dict[str, Dict[str, float]]:
    """
    Analyze errors separately for desired vs non-desired regions.
    
    Args:
        errors: Array of error values
        labels: Binary labels (1.0 for desired, 0.0 for non-desired)
        test_points: Optional array of test points for location analysis
        
    Returns:
        Dictionary with 'desired' and 'non_desired' region statistics
    """
    # Separate errors by region
    desired_mask = labels == 1.0
    non_desired_mask = labels == 0.0
    
    desired_errors = errors[desired_mask]
    non_desired_errors = errors[non_desired_mask]
    
    # Calculate statistics using centralized utility function
    desired_stats = utils.calculate_statistics(desired_errors.tolist())
    non_desired_stats = utils.calculate_statistics(non_desired_errors.tolist())
    
    results = {
        'desired': {
            'count': desired_stats['count'],
            'mean_error': desired_stats['mean'],
            'max_error': desired_stats['max'],
            'min_error': desired_stats['min'],
            'std_error': desired_stats['std'],
            'median_error': desired_stats['median']
        },
        'non_desired': {
            'count': non_desired_stats['count'],
            'mean_error': non_desired_stats['mean'],
            'max_error': non_desired_stats['max'],
            'min_error': non_desired_stats['min'],
            'std_error': non_desired_stats['std'],
            'median_error': non_desired_stats['median']
        }
    }
    
    # Add location analysis if test points provided
    if test_points is not None:
        desired_value_location = get_desired_value_in_domain()
        
        # Find errors closest to desired value
        if len(desired_errors) > 0:
            desired_points = test_points[desired_mask]
            distances_to_desired = np.abs(desired_points - desired_value_location)
            closest_idx = np.argmin(distances_to_desired)
            
            results['desired'].update({
                'error_at_center': desired_errors[closest_idx],
                'min_distance_to_center': np.min(distances_to_desired),
                'mean_distance_to_center': np.mean(distances_to_desired)
            })
    
    return results


def aggregate_statistics(
    epsilon_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate error statistics across multiple epsilon values.
    
    Args:
        epsilon_results: List of results from different epsilon values
        
    Returns:
        Dictionary with aggregated statistics and trends
    """
    if not epsilon_results:
        return {}
    
    # Extract epsilon values and error metrics
    epsilons = []
    desired_mean_errors = []
    non_desired_mean_errors = []
    desired_max_errors = []
    non_desired_max_errors = []
    
    for result in epsilon_results:
        if 'epsilon' in result and 'region_analysis' in result:
            epsilons.append(result['epsilon'])
            
            desired_stats = result['region_analysis']['desired']
            non_desired_stats = result['region_analysis']['non_desired']
            
            desired_mean_errors.append(desired_stats['mean_error'])
            non_desired_mean_errors.append(non_desired_stats['mean_error'])
            desired_max_errors.append(desired_stats['max_error'])
            non_desired_max_errors.append(non_desired_stats['max_error'])
    
    if not epsilons:
        return {}
    
    # Convert to arrays for analysis
    epsilons = np.array(epsilons)
    desired_mean_errors = np.array(desired_mean_errors)
    non_desired_mean_errors = np.array(non_desired_mean_errors)
    
    # Calculate trends (simple linear correlation)
    def calculate_trend(x, y):
        if len(x) < 2:
            return 0.0
        correlation = np.corrcoef(x, y)[0, 1] if not np.isnan(np.corrcoef(x, y)[0, 1]) else 0.0
        return correlation
    
    aggregated = {
        'epsilon_range': {
            'min': float(np.min(epsilons)),
            'max': float(np.max(epsilons)),
            'count': len(epsilons)
        },
        'desired_region': {
            'mean_error_range': {
                'min': float(np.min(desired_mean_errors)),
                'max': float(np.max(desired_mean_errors)),
                'mean': float(np.mean(desired_mean_errors)),
                'std': float(np.std(desired_mean_errors))
            },
            'error_vs_epsilon_trend': calculate_trend(epsilons, desired_mean_errors)
        },
        'non_desired_region': {
            'mean_error_range': {
                'min': float(np.min(non_desired_mean_errors)),
                'max': float(np.max(non_desired_mean_errors)),
                'mean': float(np.mean(non_desired_mean_errors)),
                'std': float(np.std(non_desired_mean_errors))
            },
            'error_vs_epsilon_trend': calculate_trend(epsilons, non_desired_mean_errors)
        },
        'comparison': {
            'desired_vs_non_desired_ratio': float(np.mean(desired_mean_errors) / np.mean(non_desired_mean_errors)) 
                                          if np.mean(non_desired_mean_errors) > 0 else float('inf'),
            'better_approximation_region': 'desired' if np.mean(desired_mean_errors) < np.mean(non_desired_mean_errors) else 'non_desired'
        }
    }
    
    return aggregated


def run_complete_error_analysis(
    function_name: str,
    epsilon: float,
    degree: int = None,
    random_seed: int = None
) -> Dict[str, Any]:
    """
    Run a complete error analysis for a single epsilon value.
    
    Args:
        function_name: Name of the indicator function to analyze
        epsilon: Epsilon value to test
        degree: Polynomial degree (uses 119 if None)
        random_seed: Random seed for reproducible results
        
    Returns:
        Dictionary with comprehensive error analysis results
    """
    if degree is None:
        degree = 119
    
    # Generate test data
    test_points, expected_labels = generate_test_points(epsilon, random_seed)
    
    # Get indicator function
    indicator_func = get_indicator_function(function_name)
    
    # Calculate approximation errors (get Chebyshev approximation values)
    true_values, approx_values = calculate_approximation_errors(
        indicator_func, test_points, epsilon, degree
    )
    # Compare Chebyshev approximation to expected binary labels (0/1)
    chebyshev_errors = np.abs(approx_values - expected_labels)
    
    # Compare original function to expected binary labels (0/1)
    function_errors = np.abs(true_values - expected_labels)
    
    # Analyze by region
    chebyshev_region_analysis = analyze_by_region(chebyshev_errors, expected_labels, test_points)
    function_region_analysis = analyze_by_region(function_errors, expected_labels, test_points)
    
    # Compile results
    results = {
        'function_name': function_name,
        'epsilon': epsilon,
        'degree': degree,
        'test_data': {
            'num_points': len(test_points),
            'num_desired': int(np.sum(expected_labels == 1.0)),
            'num_non_desired': int(np.sum(expected_labels == 0.0)),
            'domain': 'rescaled [-1,1]' if config.USE_RESCALED else f'original [{config.MIN_VAL},{config.MAX_VAL}]'
        },
        'approximation_quality': {
            'mean_chebyshev_error': float(np.mean(chebyshev_errors)),
            'max_chebyshev_error': float(np.max(chebyshev_errors)),
            'chebyshev_region_analysis': chebyshev_region_analysis
        },
        'function_performance': {
            'mean_expected_error': float(np.mean(function_errors)),
            'max_expected_error': float(np.max(function_errors)),
            'expected_region_analysis': function_region_analysis
        },
        'region_analysis': function_region_analysis  # For aggregation compatibility
    }
    
    return results


def print_error_analysis_summary(results: Dict[str, Any]):
    """
    Print a formatted summary of error analysis results with proper precision.
    
    Args:
        results: Results dictionary from run_complete_error_analysis()
    """
    print(f"\n=== ERROR ANALYSIS: {results['function_name'].upper()} ===")
    print(f"Epsilon: {results['epsilon']:.6f}")
    print(f"Degree: {results['degree']}")
    print(f"Domain: {results['test_data']['domain']}")
    
    test_data = results['test_data']
    print(f"\nTest Data:")
    print(f"  Total points: {test_data['num_points']}")
    print(f"  Desired region: {test_data['num_desired']} points")
    print(f"  Non-desired region: {test_data['num_non_desired']} points")
    
    # Chebyshev approximation quality - use scientific notation for small values
    approx_quality = results['approximation_quality']
    mean_cheb = approx_quality['mean_chebyshev_error']
    max_cheb = approx_quality['max_chebyshev_error']
    
    print(f"\nChebyshev Approximation Quality:")
    print(f"  Mean error: {mean_cheb:.2e}")
    print(f"  Max error: {max_cheb:.2e}")
    
    cheb_desired = approx_quality['chebyshev_region_analysis']['desired']
    cheb_non_desired = approx_quality['chebyshev_region_analysis']['non_desired']
    print(f"  Desired region mean error: {cheb_desired['mean_error']:.2e}")
    print(f"  Non-desired region mean error: {cheb_non_desired['mean_error']:.2e}")
    
    # Function performance against binary classification
    func_perf = results['function_performance']
    mean_exp = func_perf['mean_expected_error']
    max_exp = func_perf['max_expected_error']
    
    print(f"\nFunction Performance (vs binary labels):")
    print(f"  Mean error: {mean_exp:.2e}")
    print(f"  Max error: {max_exp:.2e}")
    
    exp_desired = func_perf['expected_region_analysis']['desired']
    exp_non_desired = func_perf['expected_region_analysis']['non_desired']
    print(f"  Desired region mean error: {exp_desired['mean_error']:.2e}")
    print(f"  Non-desired region mean error: {exp_non_desired['mean_error']:.2e}")
    
    # Approximation impact
    approx_impact = results['approximation_impact']
    mean_deg = approx_impact['mean_approximation_degradation']
    max_deg = approx_impact['max_approximation_degradation']
    
    print(f"\nApproximation Impact:")
    print(f"  Mean degradation: {mean_deg:.2e}")
    print(f"  Max degradation: {max_deg:.2e}")


if __name__ == "__main__":
    print("=== ERROR ANALYSIS MODULE TEST ===")
    
    # Test with current configuration
    test_function = config.FUNCTION_TYPE
    test_epsilon = 0.04  # Rescaled epsilon = 0.01
    test_degree = 119  # Use degree 119 as standard
    
    print(f"Testing function: {test_function}")
    print(f"Original epsilon: {test_epsilon}")
    print(f"Rescaled epsilon: {test_epsilon / 4.0}")
    print(f"Degree: {test_degree}")
    print(f"Domain: {'rescaled' if config.USE_RESCALED else 'original'}")
    
    # Run complete error analysis
    results = run_complete_error_analysis(
        function_name=test_function,
        epsilon=test_epsilon,
        degree=test_degree,
        random_seed=42
    )
    
    # Print detailed results
    print_error_analysis_summary(results)
    
    # Test with multiple epsilon values - show individual results
    print(f"\n=== MULTI-EPSILON INDIVIDUAL RESULTS ===")
    
    test_epsilons = [0.04, 0.4]  # 0.04 gives rescaled epsilon = 0.01, 0.4 gives rescaled epsilon = 0.1
    
    for eps in test_epsilons:
        rescaled_eps = eps / 4.0  # Since we use x/4 - 1 transformation
        print(f"\n" + "="*50)
        print(f"EPSILON {eps} ANALYSIS")
        print(f"Original epsilon: {eps}")
        print(f"Rescaled epsilon: {rescaled_eps}")
        print("="*50)
        
        result = run_complete_error_analysis(
            function_name=test_function,
            epsilon=eps,
            degree=119,
            random_seed=42
        )
        
        # Print detailed results for this epsilon
        print_error_analysis_summary(result)
        
        # Additional sanity checks
        func_perf = result['function_performance']
        exp_region = func_perf['expected_region_analysis']
        
        desired_err = exp_region['desired']['mean_error']
        non_desired_err = exp_region['non_desired']['mean_error']
        
        print(f"\nSANITY CHECKS for epsilon {eps}:")
        print(f"  Desired region mean error: {desired_err:.2e}")
        print(f"  Non-desired region mean error: {non_desired_err:.2e}")
        
        # Check if results make intuitive sense
        if non_desired_err == 0.0:
            print("  ❌ WARNING: Non-desired error is exactly 0.0 - this seems wrong")
        elif non_desired_err < 1e-15:
            print("  ⚠️  WARNING: Non-desired error is extremely small - may be rounding issue")
        else:
            print("  ✓ Non-desired error seems reasonable")
        
        if desired_err > non_desired_err * 1000:
            print("  ✓ Desired error >> Non-desired error (expected for impulse function)")
        elif desired_err > non_desired_err:
            print("  ✓ Desired error > Non-desired error (reasonable)")
        else:
            print("  ⚠️  WARNING: Desired error <= Non-desired error (unexpected)")
    
    print(f"\n=== ERROR ANALYSIS MODULE TEST COMPLETE ===")