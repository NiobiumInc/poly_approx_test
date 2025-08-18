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


def get_output_paths():
    """Get organized output paths based on config."""
    from pathlib import Path
    
    if hasattr(config, 'GRAPHS_BASE_PATH'):
        base_path = Path(config.GRAPHS_BASE_PATH)
    else:
        base_path = Path("graphs") / "aug6"
    
    function_path = base_path / config.FUNCTION_TYPE
    log_path = base_path / "experiment_log.csv"
    
    return {
        'base_path': base_path,
        'function_path': function_path,
        'log_path': log_path
    }


def print_banner():
    """Print application banner."""
    print("=" * 60)
    print("  POLYNOMIAL APPROXIMATION ANALYSIS")
    print(f"  Function: {config.FUNCTION_TYPE}")
    print(f"  Domain: {'rescaled [-1,1]' if config.USE_RESCALED else f'[{config.MIN_VAL},{config.MAX_VAL}]'}")
    print(f"  Epsilon range: {config.MIN_EPSILON} to {config.MAX_EPSILON}")
    print(f"  Exactly epsilon: {config.EXACTLY_EPSILON}")
    print(f"  Degree: {config.CHEB_DEGREE}")
    print("=" * 60)


def generate_epsilon_values():
    """Generate epsilon values based on config settings."""
    import numpy as np
    import config
    return np.logspace(
        np.log10(config.MIN_EPSILON),
        np.log10(config.MAX_EPSILON), 
        config.NUM_EPSILON_VALUES
    )


def run_config_based_analysis() -> Dict[str, Any]:
    """
    Run analysis based on config.py settings.
    
    Returns:
        Dictionary containing analysis results
    """
    import time
    import utils
    
    print_banner()
    
    # Generate unique run ID
    run_id = utils.create_unique_id(4)
    print(f"\nRun ID: {run_id}")
    
    # Setup output directories
    output_paths = get_output_paths()
    utils.ensure_directory_exists(output_paths['base_path'])
    utils.ensure_directory_exists(output_paths['function_path'])
    
    # Generate epsilon values from config
    epsilon_values = generate_epsilon_values()
    
    print(f"Running analysis with {len(epsilon_values)} epsilon values")
    print(f"Epsilon values: {epsilon_values[0]:.2e} to {epsilon_values[-1]:.2e}")
    
    results = {
        'run_id': run_id,
        'function_name': config.FUNCTION_TYPE,
        'epsilon_values': epsilon_values.tolist(),
        'individual_results': [],
        'summary': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_experiments': len(epsilon_values)
        }
    }
    
    # Run analysis for each epsilon value
    for i, epsilon in enumerate(epsilon_values):
        print(f"\nProgress: {i+1}/{len(epsilon_values)} - ε = {epsilon:.2e}")
        
        try:
            result = run_complete_error_analysis(
                function_name=config.FUNCTION_TYPE,
                epsilon=epsilon,
                degree=config.CHEB_DEGREE,
                random_seed=42
            )
            
            results['individual_results'].append(result)
            
            # Print key metrics
            cheb_error = result['approximation_quality']['mean_chebyshev_error']
            func_error = result['function_performance']['mean_expected_error']
            
            print(f"  Chebyshev error: {cheb_error:.2e}")
            print(f"  Function error: {func_error:.2e}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return results


def extract_result_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from a single result for reuse."""
    epsilon = result['epsilon']
    cheb_error = result['approximation_quality']['mean_chebyshev_error']
    func_error = result['function_performance']['mean_expected_error']
    ratio = func_error / cheb_error if cheb_error > 0 else float('inf')
    
    return {
        'epsilon': epsilon,
        'cheb_error': cheb_error, 
        'func_error': func_error,
        'ratio': ratio
    }


def save_results(results: Dict[str, Any]):
    """Save results to organized output structure."""
    output_paths = get_output_paths()
    
    # Create summary report
    summary_lines = [
        "POLYNOMIAL APPROXIMATION ANALYSIS RESULTS",
        "=" * 50,
        f"Function: {results['function_name']}",
        f"Analysis completed: {results['summary']['timestamp']}",
        f"Total experiments: {results['summary']['total_experiments']}",
        f"Configuration:",
        f"  Domain: {'rescaled [-1,1]' if config.USE_RESCALED else f'[{config.MIN_VAL},{config.MAX_VAL}]'}",
        f"  Chebyshev degree: {config.CHEB_DEGREE}",
        f"  Points per value: {config.POINTS_PER_VALUE}",
        "",
        "EPSILON ANALYSIS SUMMARY:",
        "-" * 25
    ]
    
    # Add epsilon-specific results and build CSV data simultaneously
    csv_lines = ['epsilon,chebyshev_error,function_error,error_ratio,cheb_desired_error,cheb_non_desired_error,func_desired_error,func_non_desired_error']
    
    for result in results['individual_results']:
        metrics = extract_result_metrics(result)
        
        # Extract region-specific errors
        cheb_quality = result['approximation_quality']['chebyshev_region_analysis']
        func_performance = result['function_performance']['expected_region_analysis']
        
        cheb_desired_error = cheb_quality['desired']['mean_error']
        cheb_non_desired_error = cheb_quality['non_desired']['mean_error']
        func_desired_error = func_performance['desired']['mean_error']
        func_non_desired_error = func_performance['non_desired']['mean_error']
        
        # Add to summary
        summary_lines.extend([
            f"ε = {metrics['epsilon']:.2e}:",
            f"  Chebyshev error: {metrics['cheb_error']:.2e}",
            f"  Function error: {metrics['func_error']:.2e}",
            f"  Error ratio: {metrics['ratio']:.1f}",
            ""
        ])
        
        # Add to CSV with region-specific data
        csv_lines.append(f"{metrics['epsilon']},{metrics['cheb_error']},{metrics['func_error']},{metrics['ratio']},{cheb_desired_error},{cheb_non_desired_error},{func_desired_error},{func_non_desired_error}")
    
    # Add output locations
    summary_lines.extend([
        "OUTPUT LOCATIONS:",
        "-" * 17,
        f"Base directory: {output_paths['base_path']}",
        f"Function plots: {output_paths['function_path']}",
        f"Summary: {output_paths['function_path'] / ('analysis_summary_' + results['run_id'] + '.txt')}"
    ])
    
    # Save summary
    summary_path = output_paths['function_path'] / f'analysis_summary_{results["run_id"]}.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # Save detailed results as CSV
    csv_path = output_paths['function_path'] / f"{config.FUNCTION_TYPE}_detailed_results_{results['run_id']}.csv"
    
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines))
    
    print(f"\nResults saved:")
    print(f"  Summary: {summary_path}")
    print(f"  Detailed CSV: {csv_path}")


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