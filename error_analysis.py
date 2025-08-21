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
    
    # Evaluate Chebyshev polynomial approximation at test points
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
    # Determine method name for labeling
    method_name = "chebyshev"
    method_display = "Chebyshev"
    
    # Compare polynomial approximation to expected binary labels (0/1)
    approx_errors = np.abs(approx_values - expected_labels)
    
    # Compare original function to expected binary labels (0/1)
    function_errors = np.abs(true_values - expected_labels)
    
    # Analyze by region
    approx_region_analysis = analyze_by_region(approx_errors, expected_labels, test_points)
    function_region_analysis = analyze_by_region(function_errors, expected_labels, test_points)
    
    # Compile results with dynamic naming
    results = {
        'function_name': function_name,
        'epsilon': epsilon,
        'degree': degree,
        'method_name': method_name,
        'method_display': method_display,
        'test_data': {
            'num_points': len(test_points),
            'num_desired': int(np.sum(expected_labels == 1.0)),
            'num_non_desired': int(np.sum(expected_labels == 0.0)),
            'domain': 'rescaled [-1,1]' if config.USE_RESCALED else f'original [{config.MIN_VAL},{config.MAX_VAL}]'
        },
        'approximation_quality': {
            f'mean_{method_name}_error': float(np.mean(approx_errors)),
            f'max_{method_name}_error': float(np.max(approx_errors)),
            f'{method_name}_region_analysis': approx_region_analysis
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
            
            # Print key metrics - use dynamic method name
            method_key = result.get('method_name', 'chebyshev')
            method_display = result.get('method_display', 'Chebyshev')
            approx_error = result['approximation_quality'][f'mean_{method_key}_error']
            func_error = result['function_performance']['mean_expected_error']
            
            print(f"  {method_display} error: {approx_error:.2e}")
            print(f"  Function error: {func_error:.2e}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return results


def extract_result_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from a single result for reuse."""
    epsilon = result['epsilon']
    method_name = result.get('method_name', 'chebyshev')  # fallback for old results
    
    # Dynamically extract the approximation error based on method
    approx_error = result['approximation_quality'][f'mean_{method_name}_error']
    func_error = result['function_performance']['mean_expected_error']
    ratio = func_error / approx_error if approx_error > 0 else float('inf')
    
    return {
        'epsilon': epsilon,
        'cheb_error': approx_error,  # keeping same field name for compatibility
        'func_error': func_error,
        'ratio': ratio
    }


def get_function_latex(function_name: str) -> str:
    """Generate LaTeX equation for the specified function with current parameters."""
    if function_name == "impulse":
        params = config.get_function_params("impulse")
        sigma = params["sigma"]
        mu_param = params["mu"]
        scaling = params["scaling"]
        
        # Determine actual center based on configuration
        if config.USE_RESCALED:
            if mu_param == 0:
                mu_display = "\\text{rescaled}(2.0)"
            else:
                mu_display = f"\\text{{rescaled}}({mu_param})"
            sigma_display = f"{sigma}/4"  # Adjusted for rescaled domain
        else:
            mu_display = f"{config.DESIRED_VALUE}" if mu_param == 0 else f"{mu_param}"
            sigma_display = f"{sigma}"
            
        return f"f(x) = {scaling} \\cdot \\exp\\left(-\\frac{{(x - {mu_display})^2}}{{2({sigma_display})^2}}\\right)"
        
    elif function_name == "plateau_sine":
        params = config.get_function_params("plateau_sine")
        base_amp = params["base_amp"]
        base_freq = params["base_freq"]
        amp = params["amplitude"]
        freq = params["freq"]
        steepness = params["steepness"]
        width = params["width"]
        
        # Calculate actual rescaled values (don't use text placeholders)
        if config.USE_RESCALED:
            # Calculate the actual rescaled desired value: x/4 - 1 where x=2.0 gives -0.5
            import numpy as np
            import indicator_functions
            mu_value = indicator_functions.rescale_to_unit_interval(np.array([config.DESIRED_VALUE]))[0]
            width_value = width / 4.0
        else:
            mu_value = config.DESIRED_VALUE
            width_value = width
            
        # Multi-line LaTeX format with parameter symbols
        latex = "\\begin{align*}\n"
        latex += "    B(x) &= \\alpha \\sin(\\phi \\pi x), \\\\\n"
        latex += "    \\sigma_+(x) &= \\frac{1}{1 + e^{-\\kappa(x - (\\mu - \\frac{w}{2}))}}, \\\\\n"
        latex += "    \\sigma_-(x) &= \\frac{1}{1 + e^{-\\kappa((\\mu + \\frac{w}{2}) - x)}}, \\\\\n"
        latex += "    m(x) &= \\sigma_+(x) \\cdot \\sigma_-(x), \\\\\n"
        latex += "    r(x) &= \\beta \\sin(\\gamma \\pi x), \\\\\n"
        latex += "    f(x) &= [1 + r(x)] m(x) + B(x) [1 - m(x)].\n"
        latex += "\\end{align*}\n\n"
        latex += "with parameters:\n"
        latex += "\\[\n"
        latex += f"\\alpha = {base_amp}, \\quad \\phi = {base_freq}, \\quad \\kappa = {steepness}, \\quad w = {width_value:.3f}, \\quad \\beta = {amp}, \\quad \\gamma = {freq}, \\quad \\mu = {mu_value:.3f}.\n"
        latex += "\\]"
        
        return latex
        
    elif function_name == "plateau_reg":
        if config.USE_RESCALED:
            mu_display = "\\text{rescaled}(2.0)"
        else:
            mu_display = f"{config.DESIRED_VALUE}"
            
        return f"f(x) = \\begin{{cases}} 1 & \\text{{if }} |x - {mu_display}| \\leq 0.5 \\\\ 0 & \\text{{otherwise}} \\end{{cases}}"
        
    elif function_name in ["plateau_sine_impulse", "plateau_sine_impulse_clean"]:
        plateau_params = config.get_function_params("plateau_sine")
        impulse_params = config.get_function_params("impulse")
        
        # This is complex - provide a general form
        if function_name == "plateau_sine_impulse_clean":
            return "f(x) = \\text{plateau}(x) \\cdot \\text{mask}(x) + \\text{impulse}(x) \\cdot (1 - \\text{mask}(x))"
        else:
            return "f(x) = \\text{plateau\\_sine}(x) \\cdot \\text{mask}(x) + \\text{impulse}(x) \\cdot (1 - \\text{mask}(x))"
    
    return f"f(x) = \\text{{Function not implemented: {function_name}}}"


def get_chebyshev_latex(degree: int) -> str:
    """Generate LaTeX equation for Chebyshev polynomial approximation."""
    if degree <= 3:
        # Show explicit form for low degrees
        return f"P_{{{degree}}}(x) = \\sum_{{k=0}}^{{{degree}}} a_k T_k(x) = a_0 + a_1 T_1(x) + a_2 T_2(x)" + (f" + a_3 T_3(x)" if degree >= 3 else "")
    else:
        return f"P_{{{degree}}}(x) = \\sum_{{k=0}}^{{{degree}}} a_k T_k(x) \\quad \\text{{where }} T_k(x) \\text{{ are Chebyshev polynomials}}"


def get_function_specific_params(function_name: str) -> Dict[str, Any]:
    """Get only the relevant parameters for the specified function type."""
    relevant_params = {}
    
    if function_name == "impulse":
        relevant_params["impulse_params"] = config.IMPULSE.copy()
    elif function_name in ["plateau_sine", "plateau_sine_impulse", "plateau_sine_impulse_clean"]:
        relevant_params["plateau_sine_params"] = config.PLATEAU_SINE.copy()
        if function_name in ["plateau_sine_impulse", "plateau_sine_impulse_clean"]:
            relevant_params["impulse_params"] = config.IMPULSE.copy()
    # plateau_reg has no additional parameters
    
    return relevant_params


def save_results(results: Dict[str, Any]):
    """Save results to organized output structure with comprehensive configuration and LaTeX."""
    output_paths = get_output_paths()
    
    function_name = results['function_name']
    
    # Create comprehensive summary report
    summary_lines = [
        "POLYNOMIAL APPROXIMATION ANALYSIS RESULTS",
        "=" * 60,
        "",
        "FUNCTION INFORMATION:",
        "-" * 21,
        f"Target Function: {function_name}",
        f"LaTeX: {get_function_latex(function_name)}",
        "",
        "APPROXIMATION METHOD:",
        "-" * 21,
        f"Method: Chebyshev Polynomial Approximation",
        f"LaTeX: {get_chebyshev_latex(config.CHEB_DEGREE)}",
        "",
        "COMPLETE CONFIGURATION:",
        "-" * 23,
        f"Analysis completed: {results['summary']['timestamp']}",
        f"Total experiments: {results['summary']['total_experiments']}",
        "",
        "Output Settings:",
        f"  DATE_FOLDER: {config.DATE_FOLDER}",
        f"  GRAPHS_BASE_PATH: {config.GRAPHS_BASE_PATH}",
        f"  ROUND_PRECISION: {config.ROUND_PRECISION}",
        "",
        "Test Execution Settings:",
        f"  FUNCTION_TYPE: {config.FUNCTION_TYPE}",
        f"  POINTS_PER_VALUE: {config.POINTS_PER_VALUE}",
        f"  USE_RESCALED: {config.USE_RESCALED}",
        "",
        "Domain Settings:",
        f"  DESIRED_VALUE: {config.DESIRED_VALUE}",
        f"  MAX_VAL: {config.MAX_VAL}",
        f"  MIN_VAL: {config.MIN_VAL}",
        f"  Effective domain: {'[-1,1] (rescaled)' if config.USE_RESCALED else f'[{config.MIN_VAL},{config.MAX_VAL}]'}",
        "",
        "Epsilon Testing:",
        f"  MIN_EPSILON: {config.MIN_EPSILON}",
        f"  MAX_EPSILON: {config.MAX_EPSILON}",
        f"  NUM_EPSILON_VALUES: {config.NUM_EPSILON_VALUES}",
        f"  EXACTLY_EPSILON: {config.EXACTLY_EPSILON}",
        "",
        "Polynomial Approximation Settings:",
        f"  CHEB_DEGREE: {config.CHEB_DEGREE}",
        ""
    ]
    
    # Add function-specific parameters
    func_params = get_function_specific_params(function_name)
    if func_params:
        summary_lines.append("Function-Specific Parameters:")
        for param_group, params in func_params.items():
            summary_lines.append(f"  {param_group}:")
            for key, value in params.items():
                summary_lines.append(f"    {key}: {value}")
        summary_lines.append("")
    
    # Add analysis results section
    summary_lines.extend([
        "EPSILON ANALYSIS SUMMARY:",
        "-" * 26
    ])
    
    # Add epsilon-specific results and build CSV data simultaneously
    # Determine method name for CSV header
    method_name = results['individual_results'][0].get('method_name', 'chebyshev') if results['individual_results'] else 'chebyshev'
    method_display = results['individual_results'][0].get('method_display', 'Chebyshev') if results['individual_results'] else 'Chebyshev'
    
    csv_lines = [f'epsilon,{method_name}_error,function_error,error_ratio,{method_name}_desired_error,{method_name}_non_desired_error,func_desired_error,func_non_desired_error']
    
    for result in results['individual_results']:
        metrics = extract_result_metrics(result)
        
        # Extract region-specific errors - dynamically find the method name
        method_name = result.get('method_name', 'chebyshev')  # fallback for old results
        approx_quality = result['approximation_quality'][f'{method_name}_region_analysis']
        func_performance = result['function_performance']['expected_region_analysis']
        
        approx_desired_error = approx_quality['desired']['mean_error']
        approx_non_desired_error = approx_quality['non_desired']['mean_error']
        func_desired_error = func_performance['desired']['mean_error']
        func_non_desired_error = func_performance['non_desired']['mean_error']
        
        # Add to summary with dynamic method name
        summary_lines.extend([
            f"ε = {metrics['epsilon']:.2e}:",
            f"  {method_display} error: {metrics['cheb_error']:.2e}",
            f"  Function error: {metrics['func_error']:.2e}",
            f"  Error ratio: {metrics['ratio']:.1f}",
            ""
        ])
        
        # Add to CSV with region-specific data
        csv_lines.append(f"{metrics['epsilon']},{metrics['cheb_error']},{metrics['func_error']},{metrics['ratio']},{approx_desired_error},{approx_non_desired_error},{func_desired_error},{func_non_desired_error}")
    
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
    
    # Polynomial approximation quality - use scientific notation for small values
    approx_quality = results['approximation_quality']
    
    # Dynamically get method info
    method_key = results.get('method_name', 'chebyshev')
    method_display = results.get('method_display', 'Chebyshev')
    
    mean_approx = approx_quality[f'mean_{method_key}_error']
    max_approx = approx_quality[f'max_{method_key}_error']
    
    print(f"\n{method_display} Approximation Quality:")
    print(f"  Mean error: {mean_approx:.2e}")
    print(f"  Max error: {max_approx:.2e}")
    approx_desired = approx_quality[f'{method_key}_region_analysis']['desired']
    approx_non_desired = approx_quality[f'{method_key}_region_analysis']['non_desired']
    print(f"  Desired region mean error: {approx_desired['mean_error']:.2e}")
    print(f"  Non-desired region mean error: {approx_non_desired['mean_error']:.2e}")
    
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