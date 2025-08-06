#!/usr/bin/env python3
"""
Main Application Runner

This runs polynomial approximation analysis based on config.py settings.
It analyzes the configured function with the configured epsilon range.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import time
import matplotlib.pyplot as plt

import config
import utils
from error_analysis import run_complete_error_analysis
from indicator_functions import get_indicator_function
from chebyshev_approximation import create_chebyshev_approximation


def get_output_paths():
    """Get organized output paths based on config."""
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
    print(f"  Degree: {config.CHEB_DEGREE}")
    print("=" * 60)


def generate_epsilon_values():
    """Generate epsilon values based on config settings."""
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
    print_banner()
    
    # Generate unique run ID
    run_id = utils.create_unique_id(8)
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


def create_function_approximation_plot(results: Dict[str, Any]):
    """
    Create a plot showing the actual function, Chebyshev approximation, and desired point.
    
    - Black line: actual function
    - Red dashed line: Chebyshev approximation 
    - Green vertical dashed line: desired point
    """
    output_paths = get_output_paths()
    plots_dir = utils.ensure_directory_exists(output_paths['function_path'])
    
    # Get function and create approximation using the first epsilon value
    function_name = config.FUNCTION_TYPE
    func = get_indicator_function(function_name)
    
    # Use first epsilon for approximation (could use any)
    first_result = results['individual_results'][0]
    epsilon = first_result['epsilon']
    
    # Create Chebyshev approximation - need to create a lambda that ignores epsilon
    func_no_epsilon = lambda x: func(x, epsilon)
    
    if config.USE_RESCALED:
        # Rescaled domain [-1, 1]
        cheb_approx = create_chebyshev_approximation(func_no_epsilon, -1, 1, config.CHEB_DEGREE)
    else:
        # Original domain [0, MAX_VAL]
        cheb_approx = create_chebyshev_approximation(func_no_epsilon, 0, config.MAX_VAL, config.CHEB_DEGREE)
    
    # Set up domain
    if config.USE_RESCALED:
        # Rescaled domain [-1, 1]
        x_vals = np.linspace(-1, 1, 1000)
        desired_point = (config.DESIRED_VALUE / 4.0) - 1  # Rescaled desired point
        x_label = 'Rescaled Domain'
        domain_info = '[-1, 1]'
    else:
        # Original domain [0, MAX_VAL]
        x_vals = np.linspace(0, config.MAX_VAL, 1000)
        desired_point = config.DESIRED_VALUE
        x_label = 'Original Domain'
        domain_info = f'[0, {config.MAX_VAL}]'
    
    # Evaluate functions (handle vectorization)
    true_values = np.array([func(x, epsilon) for x in x_vals])
    approx_values = cheb_approx(x_vals)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot actual function in black
    plt.plot(x_vals, true_values, 'k-', linewidth=2, label=f'{function_name.title()} (True Function)')
    
    # Plot Chebyshev approximation in red dashes
    plt.plot(x_vals, approx_values, 'r--', linewidth=2, label=f'Chebyshev Approximation (degree {config.CHEB_DEGREE})')
    
    # Plot desired point as vertical green dashed line
    plt.axvline(x=desired_point, color='green', linestyle='--', linewidth=2, label=f'Desired Point ({config.DESIRED_VALUE})')
    
    plt.xlabel(x_label)
    plt.ylabel('Function Value')
    plt.title(f'{function_name.title()} vs Chebyshev Approximation\nDomain: {domain_info}, ε = {epsilon:.2e}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot with run ID
    plot_filename = f'{function_name}_function_approximation_{results["run_id"]}.png'
    plt.savefig(plots_dir / plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Function approximation plot: {plot_filename}")
    
    return plots_dir


def create_connected_dot_plots(results: Dict[str, Any]):
    """
    Create connected dot plots showing epsilon vs errors.
    
    Creates 6 separate plots:
    - Chebyshev approximation errors: desired (green), non-desired (purple), both
    - Function classification errors: desired (blue), non-desired (red), both
    """
    output_paths = get_output_paths()
    plots_dir = utils.ensure_directory_exists(output_paths['function_path'])
    
    # Extract data for plotting
    rescaled_epsilons = []
    cheb_desired_errors = []
    cheb_non_desired_errors = []
    cheb_both_errors = []
    func_desired_errors = []
    func_non_desired_errors = []
    func_both_errors = []
    
    for result in results['individual_results']:
        epsilon = result['epsilon']
        rescaled_epsilon = epsilon / 4.0  # Convert to rescaled epsilon
        rescaled_epsilons.append(rescaled_epsilon)
        
        # Chebyshev errors
        cheb_quality = result['approximation_quality']['chebyshev_region_analysis']
        cheb_desired_errors.append(cheb_quality['desired']['mean_error'])
        cheb_non_desired_errors.append(cheb_quality['non_desired']['mean_error'])
        cheb_both_errors.append(result['approximation_quality']['mean_chebyshev_error'])
        
        # Function errors
        func_performance = result['function_performance']['expected_region_analysis']
        func_desired_errors.append(func_performance['desired']['mean_error'])
        func_non_desired_errors.append(func_performance['non_desired']['mean_error'])
        func_both_errors.append(result['function_performance']['mean_expected_error'])
    
    # Convert to numpy arrays for easier plotting
    rescaled_epsilons = np.array(rescaled_epsilons)
    
    # Plot 1: Chebyshev Approximation - Desired Region (Green)
    plt.figure(figsize=(10, 6))
    plt.plot(rescaled_epsilons, cheb_desired_errors, 'g-o', linewidth=2, markersize=6)
    plt.xlabel('Rescaled Epsilon')
    plt.ylabel('Average Error')
    plt.title(f'{config.FUNCTION_TYPE.title()} - Chebyshev Approximation Error (Desired Region)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(plots_dir / f'{config.FUNCTION_TYPE}_chebyshev_desired_errors_{results["run_id"]}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Chebyshev Approximation - Non-Desired Region (Purple)
    plt.figure(figsize=(10, 6))
    plt.plot(rescaled_epsilons, cheb_non_desired_errors, 'purple', marker='o', linewidth=2, markersize=6)
    plt.xlabel('Rescaled Epsilon')
    plt.ylabel('Average Error')
    plt.title(f'{config.FUNCTION_TYPE.title()} - Chebyshev Approximation Error (Non-Desired Region)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(plots_dir / f'{config.FUNCTION_TYPE}_chebyshev_non_desired_errors_{results["run_id"]}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Chebyshev Approximation - Both Regions (Desired and Non-Desired together)
    plt.figure(figsize=(10, 6))
    plt.plot(rescaled_epsilons, cheb_desired_errors, 'g-o', linewidth=2, markersize=6, label='Desired Region')
    plt.plot(rescaled_epsilons, cheb_non_desired_errors, 'purple', marker='o', linewidth=2, markersize=6, label='Non-Desired Region')
    plt.xlabel('Rescaled Epsilon')
    plt.ylabel('Average Error')
    plt.title(f'{config.FUNCTION_TYPE.title()} - Chebyshev Approximation Error (Both Regions)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(plots_dir / f'{config.FUNCTION_TYPE}_chebyshev_both_errors_{results["run_id"]}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Function Classification - Desired Region (Blue)
    plt.figure(figsize=(10, 6))
    plt.plot(rescaled_epsilons, func_desired_errors, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Rescaled Epsilon')
    plt.ylabel('Average Error')
    plt.title(f'{config.FUNCTION_TYPE.title()} - Function Classification Error (Desired Region)')
    plt.grid(True, alpha=0.3)
    if max(func_desired_errors) > 0:
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(plots_dir / f'{config.FUNCTION_TYPE}_function_desired_errors_{results["run_id"]}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Function Classification - Non-Desired Region (Red)
    plt.figure(figsize=(10, 6))
    plt.plot(rescaled_epsilons, func_non_desired_errors, 'r-o', linewidth=2, markersize=6)
    plt.xlabel('Rescaled Epsilon')
    plt.ylabel('Average Error')
    plt.title(f'{config.FUNCTION_TYPE.title()} - Function Classification Error (Non-Desired Region)')
    plt.grid(True, alpha=0.3)
    if max(func_non_desired_errors) > 0:
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(plots_dir / f'{config.FUNCTION_TYPE}_function_non_desired_errors_{results["run_id"]}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Function Classification - Both Regions (Desired and Non-Desired together)
    plt.figure(figsize=(10, 6))
    plt.plot(rescaled_epsilons, func_desired_errors, 'b-o', linewidth=2, markersize=6, label='Desired Region')
    plt.plot(rescaled_epsilons, func_non_desired_errors, 'r-o', linewidth=2, markersize=6, label='Non-Desired Region')
    plt.xlabel('Rescaled Epsilon')
    plt.ylabel('Average Error')
    plt.title(f'{config.FUNCTION_TYPE.title()} - Function Classification Error (Both Regions)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if max(max(func_desired_errors), max(func_non_desired_errors)) > 0:
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(plots_dir / f'{config.FUNCTION_TYPE}_function_both_errors_{results["run_id"]}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nError analysis plots created:")
    print(f"  Chebyshev plots: {plots_dir}/*_chebyshev_*.png")
    print(f"  Function plots: {plots_dir}/*_function_*.png")
    
    return plots_dir


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
    
    # Add epsilon-specific results
    for result in results['individual_results']:
        epsilon = result['epsilon']
        cheb_error = result['approximation_quality']['mean_chebyshev_error']
        func_error = result['function_performance']['mean_expected_error']
        ratio = func_error / cheb_error if cheb_error > 0 else float('inf')
        
        summary_lines.extend([
            f"ε = {epsilon:.2e}:",
            f"  Chebyshev error: {cheb_error:.2e}",
            f"  Function error: {func_error:.2e}",
            f"  Error ratio: {ratio:.1f}",
            ""
        ])
    
    # Add output locations
    summary_lines.extend([
        "OUTPUT LOCATIONS:",
        "-" * 17,
        f"Base directory: {output_paths['base_path']}",
        f"Function plots: {output_paths['function_path']}",
        f"Summary: {output_paths['base_path'] / 'analysis_summary.txt'}"
    ])
    
    # Save summary
    summary_path = output_paths['base_path'] / 'analysis_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # Save detailed results as CSV
    csv_path = output_paths['function_path'] / f"{config.FUNCTION_TYPE}_detailed_results_{results['run_id']}.csv"
    
    # Create CSV data
    csv_lines = ['epsilon,chebyshev_error,function_error,error_ratio']
    
    for result in results['individual_results']:
        epsilon = result['epsilon']
        cheb_error = result['approximation_quality']['mean_chebyshev_error']
        func_error = result['function_performance']['mean_expected_error']
        ratio = func_error / cheb_error if cheb_error > 0 else float('inf')
        
        csv_lines.append(f"{epsilon},{cheb_error},{func_error},{ratio}")
    
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines))
    
    print(f"\nResults saved:")
    print(f"  Summary: {summary_path}")
    print(f"  Detailed CSV: {csv_path}")


def main():
    """Main application entry point."""
    try:
        # Run analysis based on config
        results = run_config_based_analysis()
        
        # Create comprehensive visualizations
        print(f"\n{'='*60}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*60}")
        
        # Create function approximation plot
        create_function_approximation_plot(results)
        
        # Create error analysis plots
        create_connected_dot_plots(results)
        
        # Save results
        save_results(results)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()