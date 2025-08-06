"""
Experiment Runner Module

This module orchestrates comprehensive polynomial approximation experiments,
combining error analysis, visualization, and result logging.
"""

import time
import csv
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import config
import utils
from error_analysis import run_complete_error_analysis
from indicator_functions import get_indicator_function


def get_organized_output_paths(date_folder: str = None):
    """
    Get organized output paths with date-based folders.
    
    Args:
        date_folder: Date folder name (defaults to 'aug6')
        
    Returns:
        Dictionary with organized paths
    """
    if date_folder is None:
        date_folder = "aug6"
    
    base_path = Path("graphs") / date_folder
    log_path = base_path / "experiment_log.csv"
    
    return {
        'base_path': base_path,
        'impulse_path': base_path / "impulse",
        'plateau_reg_path': base_path / "plateau_reg", 
        'plateau_sine_path': base_path / "plateau_sine",
        'comparison_path': base_path / "comparisons",
        'log_path': log_path
    }


def run_single_experiment(
    function_name: str,
    epsilon: float,
    degree: int = None,
    random_seed: int = None
) -> Dict[str, Any]:
    """
    Run a complete experiment for a single function and epsilon value.
    
    Args:
        function_name: Name of the indicator function ('impulse', 'plateau_reg', 'plateau_sine')
        epsilon: Epsilon value for approximate integer generation
        degree: Chebyshev polynomial degree (defaults to 119)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing experiment results
    """
    if degree is None:
        degree = 119
        
    print(f"Running experiment: {function_name} (ε={epsilon}, degree={degree})")
    
    # Run complete error analysis
    results = run_complete_error_analysis(
        function_name=function_name,
        epsilon=epsilon,
        degree=degree,
        random_seed=random_seed
    )
    
    return results


def run_epsilon_sweep_experiment(
    function_name: str,
    epsilon_values: List[float], 
    degree: int = None,
    random_seed: int = None
) -> Dict[str, Any]:
    """
    Run experiments across multiple epsilon values for a single function.
    
    Args:
        function_name: Name of the indicator function
        epsilon_values: List of epsilon values to test
        degree: Chebyshev polynomial degree (defaults to 119)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing aggregated results across all epsilon values
    """
    if degree is None:
        degree = 119
        
    run_id = utils.create_unique_id(8)
    print(f"Starting epsilon sweep experiment (ID: {run_id})")
    print(f"Function: {function_name}")
    print(f"Epsilon values: {epsilon_values}")
    print(f"Degree: {degree}")
    
    results = {
        'run_id': run_id,
        'function_name': function_name,
        'degree': degree,
        'epsilon_values': epsilon_values,
        'individual_results': [],
        'summary': {
            'chebyshev_errors': [],
            'function_errors': [],
            'error_ratios': []
        }
    }
    
    # Run experiment for each epsilon value
    for epsilon in epsilon_values:
        experiment_result = run_single_experiment(
            function_name=function_name,
            epsilon=epsilon,
            degree=degree,
            random_seed=random_seed
        )
        
        results['individual_results'].append(experiment_result)
        
        # Extract summary statistics from error analysis structure
        approx_quality = experiment_result['approximation_quality']
        func_performance = experiment_result['function_performance']
        
        chebyshev_error = approx_quality['mean_chebyshev_error']
        function_error = func_performance['mean_expected_error']
        error_ratio = function_error / chebyshev_error if chebyshev_error > 0 else float('inf')
        
        results['summary']['chebyshev_errors'].append(chebyshev_error)
        results['summary']['function_errors'].append(function_error)
        results['summary']['error_ratios'].append(error_ratio)
        
        print(f"  ε={epsilon}: Chebyshev error = {chebyshev_error:.2e}, "
              f"Function error = {function_error:.2e}")
    
    # Create summary visualization
    _create_epsilon_sweep_plot(results)
    
    # Log experiment
    _log_epsilon_sweep_experiment(results)
    
    print(f"Epsilon sweep experiment completed (ID: {run_id})")
    
    return results


def run_multi_function_experiment(
    function_names: List[str],
    epsilon_values: List[float],
    degree: int = None,
    random_seed: int = None
) -> Dict[str, Any]:
    """
    Run experiments across multiple functions and epsilon values.
    
    Args:
        function_names: List of indicator function names
        epsilon_values: List of epsilon values to test
        degree: Chebyshev polynomial degree (defaults to 119)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing all experiment results
    """
    if degree is None:
        degree = 119
        
    master_run_id = utils.create_unique_id(8)
    print(f"Starting multi-function experiment (ID: {master_run_id})")
    print(f"Functions: {function_names}")
    print(f"Epsilon values: {epsilon_values}")
    print(f"Degree: {degree}")
    
    results = {
        'master_run_id': master_run_id,
        'function_names': function_names,
        'epsilon_values': epsilon_values,
        'degree': degree,
        'function_results': {}
    }
    
    # Run epsilon sweep for each function
    for function_name in function_names:
        print(f"\n--- Processing function: {function_name} ---")
        function_result = run_epsilon_sweep_experiment(
            function_name=function_name,
            epsilon_values=epsilon_values,
            degree=degree,
            random_seed=random_seed
        )
        results['function_results'][function_name] = function_result
    
    # Create comparison visualization
    _create_multi_function_comparison_plot(results)
    
    # Log master experiment
    _log_multi_function_experiment(results)
    
    print(f"\nMulti-function experiment completed (ID: {master_run_id})")
    
    return results


def _create_epsilon_sweep_plot(results: Dict[str, Any]) -> None:
    """Create visualization for epsilon sweep experiment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epsilon_values = results['epsilon_values']
    chebyshev_errors = results['summary']['chebyshev_errors']
    function_errors = results['summary']['function_errors']
    
    # Plot 1: Error comparison
    ax1.semilogy(epsilon_values, chebyshev_errors, 'b-o', label='Chebyshev Error', markersize=6)
    ax1.semilogy(epsilon_values, function_errors, 'r-s', label='Function Error', markersize=6)
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title(f'{results["function_name"].title()} Error Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error ratio
    error_ratios = results['summary']['error_ratios']
    ax2.plot(epsilon_values, error_ratios, 'g-^', label='Function/Chebyshev Ratio', markersize=6)
    ax2.set_xlabel('Epsilon')
    ax2.set_ylabel('Error Ratio')
    ax2.set_title('Function vs Chebyshev Error Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot in organized directory structure
    output_paths = get_organized_output_paths()
    function_name = results['function_name']
    
    if function_name == 'impulse':
        save_dir = utils.ensure_directory_exists(output_paths['impulse_path'])
    elif function_name == 'plateau_reg':
        save_dir = utils.ensure_directory_exists(output_paths['plateau_reg_path'])
    elif function_name == 'plateau_sine':
        save_dir = utils.ensure_directory_exists(output_paths['plateau_sine_path'])
    else:
        save_dir = utils.ensure_directory_exists(output_paths['base_path'])
    
    plot_path = save_dir / f"{function_name}_epsilon_sweep_{results['run_id']}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Epsilon sweep plot saved: {plot_path}")


def _create_multi_function_comparison_plot(results: Dict[str, Any]) -> None:
    """Create comparison visualization across multiple functions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epsilon_values = results['epsilon_values']
    function_names = results['function_names']
    
    # Plot 1: Chebyshev errors
    ax1 = axes[0, 0]
    for func_name in function_names:
        func_result = results['function_results'][func_name]
        chebyshev_errors = func_result['summary']['chebyshev_errors']
        ax1.semilogy(epsilon_values, chebyshev_errors, '-o', label=func_name.replace('_', ' ').title(), markersize=5)
    
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Chebyshev Error')
    ax1.set_title('Chebyshev Approximation Errors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Function errors
    ax2 = axes[0, 1]
    for func_name in function_names:
        func_result = results['function_results'][func_name]
        function_errors = func_result['summary']['function_errors']
        ax2.semilogy(epsilon_values, function_errors, '-s', label=func_name.replace('_', ' ').title(), markersize=5)
    
    ax2.set_xlabel('Epsilon')
    ax2.set_ylabel('Function Classification Error')
    ax2.set_title('Function Classification Errors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error ratios
    ax3 = axes[1, 0]
    for func_name in function_names:
        func_result = results['function_results'][func_name]
        error_ratios = func_result['summary']['error_ratios']
        ax3.plot(epsilon_values, error_ratios, '-^', label=func_name.replace('_', ' ').title(), markersize=5)
    
    ax3.set_xlabel('Epsilon')
    ax3.set_ylabel('Error Ratio')
    ax3.set_title('Function/Chebyshev Error Ratios')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table data
    table_data = []
    for func_name in function_names:
        func_result = results['function_results'][func_name]
        mean_cheb_error = np.mean(func_result['summary']['chebyshev_errors'])
        mean_func_error = np.mean(func_result['summary']['function_errors'])
        mean_ratio = np.mean(func_result['summary']['error_ratios'])
        
        table_data.append([
            func_name.replace('_', ' ').title(),
            f"{mean_cheb_error:.2e}",
            f"{mean_func_error:.2e}",
            f"{mean_ratio:.1f}"
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=['Function', 'Avg Cheb Error', 'Avg Func Error', 'Avg Ratio'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title('Summary Statistics', y=0.8)
    
    plt.tight_layout()
    
    # Save plot in organized directory structure
    output_paths = get_organized_output_paths()
    save_dir = utils.ensure_directory_exists(output_paths['comparison_path'])
    
    plot_path = save_dir / f"multi_function_comparison_{results['master_run_id']}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Multi-function comparison plot saved: {plot_path}")


def _log_epsilon_sweep_experiment(results: Dict[str, Any]) -> None:
    """Log epsilon sweep experiment results to CSV."""
    output_paths = get_organized_output_paths()
    utils.ensure_directory_exists(output_paths['base_path'])
    log_path = output_paths['log_path']
    
    # Prepare log entry
    log_entry = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'run_id': results['run_id'],
        'experiment_type': 'epsilon_sweep',
        'function_name': results['function_name'],
        'degree': results['degree'],
        'epsilon_values': str(results['epsilon_values']),
        'num_epsilon_values': len(results['epsilon_values']),
        'mean_chebyshev_error': np.mean(results['summary']['chebyshev_errors']),
        'mean_function_error': np.mean(results['summary']['function_errors']),
        'mean_error_ratio': np.mean(results['summary']['error_ratios'])
    }
    
    # Write to CSV using centralized logging
    utils.log_to_csv(log_path, log_entry)
    
    print(f"Experiment logged to: {log_path}")


def _log_multi_function_experiment(results: Dict[str, Any]) -> None:
    """Log multi-function experiment results to CSV."""
    output_paths = get_organized_output_paths()
    utils.ensure_directory_exists(output_paths['base_path'])
    log_path = output_paths['log_path']
    
    # Prepare log entry
    log_entry = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'run_id': results['master_run_id'],
        'experiment_type': 'multi_function',
        'function_name': str(results['function_names']),
        'degree': results['degree'],
        'epsilon_values': str(results['epsilon_values']),
        'num_epsilon_values': len(results['epsilon_values']),
        'mean_chebyshev_error': 'varies',
        'mean_function_error': 'varies',
        'mean_error_ratio': 'varies'
    }
    
    # Write to CSV using centralized logging
    utils.log_to_csv(log_path, log_entry)
    
    print(f"Multi-function experiment logged to: {log_path}")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing experiment runner...")
    
    # Test single experiment
    result = run_single_experiment("impulse", 0.05)
    print(f"Single experiment completed: {result['function_name']} ε={result['epsilon']}")
    
    # Test epsilon sweep
    epsilon_sweep_result = run_epsilon_sweep_experiment(
        "impulse", 
        [0.01, 0.05, 0.1]
    )
    print(f"Epsilon sweep completed: {epsilon_sweep_result['run_id']}")
    
    # Test multi-function experiment
    multi_result = run_multi_function_experiment(
        ["impulse", "plateau_reg"],
        [0.05, 0.1]
    )
    print(f"Multi-function experiment completed: {multi_result['master_run_id']}")