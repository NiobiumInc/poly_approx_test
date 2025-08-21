"""
Graph Creation Module

This module handles all plotting and visualization functionality for the 
polynomial approximation analysis system.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
import config
import utils
from indicator_functions import get_indicator_function
from chebyshev_approximation import create_chebyshev_approximation


def generate_plot_filename(plot_type: str, region: str, run_id: str) -> str:
    """Generate standardized plot filenames."""
    return f"{config.FUNCTION_TYPE}_{plot_type}_{region}_errors_{run_id}.png"


def extract_plot_data(results: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract all plotting data from results in one pass."""
    data = {
        'rescaled_epsilons': [],
        'cheb_desired_errors': [],
        'cheb_non_desired_errors': [], 
        'func_desired_errors': [],
        'func_non_desired_errors': []
    }
    
    for result in results['individual_results']:
        epsilon = result['epsilon']
        data['rescaled_epsilons'].append(epsilon / 4.0)
        
        # Get method name dynamically
        method_name = result.get('method_name', 'chebyshev')
        
        # Approximation errors (dynamically named based on method)
        approx_quality = result['approximation_quality'][f'{method_name}_region_analysis']
        data['cheb_desired_errors'].append(approx_quality['desired']['mean_error'])
        data['cheb_non_desired_errors'].append(approx_quality['non_desired']['mean_error'])
        
        # Function errors
        func_performance = result['function_performance']['expected_region_analysis']
        data['func_desired_errors'].append(func_performance['desired']['mean_error'])
        data['func_non_desired_errors'].append(func_performance['non_desired']['mean_error'])
    
    # Convert to numpy arrays
    return {key: np.array(values) for key, values in data.items()}


def get_domain_config():
    """Get domain configuration based on config settings."""
    if config.USE_RESCALED:
        return {
            'x_vals': np.linspace(-1, 1, 1000),
            'desired_point': (config.DESIRED_VALUE / 4.0) - 1,
            'x_label': 'Rescaled Domain',
            'domain_info': '[-1, 1]',
            'domain_range': (-1, 1)
        }
    else:
        return {
            'x_vals': np.linspace(0, config.MAX_VAL, 1000),
            'desired_point': config.DESIRED_VALUE,
            'x_label': 'Original Domain', 
            'domain_info': f'[0, {config.MAX_VAL}]',
            'domain_range': (0, config.MAX_VAL)
        }


def create_error_plot(x_data: np.ndarray, y_data: np.ndarray, color: str, marker: str, 
                     title: str, plots_dir: Path, filename: str, figsize=(10, 6), 
                     label: Optional[str] = None, additional_plots: Optional[List[Dict]] = None):
    """Create a standardized error plot."""
    plt.figure(figsize=figsize)
    plt.plot(x_data, y_data, color=color, marker=marker, linewidth=2, markersize=6, label=label)
    
    # Add additional plot lines if provided
    if additional_plots:
        for plot_data in additional_plots:
            plt.plot(x_data, plot_data['y_data'], color=plot_data['color'], 
                    marker=plot_data['marker'], linewidth=2, markersize=6, 
                    label=plot_data['label'])
    
    plt.xlabel('Rescaled Epsilon')
    plt.ylabel('Average Error')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if label or additional_plots:
        plt.legend()
    
    # Set log scale if data > 0
    import numpy as np
    all_y_values = list(y_data)
    if additional_plots:
        for plot_data in additional_plots:
            all_y_values.extend(plot_data['y_data'])
    
    # Filter out zeros and very small values that can cause log scale issues
    all_y_values = [y for y in all_y_values if y > 0]
    
    if len(all_y_values) > 0 and max(all_y_values) > 0:
        plt.yscale('log')
        
        # Force better ticks for "both" plots
        if "both" in filename.lower():
            y_min, y_max = min(all_y_values), max(all_y_values)
            
            # Force specific ticks based on data range
            log_min = np.floor(np.log10(y_min))
            log_max = np.ceil(np.log10(y_max))
            
            # Create ticks at every order of magnitude in the range
            if log_max - log_min <= 1:  # Less than one full decade
                # Create more ticks within the decade
                base_exp = int(log_min)
                tick_values = []
                for multiplier in [1, 2, 3, 5]:
                    val = multiplier * (10 ** base_exp)
                    if y_min <= val <= y_max * 1.1:  # Add some buffer
                        tick_values.append(val)
                # Add next decade if needed
                for multiplier in [1, 2]:
                    val = multiplier * (10 ** (base_exp + 1))
                    if val <= y_max * 1.1:
                        tick_values.append(val)
                plt.yticks(tick_values)
            else:
                # Multiple decades - use standard approach
                tick_exponents = range(int(log_min), int(log_max) + 1)
                tick_values = [10**exp for exp in tick_exponents]
                plt.yticks(tick_values)
    
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()


def create_function_approximation_plot(results: Dict[str, Any]) -> Path:
    """
    Create a plot showing the actual function, polynomial approximation, and desired point.
    
    - Black line: actual function
    - Red dashed line: Chebyshev polynomial approximation
    - Green vertical dashed line: desired point
    """
    from error_analysis import get_output_paths, calculate_approximation_errors
    output_paths = get_output_paths()
    plots_dir = utils.ensure_directory_exists(output_paths['function_path'])
    
    # Get function and create approximation using the first epsilon value
    function_name = config.FUNCTION_TYPE
    func = get_indicator_function(function_name)
    
    # Use first epsilon for approximation (could use any)
    first_result = results['individual_results'][0]
    epsilon = first_result['epsilon']
    method_display = first_result.get('method_display', 'Chebyshev')
    
    domain_config = get_domain_config()
    x_vals = domain_config['x_vals']
    
    # Get approximation values using the same system as error analysis
    true_values, approx_values = calculate_approximation_errors(
        func, x_vals, epsilon, config.CHEB_DEGREE
    )
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot actual function in black
    plt.plot(x_vals, true_values, 'k-', linewidth=2, 
             label=f'{function_name.title()} (True Function)')
    
    # Plot polynomial approximation in red dashes
    plt.plot(x_vals, approx_values, 'r--', linewidth=2, 
             label=f'{method_display} Approximation (degree {config.CHEB_DEGREE})')
    
    # Plot desired point as vertical green dashed line
    plt.axvline(x=domain_config['desired_point'], color='green', linestyle='--', 
                linewidth=2, label=f'Desired Point ({config.DESIRED_VALUE})')
    
    plt.xlabel(domain_config['x_label'])
    plt.ylabel('Function Value')
    plt.title(f'{function_name.title()} vs {method_display} Approximation\n'
              f'Domain: {domain_config["domain_info"]}, ε = {epsilon:.2e}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot with run ID
    plot_filename = f'{function_name}_function_approximation_{results["run_id"]}.png'
    plt.savefig(plots_dir / plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Function approximation plot: {plot_filename}")
    return plots_dir


def create_connected_dot_plots(results: Dict[str, Any]) -> Path:
    """
    Create connected dot plots showing epsilon vs errors.
    
    Creates 6 separate plots:
    - Chebyshev approximation errors: desired (green), non-desired (purple), both
    - Function classification errors: desired (blue), non-desired (red), both
    """
    from error_analysis import get_output_paths
    output_paths = get_output_paths()
    plots_dir = utils.ensure_directory_exists(output_paths['function_path'])
    
    # Extract all plot data in one pass
    data = extract_plot_data(results)
    run_id = results['run_id']
    
    # Define plot configurations
    plot_configs = [
        # Single region plots
        {
            'y_data': data['cheb_desired_errors'],
            'color': 'g',
            'marker': 'o',
            'title': 'Chebyshev Approximation Error (Desired Region)',
            'filename': generate_plot_filename('chebyshev', 'desired', run_id)
        },
        {
            'y_data': data['cheb_non_desired_errors'],
            'color': 'purple', 
            'marker': 'o',
            'title': 'Chebyshev Approximation Error (Non-Desired Region)',
            'filename': generate_plot_filename('chebyshev', 'non_desired', run_id)
        },
        {
            'y_data': data['func_desired_errors'],
            'color': 'b',
            'marker': 'o', 
            'title': 'Function Classification Error (Desired Region)',
            'filename': generate_plot_filename('function', 'desired', run_id)
        },
        {
            'y_data': data['func_non_desired_errors'],
            'color': 'r',
            'marker': 'o',
            'title': 'Function Classification Error (Non-Desired Region)', 
            'filename': generate_plot_filename('function', 'non_desired', run_id)
        }
    ]
    
    # Create single region plots
    for config_data in plot_configs:
        create_error_plot(
            data['rescaled_epsilons'], 
            config_data['y_data'],
            config_data['color'],
            config_data['marker'],
            f'{config.FUNCTION_TYPE.title()} - {config_data["title"]}',
            plots_dir,
            config_data['filename']
        )
    
    # Create combined plots
    create_error_plot(
        data['rescaled_epsilons'],
        data['cheb_desired_errors'],
        'g', 'o',
        f'{config.FUNCTION_TYPE.title()} - Chebyshev Approximation Error (Both Regions)',
        plots_dir,
        generate_plot_filename('chebyshev', 'both', run_id),
        label='Desired Region',
        additional_plots=[{
            'y_data': data['cheb_non_desired_errors'],
            'color': 'purple',
            'marker': 'o',
            'label': 'Non-Desired Region'
        }]
    )
    
    create_error_plot(
        data['rescaled_epsilons'],
        data['func_desired_errors'], 
        'b', 'o',
        f'{config.FUNCTION_TYPE.title()} - Function Classification Error (Both Regions)',
        plots_dir,
        generate_plot_filename('function', 'both', run_id),
        label='Desired Region',
        additional_plots=[{
            'y_data': data['func_non_desired_errors'],
            'color': 'r',
            'marker': 'o', 
            'label': 'Non-Desired Region'
        }]
    )
    
    print(f"\nError analysis plots created:")
    print(f"  Chebyshev plots: {plots_dir}/*_chebyshev_*.png")
    print(f"  Function plots: {plots_dir}/*_function_*.png")
    
    return plots_dir


def discover_existing_csv_files(base_directory: Path) -> Dict[str, Dict[str, Any]]:
    """
    Discover existing CSV files in the base directory and extract metadata.
    
    Returns:
        Dictionary mapping run_id to {function_type, csv_path, data}
    """
    import pandas as pd
    import glob
    
    csv_files = {}
    
    # Search for all detailed results CSV files
    pattern = str(base_directory / "*" / "*_detailed_results_*.csv")
    
    for csv_path in glob.glob(pattern):
        csv_path = Path(csv_path)
        filename = csv_path.name
        
        # Extract function type and run ID from filename
        # Format: {function_type}_detailed_results_{run_id}.csv
        parts = filename.replace('.csv', '').split('_detailed_results_')
        if len(parts) == 2:
            function_type = parts[0]
            run_id = parts[1]
            
            try:
                # Load CSV data
                data = pd.read_csv(csv_path)
                
                csv_files[run_id] = {
                    'function_type': function_type,
                    'csv_path': csv_path,
                    'data': data
                }
            except Exception as e:
                print(f"Warning: Could not load {csv_path}: {e}")
    
    return csv_files


def create_comparison_plots(base_directory: Path, current_run_data: Dict[str, Any] = None, 
                          specific_run_ids: List[str] = None, specific_directories: List[str] = None):
    """
    Create comparison plots for CSV files. Creates comparison versions of each individual plot type.
    
    Args:
        base_directory: Base directory to search for CSV files
        current_run_data: Data from current run (optional, for context)
        specific_run_ids: List of specific run IDs to compare (manual mode)
        specific_directories: List of specific directories to search in (manual mode)
    """
    # Discover existing CSV files
    if specific_directories:
        # Manual mode: search in specific directories
        all_csv_files = {}
        for directory in specific_directories:
            dir_path = Path(directory) if isinstance(directory, str) else directory
            csv_files = discover_existing_csv_files(dir_path)
            all_csv_files.update(csv_files)
    else:
        # Automatic mode: search in base directory
        all_csv_files = discover_existing_csv_files(base_directory)
    
    # Filter by specific run IDs if provided
    if specific_run_ids:
        csv_files = {run_id: file_info for run_id, file_info in all_csv_files.items() 
                    if run_id in specific_run_ids}
        if len(csv_files) < len(specific_run_ids):
            missing_ids = set(specific_run_ids) - set(csv_files.keys())
            print(f"Warning: Could not find CSV files for run IDs: {missing_ids}")
    else:
        csv_files = all_csv_files
    
    if len(csv_files) < 2:
        print(f"Found only {len(csv_files)} CSV files. Need at least 2 for comparison.")
        if specific_run_ids:
            print(f"Available run IDs: {list(all_csv_files.keys())}")
        return
    
    mode_desc = "manual" if (specific_run_ids or specific_directories) else "automatic"
    print(f"\nCreating comparison plots for {len(csv_files)} runs (mode: {mode_desc})...")
    print(f"Run IDs: {list(csv_files.keys())}")
    
    # Create comparison versions of each individual plot type
    _create_all_comparison_plots(csv_files, base_directory)


def _create_all_comparison_plots(csv_files: Dict[str, Dict], base_directory: Path):
    """
    Create comparison versions of each individual plot type.
    This matches the 6 individual plot types but with multiple runs overlaid.
    """
    # Create output directory
    output_dir = utils.ensure_directory_exists(base_directory / "comparisons")
    
    # Generate plot filename suffix with all run IDs
    run_ids = sorted(csv_files.keys())
    ids_string = "_".join(run_ids)
    
    # Extract plot data for all runs
    all_plot_data = {}
    for run_id, file_info in csv_files.items():
        data = file_info['data']
        rescaled_epsilons = data['epsilon'] / 4.0  # Convert to rescaled epsilon
        
        # Check if the CSV has the new region-specific columns
        required_columns = ['cheb_desired_error', 'cheb_non_desired_error', 'func_desired_error', 'func_non_desired_error']
        if all(col in data.columns for col in required_columns):
            # Use region-specific data from enhanced CSV
            all_plot_data[run_id] = {
                'function_type': file_info['function_type'],
                'rescaled_epsilons': rescaled_epsilons,
                'cheb_desired_errors': data['cheb_desired_error'],
                'cheb_non_desired_errors': data['cheb_non_desired_error'],
                'func_desired_errors': data['func_desired_error'],
                'func_non_desired_errors': data['func_non_desired_error'],
            }
        else:
            # Fall back to total errors for old CSV format
            print(f"Warning: {run_id} using total errors (old CSV format)")
            all_plot_data[run_id] = {
                'function_type': file_info['function_type'],
                'rescaled_epsilons': rescaled_epsilons,
                'cheb_desired_errors': data['chebyshev_error'],
                'cheb_non_desired_errors': data['chebyshev_error'],
                'func_desired_errors': data['function_error'],
                'func_non_desired_errors': data['function_error'],
            }
    
    # Define the 6 comparison plots to create (matching individual plot structure)
    plot_configs = [
        {
            'y_data_key': 'cheb_desired_errors',
            'color': 'g',
            'marker': 'o',
            'title': 'Chebyshev Approximation Error (Desired Region)',
            'filename_base': 'chebyshev_desired_errors'
        },
        {
            'y_data_key': 'cheb_non_desired_errors', 
            'color': 'purple',
            'marker': 'o',
            'title': 'Chebyshev Approximation Error (Non-Desired Region)',
            'filename_base': 'chebyshev_non_desired_errors'
        },
        {
            'y_data_key': 'func_desired_errors',
            'color': 'b',
            'marker': 'o',
            'title': 'Function Classification Error (Desired Region)',
            'filename_base': 'function_desired_errors'
        },
        {
            'y_data_key': 'func_non_desired_errors',
            'color': 'r', 
            'marker': 'o',
            'title': 'Function Classification Error (Non-Desired Region)',
            'filename_base': 'function_non_desired_errors'
        }
    ]
    
    # Create individual comparison plots
    for plot_config in plot_configs:
        plt.figure(figsize=(10, 6))
        
        # Plot each run
        for i, (run_id, plot_data) in enumerate(all_plot_data.items()):
            func_type = plot_data['function_type']
            color = plt.cm.Set1(i / len(all_plot_data))  # Different color for each run
            
            label = f"{func_type.title()} ({run_id})"
            
            plt.plot(
                plot_data['rescaled_epsilons'], 
                plot_data[plot_config['y_data_key']], 
                color=color, 
                marker=plot_config['marker'], 
                linewidth=2, 
                markersize=6,
                label=label
            )
        
        plt.xlabel('Rescaled Epsilon')
        plt.ylabel('Average Error')
        plt.title(f'Comparison - {plot_config["title"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set log scale if data > 0
        y_max = max([max(plot_data[plot_config['y_data_key']]) for plot_data in all_plot_data.values()])
        if y_max > 0:
            plt.yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{plot_config['filename_base']}_comparison_{ids_string}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")
    
    # Create the "both regions" comparison plots
    _create_both_regions_comparison_plots(all_plot_data, output_dir, ids_string)


def _create_both_regions_comparison_plots(all_plot_data: Dict, output_dir: Path, ids_string: str):
    """Create the 'both regions' comparison plots."""
    
    # Chebyshev Both Regions
    plt.figure(figsize=(10, 6))
    for i, (run_id, plot_data) in enumerate(all_plot_data.items()):
        func_type = plot_data['function_type']
        color = plt.cm.Set1(i / len(all_plot_data))
        
        plt.plot(plot_data['rescaled_epsilons'], plot_data['cheb_desired_errors'], 
                color=color, marker='o', linewidth=2, markersize=6, 
                label=f"{func_type.title()} ({run_id}) - Desired", linestyle='-')
        plt.plot(plot_data['rescaled_epsilons'], plot_data['cheb_non_desired_errors'],
                color=color, marker='s', linewidth=2, markersize=6,
                label=f"{func_type.title()} ({run_id}) - Non-Desired", linestyle='--', alpha=0.7)
    
    plt.xlabel('Rescaled Epsilon')
    plt.ylabel('Average Error')
    plt.title('Comparison - Chebyshev Approximation Error (Both Regions)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    
    filename = f"chebyshev_both_errors_comparison_{ids_string}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")
    
    # Function Both Regions
    plt.figure(figsize=(10, 6))
    for i, (run_id, plot_data) in enumerate(all_plot_data.items()):
        func_type = plot_data['function_type']
        color = plt.cm.Set1(i / len(all_plot_data))
        
        plt.plot(plot_data['rescaled_epsilons'], plot_data['func_desired_errors'],
                color=color, marker='o', linewidth=2, markersize=6,
                label=f"{func_type.title()} ({run_id}) - Desired", linestyle='-')
        plt.plot(plot_data['rescaled_epsilons'], plot_data['func_non_desired_errors'],
                color=color, marker='s', linewidth=2, markersize=6,
                label=f"{func_type.title()} ({run_id}) - Non-Desired", linestyle='--', alpha=0.7)
    
    plt.xlabel('Rescaled Epsilon')
    plt.ylabel('Average Error')
    plt.title('Comparison - Function Classification Error (Both Regions)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Collect all y data for proper range calculation
    import numpy as np
    all_y_data = []
    for plot_data in all_plot_data.values():
        all_y_data.extend(plot_data['func_desired_errors'])
        all_y_data.extend(plot_data['func_non_desired_errors'])
    y_max = max(all_y_data)
    
    if y_max > 0:
        plt.yscale('log')
    
    plt.tight_layout()
    
    filename = f"function_both_errors_comparison_{ids_string}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def create_automatic_comparisons(results: Dict[str, Any]):
    """
    Automatically create comparison plots after individual analysis.
    This function is called from main.py after individual plots are created.
    """
    from error_analysis import get_output_paths
    
    output_paths = get_output_paths()
    base_directory = output_paths['base_path']
    
    print(f"\n{'='*60}")
    print("CREATING COMPARISON PLOTS")
    print(f"{'='*60}")
    
    create_comparison_plots(base_directory, results)
    
    print(f"Comparison plots saved to: {base_directory / 'comparisons'}")
    
    return base_directory / 'comparisons'