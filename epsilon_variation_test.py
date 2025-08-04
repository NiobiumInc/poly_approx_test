"""
Epsilon Variation Testing for Chebyshev Approximations

This module tests how well Chebyshev approximations of indicator functions
perform when evaluated on approximate integer points with varying epsilon values.
"""

import numpy as np
import math
import csv
import os
from typing import List, Dict, Any, Tuple
from indicator_functions import FunctionConfig, get_function_by_name
from chebyshev_approximation import ChebyshevApproximator
from generate_approx_integers import (
    generate_approx_integers_excluding, 
    generate_approx_desired_value,
    rescale_to_unit_interval
)
from config import GRAPHS_BASE_PATH


class EpsilonVariationTester:
    """
    Test Chebyshev approximations across different epsilon values.
    """
    
    def __init__(self, base_config: FunctionConfig):
        self.base_config = base_config
        self.results = []
        
    def generate_epsilon_values(self, min_eps: float = 0.001, max_eps: float = 0.4, 
                               num_eps: int = 20) -> List[float]:
        """
        Generate evenly distributed epsilon values.
        
        Args:
            min_eps: Minimum epsilon value
            max_eps: Maximum epsilon value  
            num_eps: Number of epsilon values to generate
            
        Returns:
            List of epsilon values
        """
        return np.linspace(min_eps, max_eps, num_eps).tolist()
    
    def generate_test_points(self, epsilon: float, n_excluding: int = 800, 
                           n_desired: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate approximate integer test points for given epsilon, ensuring proper separation.
        
        Args:
            epsilon: Maximum deviation from integers
            n_excluding: Number of points excluding desired value
            n_desired: Number of points near desired value
            
        Returns:
            Tuple of (excluding_points, desired_points)
        """
        # Ensure epsilon doesn't cause overlap issues
        safe_epsilon = min(epsilon, 0.4)  # Keep some margin from 0.5
        
        # Generate points excluding desired value, but filter out any that might
        # accidentally fall in the desired range due to large epsilon
        max_attempts = n_excluding * 3
        excluding_points = []
        attempts = 0
        
        desired_range_low = self.base_config.desired_value - 0.5
        desired_range_high = self.base_config.desired_value + 0.5
        
        while len(excluding_points) < n_excluding and attempts < max_attempts:
            # Generate a batch of candidates
            candidates = generate_approx_integers_excluding(
                n=n_excluding - len(excluding_points),
                max_value=self.base_config.max_val,
                excluded_value=self.base_config.desired_value,
                epsilon=safe_epsilon
            )
            
            # Filter out any that accidentally fall in desired range
            for point in candidates:
                if not (desired_range_low <= point <= desired_range_high):
                    excluding_points.append(point)
                    if len(excluding_points) >= n_excluding:
                        break
            
            attempts += 1
        
        # If we still don't have enough, fill with points far from desired range
        while len(excluding_points) < n_excluding:
            # Generate points far from desired value, ensuring they stay in [0, max_val]
            safe_integers = [i for i in range(int(self.base_config.max_val) + 1) 
                           if abs(i - self.base_config.desired_value) >= 2]
            if safe_integers:
                base_int = np.random.choice(safe_integers)
                point = base_int + np.random.uniform(-safe_epsilon, safe_epsilon)
                # Clip to domain bounds
                point = max(0, min(self.base_config.max_val, point))
                excluding_points.append(point)
        
        excluding_points = np.array(excluding_points[:n_excluding])
        
        # Clip all points to domain bounds [0, max_val]
        excluding_points = np.clip(excluding_points, 0, self.base_config.max_val)
        
        # Generate points near desired value
        desired_points = generate_approx_desired_value(
            n=n_desired,
            desired_value=self.base_config.desired_value,
            epsilon=safe_epsilon
        )
        
        # Clip desired points to domain bounds as well
        desired_points = np.clip(desired_points, 0, self.base_config.max_val)
        
        return excluding_points, desired_points
    
    def evaluate_approximation_quality(self, function_name: str, epsilon: float,
                                     cheb_degree: int = 15, use_rescaled: bool = False) -> Dict[str, Any]:
        """
        Evaluate Chebyshev approximation quality for given parameters.
        
        Args:
            function_name: Name of indicator function
            epsilon: Epsilon value for approximate integers
            cheb_degree: Degree of Chebyshev approximation
            use_rescaled: Whether to use rescaled domain
            
        Returns:
            Dictionary with evaluation results
        """
        # Create configuration
        config = FunctionConfig(
            id="",
            function_name=function_name,
            cheb_degree=cheb_degree,
            desired_value=self.base_config.desired_value,
            impulse_mu=self.base_config.impulse_mu,
            impulse_sigma=self.base_config.impulse_sigma,
            use_rescaled=use_rescaled,
            epsilons=epsilon
        )
        config.id = config.generate_unique_id()
        
        # Generate test points
        excluding_points, desired_points = self.generate_test_points(epsilon)
        all_points = np.concatenate([excluding_points, desired_points])
        
        # Handle rescaling if needed
        if use_rescaled:
            # Rescale test points to [-1, 1]
            rescaled_points = rescale_to_unit_interval(all_points, max_value=config.max_val)
            eval_points = rescaled_points.tolist()
            domain = (-1.0, 1.0)
            
            # Calculate rescaled epsilon for logging
            rescaled_epsilon = epsilon * (2.0 / config.max_val)
        else:
            eval_points = all_points.tolist()
            domain = (0.0, float(config.max_val))
            rescaled_epsilon = None
        
        # Create and train Chebyshev approximation
        approximator = ChebyshevApproximator(config)
        results = approximator.run_approximation_analysis(
            a=domain[0], b=domain[1], num_eval_points=100
        )
        
        # Evaluate approximation on test points
        true_values = [approximator.function(x) for x in eval_points]
        approx_values = approximator.evaluate_approximation(eval_points)
        
        # Calculate errors comparing Chebyshev approximation to target values (0 or 1)
        # Target assignment based on whether point is in desired range [desired_value-0.5, desired_value+0.5]
        if use_rescaled:
            # Rescale the desired range to match the rescaled domain
            range_points = rescale_to_unit_interval(
                np.array([config.desired_value - 0.5, config.desired_value + 0.5]), 
                max_value=config.max_val
            )
            desired_range_low = range_points[0]
            desired_range_high = range_points[1]
        else:
            desired_range_low = config.desired_value - 0.5
            desired_range_high = config.desired_value + 0.5
        
        target_errors_excluding = []  # Points outside desired range, target = 0
        target_errors_desired = []    # Points inside desired range, target = 1
        
        for i, (eval_point, approx_val) in enumerate(zip(eval_points, approx_values)):
            # Determine if point is in desired range
            if desired_range_low <= eval_point <= desired_range_high:
                # Point is in desired range, target = 1
                error = abs(approx_val - 1.0)
                target_errors_desired.append(error)
            else:
                # Point is outside desired range, target = 0  
                error = abs(approx_val - 0.0)
                target_errors_excluding.append(error)
        
        # Overall point errors (comparing to true function values)
        point_errors = [abs(true - approx) for true, approx in zip(true_values, approx_values)]
        max_point_error = max(point_errors)
        rms_point_error = math.sqrt(sum(e**2 for e in point_errors) / len(point_errors))
        
        # Calculate classification accuracy 
        threshold = 0.5
        correct_classifications = 0
        total_points = len(eval_points)
        
        for eval_point, approx_val in zip(eval_points, approx_values):
            # True classification based on whether point is in desired range
            if desired_range_low <= eval_point <= desired_range_high:
                true_class = 1  # Should be 1
            else:
                true_class = 0  # Should be 0
                
            # Approximate classification based on threshold
            approx_class = 1 if approx_val > threshold else 0
            
            if true_class == approx_class:
                correct_classifications += 1
        
        classification_accuracy = correct_classifications / total_points if total_points > 0 else 0
        
        # Compile results
        result = {
            'config': config,
            'epsilon': epsilon,
            'rescaled_epsilon': rescaled_epsilon,
            'use_rescaled': use_rescaled,
            'num_test_points': len(eval_points),
            'max_point_error': max_point_error,
            'rms_point_error': rms_point_error,
            'max_excluding_target_error': max(target_errors_excluding) if target_errors_excluding else 0,
            'max_desired_target_error': max(target_errors_desired) if target_errors_desired else 0,
            'rms_excluding_target_error': math.sqrt(sum(e**2 for e in target_errors_excluding) / len(target_errors_excluding)) if target_errors_excluding else 0,
            'rms_desired_target_error': math.sqrt(sum(e**2 for e in target_errors_desired) / len(target_errors_desired)) if target_errors_desired else 0,
            'classification_accuracy': classification_accuracy,
            'approximation_max_error': results['max_error'],
            'approximation_rms_error': results['rms_error'],
            'eval_points': eval_points,
            'true_values': true_values,
            'approx_values': approx_values,
            'point_errors': point_errors,
            'target_errors_excluding': target_errors_excluding,
            'target_errors_desired': target_errors_desired,
            'num_excluding_points': len(target_errors_excluding),
            'num_desired_points': len(target_errors_desired)
        }
        
        return result
    
    def run_epsilon_variation_test(self, function_name: str, 
                                 epsilon_values: List[float] = None,
                                 cheb_degree: int = 15,
                                 test_both_domains: bool = True) -> List[Dict[str, Any]]:
        """
        Run epsilon variation test for a single function.
        
        Args:
            function_name: Name of indicator function to test
            epsilon_values: List of epsilon values to test (if None, generates default)
            cheb_degree: Degree of Chebyshev approximation
            test_both_domains: Whether to test both original and rescaled domains
            
        Returns:
            List of result dictionaries
        """
        if epsilon_values is None:
            epsilon_values = self.generate_epsilon_values()
        
        results = []
        if test_both_domains:
            domains_to_test = [False, True]  # Test both original and rescaled
        else:
            # Use the domain specified in the base config
            domains_to_test = [self.base_config.use_rescaled]
        
        print(f"Testing {function_name} with {len(epsilon_values)} epsilon values...")
        print(f"Epsilon range: {min(epsilon_values):.3f} to {max(epsilon_values):.3f}")
        
        for use_rescaled in domains_to_test:
            domain_str = "rescaled" if use_rescaled else "original"
            print(f"\n--- Testing {domain_str} domain ---")
            
            # Generate unique run ID for this function/domain combination
            from indicator_functions import FunctionConfig
            temp_config = FunctionConfig()
            run_id = temp_config.generate_unique_id()
            
            domain_results = []
            
            for i, epsilon in enumerate(epsilon_values):
                print(f"Epsilon {i+1}/{len(epsilon_values)}: {epsilon:.3f}", end=" ")
                
                try:
                    result = self.evaluate_approximation_quality(
                        function_name=function_name,
                        epsilon=epsilon,
                        cheb_degree=cheb_degree,
                        use_rescaled=use_rescaled
                    )
                    # Use the same run_id for all epsilons in this domain test
                    result['config'].id = run_id
                    result['run_id'] = run_id
                    domain_results.append(result)
                    print(f"✓ Max error: {result['max_point_error']:.2e}")
                    
                except Exception as e:
                    print(f"✗ Error: {str(e)}")
                    continue
            
            if domain_results:
                # Save function plot, CSV, and epsilon plots for this run
                self.save_run_outputs(domain_results, function_name, use_rescaled, epsilon_values)
                results.extend(domain_results)
        
        self.results.extend(results)
        return results
    
    def save_run_outputs(self, results: List[Dict[str, Any]], function_name: str, 
                        use_rescaled: bool, epsilon_values: List[float]):
        """
        Save all outputs for a single run (function + domain combination).
        
        Args:
            results: Results for this run
            function_name: Name of function being tested
            use_rescaled: Whether rescaled domain was used
            epsilon_values: List of epsilon values tested
        """
        if not results:
            return
        
        run_id = results[0]['run_id']
        function_folder = f'{GRAPHS_BASE_PATH}/{function_name}'
        os.makedirs(function_folder, exist_ok=True)
        
        # 1. Save function plot
        self.save_function_plot(results[0], function_folder, run_id, use_rescaled)
        
        # 2. Save CSV log for this run
        self.save_run_csv(results, function_folder, run_id, use_rescaled)
        
        # 3. Save three epsilon error plots
        self.save_epsilon_error_plots(results, function_folder, run_id, use_rescaled, epsilon_values)
        
        # 4. Save summary log with averages
        self.save_summary_log(results, function_folder, run_id, use_rescaled)
        
        domain_str = "rescaled" if use_rescaled else "original"
        print(f"  Saved outputs for {function_name} ({domain_str}) with ID {run_id}")
    
    def save_function_plot(self, sample_result: Dict[str, Any], folder: str, 
                          run_id: str, use_rescaled: bool):
        """Save plot of the indicator function."""
        import matplotlib.pyplot as plt
        
        config = sample_result['config']
        function_name = config.function_name
        
        # Set domain based on rescaling
        if use_rescaled:
            x_vals = np.linspace(-1, 1, 200)
            domain_str = "[-1,1]"
        else:
            x_vals = np.linspace(0, 8, 200)
            domain_str = "[0,8]"
        
        # Get function and evaluate
        func = get_function_by_name(function_name, config)
        y_vals = [func(x) for x in x_vals]
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(x_vals, y_vals, 'b-', linewidth=2, label=function_name)
        ax.set_title(f'{function_name} - Rescaled: {use_rescaled} - Domain: {domain_str}')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at desired value location
        if use_rescaled:
            from generate_approx_integers import rescale_to_unit_interval
            desired_rescaled = rescale_to_unit_interval(
                np.array([config.desired_value]), max_value=config.max_val
            )[0]
            ax.axvline(x=desired_rescaled, color='r', linestyle='--', alpha=0.7, label='Desired Value')
        else:
            ax.axvline(x=config.desired_value, color='r', linestyle='--', alpha=0.7, label='Desired Value')
        ax.legend()
        
        plt.tight_layout()
        
        # Save with ID prefix
        rescaled_suffix = "rescaled" if use_rescaled else "original"
        filename = f'{run_id}_{function_name}_{rescaled_suffix}_function.png'
        filepath = f'{folder}/{filename}'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_run_csv(self, results: List[Dict[str, Any]], folder: str, 
                    run_id: str, use_rescaled: bool):
        """Save CSV log for this specific run."""
        if not results:
            return
        
        rescaled_suffix = "rescaled" if use_rescaled else "original"
        csv_filename = f'{run_id}_epsilon_test_{rescaled_suffix}.csv'
        csv_path = f'{folder}/{csv_filename}'
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'run_id', 'function_name', 'cheb_degree', 'use_rescaled',
                'epsilon', 'rescaled_epsilon', 'num_test_points', 
                'max_point_error', 'rms_point_error',
                'max_excluding_target_error', 'max_desired_target_error', 
                'rms_excluding_target_error', 'rms_desired_target_error',
                'classification_accuracy', 'approximation_max_error', 'approximation_rms_error',
                'desired_value', 'impulse_mu', 'impulse_sigma', 'impulse_scaling',
                'sp_amplitude', 'sp_base_amp', 'sp_base_freq', 'sp_freq', 'sp_steepness', 'sp_width'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                config = result['config']
                function_name = config.function_name
                
                row = {
                    'run_id': run_id,
                    'function_name': function_name,
                    'cheb_degree': config.cheb_degree,
                    'use_rescaled': use_rescaled,
                    'epsilon': result['epsilon'],
                    'rescaled_epsilon': result['rescaled_epsilon'] if result['rescaled_epsilon'] is not None else 'n/a',
                    'num_test_points': result['num_test_points'],
                    'max_point_error': result['max_point_error'],
                    'rms_point_error': result['rms_point_error'],
                    'max_excluding_target_error': result['max_excluding_target_error'],
                    'max_desired_target_error': result['max_desired_target_error'],
                    'rms_excluding_target_error': result['rms_excluding_target_error'],
                    'rms_desired_target_error': result['rms_desired_target_error'],
                    'classification_accuracy': result['classification_accuracy'] if result['classification_accuracy'] is not None else 'n/a',
                    'approximation_max_error': result['approximation_max_error'],
                    'approximation_rms_error': result['approximation_rms_error'],
                    'desired_value': config.desired_value,
                    'impulse_mu': config.impulse_mu if function_name == 'impulse' else 'n/a',
                    'impulse_sigma': config.impulse_sigma if function_name == 'impulse' else 'n/a',
                    'impulse_scaling': config.impulse_scaling if function_name == 'impulse' else 'n/a',
                    'sp_amplitude': config.sp_amplitude if function_name.startswith('plateau') else 'n/a',
                    'sp_base_amp': config.sp_base_amp if function_name.startswith('plateau') else 'n/a',
                    'sp_base_freq': config.sp_base_freq if function_name.startswith('plateau') else 'n/a',
                    'sp_freq': config.sp_freq if function_name.startswith('plateau') else 'n/a',
                    'sp_steepness': config.sp_steepness if function_name.startswith('plateau') else 'n/a',
                    'sp_width': config.sp_width if function_name.startswith('plateau') else 'n/a',
                }
                
                writer.writerow(row)
    
    def save_epsilon_error_plots(self, results: List[Dict[str, Any]], folder: str,
                                run_id: str, use_rescaled: bool, epsilon_values: List[float]):
        """Save three epsilon error plots."""
        import matplotlib.pyplot as plt
        
        if not results:
            return
        
        # Extract data for plotting
        epsilons_to_plot = []
        desired_errors = []
        excluding_errors = []
        
        for result in results:
            if use_rescaled and result['rescaled_epsilon'] is not None:
                epsilons_to_plot.append(result['rescaled_epsilon'])
            else:
                epsilons_to_plot.append(result['epsilon'])
            
            # Use target errors: how far Chebyshev output is from 0/1 targets
            desired_errors.append(result['rms_desired_target_error'])
            excluding_errors.append(result['rms_excluding_target_error'])
        
        if not epsilons_to_plot:
            return
        
        rescaled_suffix = "rescaled" if use_rescaled else "original"
        epsilon_label = "Rescaled Epsilon" if use_rescaled else "Epsilon"
        
        # 1. Green plot - Desired values error (how far from 1)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epsilons_to_plot, desired_errors, 'go-', linewidth=2, markersize=6, label='Desired Values Error')
        ax.set_xlabel(epsilon_label)
        ax.set_ylabel('RMS Error from Target (1.0)')
        ax.set_title(f'Desired Values Error vs {epsilon_label} - {results[0]["config"].function_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        filename = f'{run_id}_epsilon_desired_errors_{rescaled_suffix}.png'
        plt.savefig(f'{folder}/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Red plot - Excluding values error (how far from 0)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epsilons_to_plot, excluding_errors, 'ro-', linewidth=2, markersize=6, label='Excluding Values Error')
        ax.set_xlabel(epsilon_label)
        ax.set_ylabel('RMS Error from Target (0.0)')
        ax.set_title(f'Excluding Values Error vs {epsilon_label} - {results[0]["config"].function_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        filename = f'{run_id}_epsilon_excluding_errors_{rescaled_suffix}.png'
        plt.savefig(f'{folder}/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Combined plot - Both errors
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(epsilons_to_plot, desired_errors, 'go-', linewidth=2, markersize=6, label='Desired Values Error (from 1.0)')
        ax.plot(epsilons_to_plot, excluding_errors, 'ro-', linewidth=2, markersize=6, label='Excluding Values Error (from 0.0)')
        ax.set_xlabel(epsilon_label)
        ax.set_ylabel('RMS Error from Target')
        ax.set_title(f'Approximation Errors vs {epsilon_label} - {results[0]["config"].function_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        filename = f'{run_id}_epsilon_combined_errors_{rescaled_suffix}.png'
        plt.savefig(f'{folder}/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_summary_log(self, results: List[Dict[str, Any]], folder: str,
                        run_id: str, use_rescaled: bool):
        """Save summary log with average errors and run information."""
        if not results:
            return
        
        rescaled_suffix = "rescaled" if use_rescaled else "original"
        summary_filename = f'{run_id}_summary_{rescaled_suffix}.csv'
        summary_path = f'{folder}/{summary_filename}'
        
        # Calculate averages
        avg_excluding_error = sum(r['rms_excluding_target_error'] for r in results) / len(results)
        avg_desired_error = sum(r['rms_desired_target_error'] for r in results) / len(results)
        avg_point_error = sum(r['rms_point_error'] for r in results) / len(results)
        
        min_excluding_error = min(r['rms_excluding_target_error'] for r in results)
        max_excluding_error = max(r['rms_excluding_target_error'] for r in results)
        min_desired_error = min(r['rms_desired_target_error'] for r in results)
        max_desired_error = max(r['rms_desired_target_error'] for r in results)
        
        # Get epsilon range
        epsilon_values = [r['epsilon'] for r in results]
        rescaled_epsilon_values = [r['rescaled_epsilon'] for r in results if r['rescaled_epsilon'] is not None]
        
        config = results[0]['config']
        
        with open(summary_path, 'w', newline='') as csvfile:
            fieldnames = [
                'run_id', 'function_name', 'cheb_degree', 'use_rescaled',
                'num_epsilon_values', 'min_epsilon', 'max_epsilon',
                'epsilon_range', 'rescaled_epsilon_range',
                'avg_excluding_target_error', 'avg_desired_target_error', 'avg_point_error',
                'min_excluding_target_error', 'max_excluding_target_error',
                'min_desired_target_error', 'max_desired_target_error',
                'num_test_points_per_epsilon', 'total_evaluations',
                'desired_value', 'impulse_mu', 'impulse_sigma', 'impulse_scaling',
                'sp_amplitude', 'sp_base_amp', 'sp_base_freq', 'sp_freq', 'sp_steepness', 'sp_width'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            row = {
                'run_id': run_id,
                'function_name': config.function_name,
                'cheb_degree': config.cheb_degree,
                'use_rescaled': use_rescaled,
                'num_epsilon_values': len(results),
                'min_epsilon': min(epsilon_values),
                'max_epsilon': max(epsilon_values),
                'epsilon_range': str(epsilon_values),
                'rescaled_epsilon_range': str(rescaled_epsilon_values) if rescaled_epsilon_values else 'n/a',
                'avg_excluding_target_error': avg_excluding_error,
                'avg_desired_target_error': avg_desired_error,
                'avg_point_error': avg_point_error,
                'min_excluding_target_error': min_excluding_error,
                'max_excluding_target_error': max_excluding_error,
                'min_desired_target_error': min_desired_error,
                'max_desired_target_error': max_desired_error,
                'num_test_points_per_epsilon': results[0]['num_test_points'],
                'total_evaluations': len(results) * results[0]['num_test_points'],
                'desired_value': config.desired_value,
                'impulse_mu': config.impulse_mu if config.function_name == 'impulse' else 'n/a',
                'impulse_sigma': config.impulse_sigma if config.function_name == 'impulse' else 'n/a',
                'impulse_scaling': config.impulse_scaling if config.function_name == 'impulse' else 'n/a',
                'sp_amplitude': config.sp_amplitude if config.function_name.startswith('plateau') else 'n/a',
                'sp_base_amp': config.sp_base_amp if config.function_name.startswith('plateau') else 'n/a',
                'sp_base_freq': config.sp_base_freq if config.function_name.startswith('plateau') else 'n/a',
                'sp_freq': config.sp_freq if config.function_name.startswith('plateau') else 'n/a',
                'sp_steepness': config.sp_steepness if config.function_name.startswith('plateau') else 'n/a',
                'sp_width': config.sp_width if config.function_name.startswith('plateau') else 'n/a',
            }
            
            writer.writerow(row)
    
    def log_results_to_csv(self, results: List[Dict[str, Any]], 
                          csv_path: str = f"{GRAPHS_BASE_PATH}/epsilon_variation_results.csv"):
        """
        Log epsilon variation results to CSV.
        
        Args:
            results: List of result dictionaries
            csv_path: Path to CSV file
        """
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        if not results:
            print("No results to log.")
            return
        
        # Collect all epsilon values for this run
        all_epsilons = [r['epsilon'] for r in results]
        rescaled_epsilons = [r['rescaled_epsilon'] for r in results if r['rescaled_epsilon'] is not None]
        
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as csvfile:
            # Define all possible fields
            fieldnames = [
                'run_id', 'function_name', 'cheb_degree', 'use_rescaled',
                'epsilon_values', 'rescaled_epsilon_values', 'num_epsilon_values',
                'min_epsilon', 'max_epsilon', 'epsilon', 'rescaled_epsilon', 
                'num_test_points', 'max_point_error', 'rms_point_error',
                'max_excluding_error', 'max_desired_error', 'rms_excluding_error', 'rms_desired_error',
                'classification_accuracy', 'approximation_max_error', 'approximation_rms_error',
                'desired_value', 'impulse_mu', 'impulse_sigma', 'impulse_scaling',
                'sp_amplitude', 'sp_base_amp', 'sp_base_freq', 'sp_freq', 'sp_steepness', 'sp_width'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            # Group results by run (same function/domain combination)
            runs = {}
            for result in results:
                key = (result['config'].function_name, result['use_rescaled'])
                if key not in runs:
                    runs[key] = []
                runs[key].append(result)
            
            # Write one row per epsilon value, but include summary info
            for (function_name, use_rescaled), run_results in runs.items():
                run_id = run_results[0]['config'].id
                
                for result in run_results:
                    config = result['config']
                    
                    row = {
                        'run_id': run_id,
                        'function_name': function_name,
                        'cheb_degree': config.cheb_degree,
                        'use_rescaled': use_rescaled,
                        'epsilon_values': str(all_epsilons),
                        'rescaled_epsilon_values': str(rescaled_epsilons) if rescaled_epsilons else 'n/a',
                        'num_epsilon_values': len(all_epsilons),
                        'min_epsilon': min(all_epsilons),
                        'max_epsilon': max(all_epsilons),
                        'epsilon': result['epsilon'],
                        'rescaled_epsilon': result['rescaled_epsilon'] if result['rescaled_epsilon'] is not None else 'n/a',
                        'num_test_points': result['num_test_points'],
                        'max_point_error': result['max_point_error'],
                        'rms_point_error': result['rms_point_error'],
                        'max_excluding_error': result['max_excluding_error'],
                        'max_desired_error': result['max_desired_error'],
                        'rms_excluding_error': result['rms_excluding_error'],
                        'rms_desired_error': result['rms_desired_error'],
                        'classification_accuracy': result['classification_accuracy'] if result['classification_accuracy'] is not None else 'n/a',
                        'approximation_max_error': result['approximation_max_error'],
                        'approximation_rms_error': result['approximation_rms_error'],
                        'desired_value': config.desired_value,
                        'impulse_mu': config.impulse_mu if function_name == 'impulse' else 'n/a',
                        'impulse_sigma': config.impulse_sigma if function_name == 'impulse' else 'n/a',
                        'impulse_scaling': config.impulse_scaling if function_name == 'impulse' else 'n/a',
                        'sp_amplitude': config.sp_amplitude if function_name.startswith('plateau') else 'n/a',
                        'sp_base_amp': config.sp_base_amp if function_name.startswith('plateau') else 'n/a',
                        'sp_base_freq': config.sp_base_freq if function_name.startswith('plateau') else 'n/a',
                        'sp_freq': config.sp_freq if function_name.startswith('plateau') else 'n/a',
                        'sp_steepness': config.sp_steepness if function_name.startswith('plateau') else 'n/a',
                        'sp_width': config.sp_width if function_name.startswith('plateau') else 'n/a',
                    }
                    
                    writer.writerow(row)
        
        print(f"Results logged to {csv_path}")


def run_single_function_epsilon_test(function_name: str = 'impulse', 
                                    base_config: 'FunctionConfig' = None,
                                    epsilon_values: list = None,
                                    cheb_degree: int = 119,
                                    test_both_domains: bool = True):
    """
    Run epsilon variation test for a single function.
    
    Args:
        function_name: Function to test ('impulse', 'plateau_reg', 'plateau_sine')
        base_config: Configuration object with all parameters
        epsilon_values: List of epsilon values to test
        cheb_degree: Degree of Chebyshev polynomial
        test_both_domains: Whether to test both original and rescaled domains
    """
    print(f"=== Epsilon Variation Test for {function_name} ===")
    
    # Use provided config or create default
    if base_config is None:
        base_config = FunctionConfig(
            function_name=function_name,
            desired_value=2.0,
            impulse_mu=2.0,
            impulse_sigma=0.3,
            impulse_scaling=1.0
        )
    else:
        # Update function name in the config to match the requested function
        base_config.function_name = function_name
    
    # Create tester
    tester = EpsilonVariationTester(base_config)
    
    # Use provided epsilon values or generate default
    if epsilon_values is None:
        epsilon_values = tester.generate_epsilon_values(min_eps=0.001, max_eps=0.4, num_eps=20)
    
    print(f"Testing with {len(epsilon_values)} epsilon values: {epsilon_values[:3]}...{epsilon_values[-3:]}")
    
    # Run test
    results = tester.run_epsilon_variation_test(
        function_name=function_name,
        epsilon_values=epsilon_values,
        cheb_degree=cheb_degree,
        test_both_domains=test_both_domains
    )
    
    print(f"\nTest Complete: {len(results)} evaluations")
    print(f"Files saved to: {GRAPHS_BASE_PATH}/{function_name}/")
    
    return results


def run_comprehensive_epsilon_test():
    """
    Run comprehensive epsilon variation test for all indicator functions.
    """
    print("=== Comprehensive Epsilon Variation Test ===")
    
    functions_to_test = ['impulse', 'plateau_reg', 'plateau_sine']
    all_results = []
    
    for function_name in functions_to_test:
        print(f"\n{'='*50}")
        results = run_single_function_epsilon_test(function_name, test_both_domains=True, num_eps=20)
        all_results.extend(results)
        print(f"Completed {function_name}: {len(results)} test cases")
    
    print(f"\n{'='*50}")
    print(f"All Tests Complete: {len(all_results)} total evaluations")
    print(f"{'='*50}")
    
    return all_results


if __name__ == "__main__":
    # Test just impulse function first with new signature
    import numpy as np
    from indicator_functions import FunctionConfig
    
    # Create a simple config for testing
    test_config = FunctionConfig(
        function_name='impulse',
        desired_value=2.0,
        impulse_mu=2.0,
        impulse_sigma=0.3,
        impulse_scaling=1.0
    )
    
    # Generate 5 epsilon values for testing
    test_epsilons = np.linspace(0.001, 0.4, 5).tolist()
    
    results = run_single_function_epsilon_test(
        function_name='impulse',
        base_config=test_config,
        epsilon_values=test_epsilons,
        cheb_degree=119,
        test_both_domains=True
    )
    
    # Uncomment to run all functions
    # results = run_comprehensive_epsilon_test()