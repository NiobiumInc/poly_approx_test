"""
Chebyshev Polynomial Approximation Implementation

This module implements Chebyshev polynomial approximation for functions over arbitrary intervals.
The implementation uses the discrete cosine transform (DCT) approach for coefficient computation
and Clenshaw's algorithm for efficient evaluation.
"""

import math
import numpy as np
from typing import Callable, List, Union, Tuple, Dict, Any
from indicator_functions import FunctionConfig, get_function_by_name
from config import GRAPHS_BASE_PATH


def eval_chebyshev_coefficients(
    func: Callable[[float], float], a: float, b: float, degree: int
) -> List[float]:
    """
    Compute Chebyshev polynomial coefficients for approximating a given function `func`
    over the interval [a, b], using the first `degree + 1` Chebyshev polynomials.

    This implementation works because:
    1. It evaluates the function at Chebyshev nodes (roots of Chebyshev polynomials)
    2. Uses the discrete orthogonality relation of Chebyshev polynomials
    3. The Chebyshev nodes provide optimal interpolation points that minimize 
       the maximum approximation error (Chebyshev equioscillation theorem)

    Args:
        func: The function to approximate.
        a (float): Lower bound of the interval.
        b (float): Upper bound of the interval.
        degree (int): Degree of the Chebyshev approximation.

    Returns:
        List[float]: List of Chebyshev coefficients (length degree + 1).
    """
    if degree <= 0:
        raise ValueError("The degree of approximation must be positive")

    coeff_total = degree + 1  # Number of coefficients to compute

    # Constants for mapping from Chebyshev nodes (in [-1, 1]) to [a, b]
    b_minus_a = 0.5 * (b - a)
    b_plus_a = 0.5 * (b + a)
    pi_by_deg = math.pi / coeff_total

    # Evaluate the function at the Chebyshev nodes mapped to [a, b]
    # Chebyshev nodes: x_k = cos(π(k + 0.5)/n) for k = 0, 1, ..., n-1
    function_points = [
        func(math.cos(pi_by_deg * (i + 0.5)) * b_minus_a + b_plus_a)
        for i in range(coeff_total)
    ]

    mult_factor = 2.0 / coeff_total  # Normalization factor
    coefficients = []

    # Compute each Chebyshev coefficient using the discrete orthogonality relation
    # c_j = (2/n) * Σ[k=0 to n-1] f(x_k) * cos(jπ(k + 0.5)/n)
    for i in range(coeff_total):
        coeff = sum(
            function_points[j] * math.cos(pi_by_deg * i * (j + 0.5))
            for j in range(coeff_total)
        )
        coefficients.append(coeff * mult_factor)

    return coefficients


def eval_chebyshev_function(
    func: Callable[[float], float],
    eval_points: List[float],
    a: float,
    b: float,
    deg: int,
) -> List[float]:
    """
    Evaluate a function at arbitrary points by approximating it with a Chebyshev polynomial.

    This implementation works because:
    1. It computes Chebyshev coefficients using the optimal interpolation nodes
    2. Uses Clenshaw's recurrence algorithm for stable polynomial evaluation
    3. Maps evaluation points from [a,b] to [-1,1] where Chebyshev polynomials are defined
    4. The recurrence T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x) is numerically stable

    Args:
        func: The target function to approximate.
        eval_points (List[float]): The input points at which to evaluate the approximation.
        a (float): Lower bound of the approximation interval.
        b (float): Upper bound of the approximation interval.
        deg (int): Degree of the Chebyshev approximation.

    Returns:
        List[float]: Approximated values of the function at the given input points.
    """
    if not (1 < deg <= 2031):
        raise ValueError("Degree must be in range (1, 2031] for numerical stability")

    # Get Chebyshev coefficients of the approximation polynomial
    coeffs = eval_chebyshev_coefficients(func, a, b, deg)
    coeffs[0] /= 2.0  # First coefficient is halved due to Chebyshev series convention

    # Rescale eval points from [a, b] to [-1, 1] using affine transformation
    # x_normalized = 2*(x - a)/(b - a) - 1 = x * 2/(b-a) - (a+b)/(b-a)
    scale_factor = 2.0 / (b - a)
    offset = -(a + b) / (b - a)

    result = []
    for point in eval_points:
        x = point * scale_factor + offset  # Rescaled point in [-1, 1]
        x2 = 2 * x  # Used for recurrence

        # Initialize Clenshaw recurrence: T_0(x) = 1, T_1(x) = x
        if len(coeffs) == 1:
            result.append(coeffs[0])
            continue
        elif len(coeffs) == 2:
            result.append(coeffs[0] + coeffs[1] * x)
            continue

        # Start with the last two terms and work backwards
        # This is Clenshaw's algorithm for evaluating Chebyshev series
        b_next = 0.0
        b_curr = coeffs[-1]
        
        # Work backwards through coefficients
        for j in range(len(coeffs) - 2, 0, -1):
            b_prev = b_next
            b_next = b_curr
            b_curr = coeffs[j] + x2 * b_next - b_prev

        # Final step: add the constant term
        y = coeffs[0] + x * b_curr - b_next
        result.append(y)

    return result


class ChebyshevApproximator:
    """
    Main class for performing Chebyshev approximation with parameter tracking.
    """
    
    def __init__(self, config: FunctionConfig):
        self.config = config
        self.function = get_function_by_name(config.function_name, config)
        self.coefficients = None
        self.approximation_interval = None
        
    def compute_approximation(self, a: float, b: float) -> List[float]:
        """
        Compute Chebyshev coefficients for the configured function.
        
        Args:
            a: Lower bound of approximation interval
            b: Upper bound of approximation interval
            
        Returns:
            List of Chebyshev coefficients
        """
        self.approximation_interval = (a, b)
        self.coefficients = eval_chebyshev_coefficients(
            self.function, a, b, self.config.cheb_degree
        )
        return self.coefficients
    
    def evaluate_approximation(self, eval_points: List[float]) -> List[float]:
        """
        Evaluate the Chebyshev approximation at given points.
        
        Args:
            eval_points: Points to evaluate the approximation at
            
        Returns:
            Approximated function values
        """
        if self.coefficients is None or self.approximation_interval is None:
            raise ValueError("Must compute approximation first using compute_approximation()")
        
        a, b = self.approximation_interval
        return eval_chebyshev_function(
            self.function, eval_points, a, b, self.config.cheb_degree
        )
    
    def compute_approximation_error(self, eval_points: List[float]) -> Tuple[List[float], float, float]:
        """
        Compute approximation errors at evaluation points.
        
        Args:
            eval_points: Points to evaluate errors at
            
        Returns:
            Tuple of (error_list, max_error, rms_error)
        """
        if self.coefficients is None:
            raise ValueError("Must compute approximation first")
        
        true_values = [self.function(x) for x in eval_points]
        approx_values = self.evaluate_approximation(eval_points)
        
        errors = [abs(true - approx) for true, approx in zip(true_values, approx_values)]
        max_error = max(errors)
        rms_error = math.sqrt(sum(e**2 for e in errors) / len(errors))
        
        return errors, max_error, rms_error
    
    def run_approximation_analysis(self, a: float = None, b: float = None, 
                                 num_eval_points: int = 100) -> Dict[str, Any]:
        """
        Run complete approximation analysis and return results.
        Uses rescaled domain [-1,1] if use_rescaled=True, otherwise [0,8].
        
        Args:
            a: Lower bound of approximation interval (auto-set if None)
            b: Upper bound of approximation interval (auto-set if None)
            num_eval_points: Number of points to evaluate for error analysis
            
        Returns:
            Dictionary containing all results and configuration
        """
        # Set domain based on rescaling preference
        if a is None or b is None:
            if self.config.use_rescaled:
                a, b = -1.0, 1.0
            else:
                a, b = 0.0, 8.0
        
        # Compute approximation
        coeffs = self.compute_approximation(a, b)
        
        # Generate evaluation points
        eval_points = np.linspace(a, b, num_eval_points).tolist()
        
        # Compute errors
        errors, max_error, rms_error = self.compute_approximation_error(eval_points)
        
        # Compile results
        results = {
            'config': self.config.to_dict(),
            'coefficients': coeffs,
            'approximation_interval': (a, b),
            'num_eval_points': num_eval_points,
            'max_error': max_error,
            'rms_error': rms_error,
            'eval_points': eval_points,
            'errors': errors
        }
        
        return results
    
    def save_approximation_plot(self, results: Dict[str, Any], base_dir: str = GRAPHS_BASE_PATH):
        """
        Save a plot of the approximation results in function-specific folder.
        
        Args:
            results: Results dictionary from run_approximation_analysis
            base_dir: Base directory to save plots in
        """
        import matplotlib.pyplot as plt
        import os
        
        # Create function-specific folder
        function_folder = os.path.join(base_dir, self.config.function_name)
        os.makedirs(function_folder, exist_ok=True)
        
        eval_points = results['eval_points']
        true_values = [self.function(x) for x in eval_points]
        approx_values = self.evaluate_approximation(eval_points)
        errors = results['errors']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot function and approximation
        ax1.plot(eval_points, true_values, 'b-', linewidth=2, label='True Function')
        ax1.plot(eval_points, approx_values, 'r--', linewidth=2, label='Chebyshev Approximation')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title(f'{self.config.function_name} - Degree {self.config.cheb_degree} - Rescaled: {self.config.use_rescaled}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot errors
        ax2.semilogy(eval_points, errors, 'g-', linewidth=2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title(f'Approximation Error (Max: {results["max_error"]:.2e}, RMS: {results["rms_error"]:.2e})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Generate filename with ID at end
        if not self.config.id:
            self.config.id = self.config.generate_unique_id()
        
        rescaled_suffix = "rescaled" if self.config.use_rescaled else "original"
        filename = f'chebyshev_{self.config.function_name}_{rescaled_suffix}_{self.config.id}.png'
        filepath = os.path.join(function_folder, filename)
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved approximation plot: {filepath}")
        return filepath
    
    def log_results_to_csv(self, results: Dict[str, Any], csv_path: str = f"{GRAPHS_BASE_PATH}/chebyshev_results.csv"):
        """
        Log approximation results to CSV file.
        
        Args:
            results: Results dictionary from run_approximation_analysis
            csv_path: Path to CSV file
        """
        import csv
        import os
        
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Combine config and results
        log_dict = self.config.to_csv_dict()
        log_dict.update({
            'max_error': results['max_error'],
            'rms_error': results['rms_error'],
            'approximation_interval_a': results['approximation_interval'][0],
            'approximation_interval_b': results['approximation_interval'][1],
            'num_eval_points': results['num_eval_points'],
            'num_coefficients': len(results['coefficients'])
        })
        
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = list(log_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(log_dict)
        
        print(f"Results logged to {csv_path}")


def run_single_chebyshev_approximation(function_name: str, use_rescaled: bool = False, degree: int = 15):
    """
    Run Chebyshev approximation for a single function.
    
    Args:
        function_name: Name of function to approximate
        use_rescaled: Whether to use rescaled domain [-1,1] or original [0,8]  
        degree: Degree of Chebyshev approximation
    """
    # Create configuration
    config = FunctionConfig(
        id="",  # Will be generated
        function_name=function_name,
        cheb_degree=degree,
        desired_value=2.0,
        impulse_mu=2.0,
        impulse_sigma=0.3,
        use_rescaled=use_rescaled
    )
    
    # Generate unique ID
    config.id = config.generate_unique_id()
    
    print(f"=== Chebyshev Approximation: {function_name} (Rescaled: {use_rescaled}) ===")
    config.print_config()
    
    # Run approximation
    approximator = ChebyshevApproximator(config)
    results = approximator.run_approximation_analysis(num_eval_points=100)
    
    print(f"\nApproximation Results:")
    print(f"Domain: {results['approximation_interval']}")
    print(f"Max Error: {results['max_error']:.2e}")
    print(f"RMS Error: {results['rms_error']:.2e}")
    print(f"First 5 coefficients: {results['coefficients'][:5]}")
    
    # Save plot and log results
    filepath = approximator.save_approximation_plot(results)
    approximator.log_results_to_csv(results)
    
    print(f"Plot saved to: {filepath}")
    return config, results


def demonstrate_chebyshev_approximation():
    """
    Demonstrate Chebyshev approximation running functions individually.
    """
    print("=== Individual Chebyshev Approximation Demos ===")
    
    # Example: Run just impulse
    print("\n--- Impulse Approximation ---")
    config1, results1 = run_single_chebyshev_approximation('impulse', use_rescaled=False)
    config2, results2 = run_single_chebyshev_approximation('impulse', use_rescaled=True)
    
    print("\n--- Plateau Sine Approximation ---")
    config3, results3 = run_single_chebyshev_approximation('plateau_sine', use_rescaled=False, degree=20)
    
    print("-" * 50)


if __name__ == "__main__":
    demonstrate_chebyshev_approximation()