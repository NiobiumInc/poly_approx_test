"""
Chebyshev Polynomial Approximation Module

This module provides functions for computing Chebyshev polynomial coefficients
and evaluating Chebyshev approximations of arbitrary functions.
Uses NumPy's optimized implementation with our original interface.
"""

import math
import numpy as np
import numpy.polynomial.chebyshev as np_cheb
from typing import Callable, List
import config
import utils


def eval_chebyshev_coefficients(
    func: Callable[[float], float], a: float, b: float, degree: int
) -> List[float]:
    """
    Compute Chebyshev polynomial coefficients for approximating a given function `func`
    over the interval [a, b], using the first `degree + 1` Chebyshev polynomials.

    Uses NumPy's optimized implementation with the same Chebyshev nodes as our
    original algorithm for mathematical consistency.

    Args:
        func: The function to approximate.
        a (float): Lower bound of the interval.
        b (float): Upper bound of the interval.
        degree (int): Degree of the Chebyshev approximation.

    Returns:
        List[float]: List of Chebyshev coefficients (length degree + 1).
    
    Raises:
        ValueError: If degree is 0 or negative.
    """
    if degree <= 0:
        raise ValueError("The degree of approximation must be positive")

    coeff_total = degree + 1
    
    # Generate the same Chebyshev nodes as our original implementation
    pi_by_deg = math.pi / coeff_total
    cheb_nodes_unit = [math.cos(pi_by_deg * (i + 0.5)) for i in range(coeff_total)]
    
    # Map nodes from [-1, 1] to [a, b]
    b_minus_a = 0.5 * (b - a)
    b_plus_a = 0.5 * (b + a)
    x_nodes = [node * b_minus_a + b_plus_a for node in cheb_nodes_unit]
    
    # Evaluate function at these nodes
    y_values = [func(x) for x in x_nodes]
    
    # Use NumPy to fit Chebyshev polynomial
    numpy_poly = np_cheb.Chebyshev.fit(x_nodes, y_values, degree, domain=[a, b])
    
    # Extract coefficients and adjust for our convention
    coeffs = numpy_poly.coef.tolist()
    
    # Adjust first coefficient to match our original normalization
    # (NumPy uses c_0, we used 2*c_0 internally)
    coeffs[0] *= 2.0
    
    return coeffs


def eval_chebyshev_function(
    func: Callable[[float], float],
    eval_points: List[float],
    a: float,
    b: float,
    deg: int,
) -> List[float]:
    """
    Evaluate a function at arbitrary points by approximating it with a Chebyshev polynomial.

    Uses NumPy's optimized Chebyshev evaluation with our original interface and
    mathematical behavior.

    Args:
        func: The target function to approximate.
        eval_points (List[float]): The input points at which to evaluate the approximation.
        a (float): Lower bound of the approximation interval.
        b (float): Upper bound of the approximation interval.
        deg (int): Degree of the Chebyshev approximation.

    Returns:
        List[float]: Approximated values of the function at the given input points.
    
    Raises:
        ValueError: If degree is not in safe evaluation range [2, 2031].
    """
    if not (1 < deg <= 2031):
        raise ValueError(f"Degree must be in range (1, 2031], got {deg}")

    # Get coefficients using our method
    coeffs = eval_chebyshev_coefficients(func, a, b, deg)
    
    # Adjust coefficients back to NumPy's convention for evaluation
    adjusted_coeffs = coeffs.copy()
    adjusted_coeffs[0] /= 2.0
    
    # Create NumPy Chebyshev polynomial object
    numpy_poly = np_cheb.Chebyshev(adjusted_coeffs, domain=[a, b])
    
    # Evaluate at all points using NumPy's optimized evaluation
    result = [float(numpy_poly(point)) for point in eval_points]
    
    return result


def create_chebyshev_approximation(func: Callable[[float], float], 
                                 a: float, b: float, degree: int) -> np_cheb.Chebyshev:
    """
    Create a NumPy Chebyshev polynomial object for a function.
    
    This provides access to all of NumPy's Chebyshev functionality while
    maintaining consistency with our original algorithm.
    
    Args:
        func: Function to approximate
        a: Lower bound of interval
        b: Upper bound of interval  
        degree: Polynomial degree
        
    Returns:
        NumPy Chebyshev polynomial object
    """
    # Get coefficients using our method
    coeffs = eval_chebyshev_coefficients(func, a, b, degree)
    
    # Adjust for NumPy convention
    adjusted_coeffs = coeffs.copy()
    adjusted_coeffs[0] /= 2.0
    
    # Create and return NumPy polynomial
    return np_cheb.Chebyshev(adjusted_coeffs, domain=[a, b])


def evaluate_approximation_error(
    true_values: List[float], 
    approx_values: List[float], 
    labels: List[float]
) -> tuple[float, float]:
    """
    Calculate average approximation error for desired and non-desired regions.
    
    Args:
        true_values: True function values (labels in this context)
        approx_values: Approximated function values
        labels: Binary labels indicating desired (1.0) vs non-desired (0.0) regions
        
    Returns:
        Tuple of (avg_error_desired, avg_error_other)
    """
    err_desired = 0.0
    err_other = 0.0
    count_desired = 0
    count_other = 0
    
    for pred, lbl in zip(approx_values, labels):
        err = abs(pred - lbl)
        if lbl == 1.0:
            err_desired += err
            count_desired += 1
        else:
            err_other += err
            count_other += 1
    
    avg_error_desired = err_desired / count_desired if count_desired else 0.0
    avg_error_other = err_other / count_other if count_other else 0.0
    
    return avg_error_desired, avg_error_other


def approximate_function_on_domain(func: Callable[[float], float], 
                                 domain_bounds: tuple = None,
                                 degree: int = None) -> dict:
    """
    Approximate a function using configuration settings.
    
    Args:
        func: Function to approximate
        domain_bounds: Optional (a, b) bounds, uses config domain if None
        degree: Optional degree, uses config.CHEB_DEGREE if None
        
    Returns:
        Dictionary with coefficients, domain info, and evaluation function
    """
    # Use configuration defaults
    if domain_bounds is None:
        if config.USE_RESCALED:
            domain_bounds = (-1.0, 1.0)
        else:
            domain_bounds = (config.MIN_VAL, config.MAX_VAL)
    
    if degree is None:
        degree = config.CHEB_DEGREE
    
    a, b = domain_bounds
    
    # Compute coefficients
    coeffs = eval_chebyshev_coefficients(func, a, b, degree)
    
    # Create evaluation function
    def evaluate_at_points(points: List[float]) -> List[float]:
        return eval_chebyshev_function(func, points, a, b, degree)
    
    return {
        "coefficients": coeffs,
        "domain": domain_bounds,
        "degree": degree,
        "evaluate": evaluate_at_points,
        "num_coefficients": len(coeffs)
    }


if __name__ == "__main__":
    print("=== NUMPY-BASED CHEBYSHEV APPROXIMATION TEST ===")
    
    # Test with configuration system
    print(f"Using configuration:")
    print(f"  Degree: {config.CHEB_DEGREE}")
    print(f"  Domain: {'[-1,1]' if config.USE_RESCALED else f'[{config.MIN_VAL},{config.MAX_VAL}]'}")
    print(f"  Backend: NumPy's optimized implementation")
    
    # Test function: f(x) = sin(x)
    def test_func(x):
        return math.sin(x)
    
    print(f"\nApproximating sin(x):")
    
    # Get approximation using config
    approximation = approximate_function_on_domain(test_func)
    
    print(f"  Domain: {approximation['domain']}")
    print(f"  Degree: {approximation['degree']}")
    print(f"  Number of coefficients: {approximation['num_coefficients']}")
    print(f"  First 5 coefficients: {[utils.format_number(c) for c in approximation['coefficients'][:5]]}")
    
    # Test evaluation at specific points
    if config.USE_RESCALED:
        test_points = [-1.0, -0.5, 0.0, 0.5, 1.0]
    else:
        test_points = [0, 2, 4, 6, 8]
    
    approx_values = approximation['evaluate'](test_points)
    
    print(f"\nEvaluation comparison:")
    print("Point     | True Value | Approximation | Error")
    print("-" * 45)
    for point, approx in zip(test_points, approx_values):
        true_val = test_func(point)
        error = abs(true_val - approx)
        print(f"{point:8.3f} | {true_val:10.6f} | {approx:13.6f} | {utils.format_number(error)}")
    
    # Test direct NumPy polynomial access
    print(f"\nTesting direct NumPy polynomial access:")
    domain = (-1, 1) if config.USE_RESCALED else (config.MIN_VAL, config.MAX_VAL)
    numpy_poly = create_chebyshev_approximation(test_func, domain[0], domain[1], 10)
    
    print(f"  NumPy polynomial domain: {numpy_poly.domain}")
    print(f"  NumPy polynomial degree: {numpy_poly.degree()}")
    print(f"  Can use all NumPy operations: derivative, integration, etc.")
    
    # Show derivative capability
    derivative_poly = numpy_poly.deriv()
    test_point = 0.5 if config.USE_RESCALED else 4.0
    approx_derivative = derivative_poly(test_point)
    true_derivative = math.cos(test_point)  # derivative of sin(x)
    
    print(f"  Derivative at x={test_point}: true={true_derivative:.6f}, approx={approx_derivative:.6f}")
    
    # Test with configuration-based error analysis
    print(f"\nTesting approximation error calculation:")
    labels = [1.0 if abs(p) < 0.1 else 0.0 for p in test_points]  # Arbitrary labels for test
    true_values = [test_func(p) for p in test_points]
    
    err_desired, err_other = evaluate_approximation_error(true_values, approx_values, labels)
    print(f"  Average error (desired region): {utils.format_number(err_desired)}")
    print(f"  Average error (other region): {utils.format_number(err_other)}")
    
    print(f"\n=== NUMPY CHEBYSHEV TEST COMPLETE ===")