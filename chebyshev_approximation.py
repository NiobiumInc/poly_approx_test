"""
Chebyshev Polynomial Approximation Implementation

This module implements Chebyshev polynomial approximation for functions over arbitrary intervals.
The implementation uses the discrete cosine transform (DCT) approach for coefficient computation
and Clenshaw's algorithm for efficient evaluation.
"""

import math
import numpy as np
from typing import Callable, List, Union


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


def demonstrate_chebyshev_approximation():
    """
    Demonstrate Chebyshev approximation with example functions.
    """
    # Test function: f(x) = sin(x) on [0, 2π]
    def test_func(x):
        return math.sin(x)
    
    a, b = 0, 2 * math.pi
    degree = 10
    
    print("=== Chebyshev Approximation Demo ===")
    print(f"Approximating sin(x) on [{a:.2f}, {b:.2f}] with degree {degree}")
    
    # Get coefficients
    coeffs = eval_chebyshev_coefficients(test_func, a, b, degree)
    print(f"First 5 coefficients: {coeffs[:5]}")
    
    # Test evaluation at specific points
    test_points = [0, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]
    approx_values = eval_chebyshev_function(test_func, test_points, a, b, degree)
    
    print("\nComparison (Point, True Value, Approximation, Error):")
    for point, approx in zip(test_points, approx_values):
        true_val = test_func(point)
        error = abs(true_val - approx)
        print(f"{point:.3f}, {true_val:.6f}, {approx:.6f}, {error:.2e}")


if __name__ == "__main__":
    demonstrate_chebyshev_approximation()