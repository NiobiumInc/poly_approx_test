"""
Indicator Functions Module

This module implements various indicator functions for polynomial approximation analysis.
All functions are designed to distinguish between desired and non-desired approximate
integer values within specified epsilon ranges.
"""

import math
import numpy as np
from typing import Callable
import config
import utils


def rescale_to_unit_interval(values: np.ndarray, min_val: float = None, 
                           max_val: float = None) -> np.ndarray:
    """
    Rescale values from [0, 8] to [-1, 1] using x/4 - 1 transformation.
    
    Args:
        values: Array of values to rescale (should be in [0, 8])
        min_val: Ignored (kept for API compatibility)
        max_val: Ignored (kept for API compatibility)
        
    Returns:
        Array of values rescaled to [-1, 1] using x/4 - 1
    """
    # Use the specified transformation: x/4 - 1
    # This maps [0, 8] -> [-1, 1]
    return values / 4.0 - 1.0


def impulse(x: float, epsilon: float = 0) -> float:
    """
    Gaussian impulse function centered at the desired value.
    
    This function creates a Gaussian peak at the desired value location,
    with the peak height normalized and width controlled by sigma parameter.
    
    Args:
        x: Input value
        epsilon: Epsilon parameter (used for domain rescaling calculations)
        
    Returns:
        Gaussian impulse value at x
    """
    # Get impulse parameters from config
    impulse_params = config.get_function_params("impulse")
    sigma = impulse_params["sigma"]
    mu = impulse_params["mu"]
    scaling = impulse_params["scaling"]
    
    # Determine center location
    if mu == 0:
        if config.USE_RESCALED:
            # Rescale desired value to [-1, 1] domain
            mu = rescale_to_unit_interval(np.array([config.DESIRED_VALUE]))[0]
        else:
            mu = config.DESIRED_VALUE
    
    # Compute Gaussian
    x_shifted = x - mu
    sigma2 = 2 * sigma**2
    
    return scaling * np.exp(-(x_shifted**2) / sigma2)


def plateau_sine(x: float, epsilon: float = 0) -> float:
    """
    Smooth plateau function with sinusoidal modulation and sigmoid transitions.
    
    This function creates a plateau region around the desired value with smooth
    transitions using sigmoid functions, and includes both base wave modulation
    and internal ripples within the plateau.
    
    Args:
        x: Input value
        epsilon: Epsilon parameter (used for domain rescaling calculations)
        
    Returns:
        Plateau sine function value at x
    """
    # Get plateau sine parameters from config
    plateau_params = config.get_function_params("plateau_sine")
    sp_amplitude = plateau_params["amplitude"]
    sp_base_amp = plateau_params["base_amp"]
    sp_base_freq = plateau_params["base_freq"]
    sp_freq = plateau_params["freq"]
    sp_steepness = plateau_params["steepness"]
    sp_width = plateau_params["width"]
    
    # Determine center location and width
    if config.USE_RESCALED:
        # Rescale desired value to [-1, 1] domain using x/4 - 1
        mu = rescale_to_unit_interval(np.array([config.DESIRED_VALUE]))[0]
        # Adjust width for rescaled domain: [0,8] -> [-1,1] has scale factor of 1/4
        width = sp_width / 4.0
    else:
        mu = config.DESIRED_VALUE
        width = sp_width
    
    # Base wave (present everywhere)
    base_wave = sp_base_amp * np.sin(sp_base_freq * x * np.pi)
    
    # Sigmoid transitions for plateau
    rise = 1 / (1 + np.exp(-sp_steepness * (x - (mu - width / 2))))
    fall = 1 / (1 + np.exp(-sp_steepness * (mu + width / 2 - x)))
    
    # Plateau mask (1 inside plateau, 0 outside)
    plateau_mask = rise * fall
    
    # Ripples within the plateau
    ripple_in_plateau = sp_amplitude * np.sin(sp_freq * x * np.pi)
    
    # Combine: main plateau (1 + ripples) where mask is active, base wave elsewhere
    return (1 + ripple_in_plateau) * plateau_mask + base_wave * (1 - plateau_mask)


def plateau_reg(x: float, epsilon: float = 0) -> float:
    """
    Regular plateau function with hard cutoff transitions.
    
    This function creates a simple step function that returns 1 within
    a fixed interval around the desired value and 0 elsewhere.
    
    Args:
        x: Input value
        epsilon: Epsilon parameter (used for domain rescaling calculations)
        
    Returns:
        1.0 if x is within the plateau region, 0.0 otherwise
    """
    if config.USE_RESCALED:
        # Rescale the plateau boundaries to [-1, 1] domain
        low, high = rescale_to_unit_interval(
            np.array([config.DESIRED_VALUE - 0.5, config.DESIRED_VALUE + 0.5])
        )
        return 1.0 if low <= x <= high else 0.0
    else:
        # Use original domain with fixed ±0.5 interval
        return 1.0 if (config.DESIRED_VALUE - 0.5 <= x <= config.DESIRED_VALUE + 0.5) else 0.0


def plateau_sine_impulse(x: float, epsilon: float = 0) -> float:
    """
    Hybrid function: sinusoidal plateau in desired region, impulse function elsewhere.
    
    This function mathematically combines clean plateau behavior around the desired value
    with impulse (Gaussian) behavior in non-desired regions. Uses smooth transitions
    and frequency damping to minimize polynomial approximation artifacts.
    
    Mathematical approach:
    - Clean plateau region: no global oscillatory components
    - Impulse regions: pure Gaussian decay
    - Smooth frequency damping based on distance from plateau center
    - Eliminates global sinusoidal components that cause approximation issues
    
    Args:
        x: Input value
        epsilon: Epsilon parameter (used for domain rescaling calculations)
        
    Returns:
        Optimized plateau-sine-impulse function value at x
    """
    # Get parameters from both function types
    plateau_params = config.get_function_params("plateau_sine")
    impulse_params = config.get_function_params("impulse")
    
    # Plateau sine parameters
    sp_amplitude = plateau_params["amplitude"]
    sp_base_amp = plateau_params["base_amp"]  # Should be 0.0 now
    sp_base_freq = plateau_params["base_freq"]
    sp_freq = plateau_params["freq"]
    sp_steepness = plateau_params["steepness"]
    sp_width = plateau_params["width"]
    
    # Impulse parameters
    imp_sigma = impulse_params["sigma"]
    imp_mu = impulse_params["mu"]
    imp_scaling = impulse_params["scaling"]
    
    # Determine center location and width
    if config.USE_RESCALED:
        # Rescale desired value to [-1, 1] domain using x/4 - 1
        mu = rescale_to_unit_interval(np.array([config.DESIRED_VALUE]))[0]
        # Adjust width for rescaled domain: [0,8] -> [-1,1] has scale factor of 1/4
        width = sp_width / 4.0
        # Also adjust sigma for rescaled domain
        sigma_rescaled = imp_sigma / 4.0
    else:
        mu = config.DESIRED_VALUE
        width = sp_width
        sigma_rescaled = imp_sigma
    
    # Calculate impulse center
    if imp_mu == 0:
        impulse_center = mu  # Same as plateau center
    else:
        if config.USE_RESCALED:
            impulse_center = rescale_to_unit_interval(np.array([imp_mu]))[0]
        else:
            impulse_center = imp_mu
    
    # Sigmoid transitions for plateau region
    rise = 1 / (1 + np.exp(-sp_steepness * (x - (mu - width / 2))))
    fall = 1 / (1 + np.exp(-sp_steepness * (mu + width / 2 - x)))
    
    # Plateau mask (1 inside plateau, 0 outside)
    plateau_mask = rise * fall
    
    # Distance-based frequency damping (Option 4 approach)
    distance_from_center = abs(x - mu)
    
    # Create a smooth damping function that reduces oscillations away from plateau
    # This helps the polynomial approximation by reducing high-frequency content
    damping_width = width * 1.5  # Damping extends beyond plateau edges
    frequency_damping = np.exp(-(distance_from_center**2) / (2 * (damping_width/4)**2))
    
    # Clean plateau behavior (Option 3 approach)
    # Only include ripples where they're supposed to be, with smooth damping
    damped_ripples = sp_amplitude * np.sin(sp_freq * x * np.pi) * frequency_damping
    
    # Base plateau level (1.0) with damped ripples
    plateau_component = 1.0 + damped_ripples
    
    # Pure impulse behavior (no oscillatory components)
    x_shifted = x - impulse_center
    sigma2 = 2 * sigma_rescaled**2
    impulse_component = imp_scaling * np.exp(-(x_shifted**2) / sigma2)
    
    # Clean mathematical combination:
    # - Strong plateau behavior inside plateau region
    # - Clean impulse decay outside plateau region
    # - Damped oscillations create smooth polynomial-friendly transitions
    return plateau_component * plateau_mask + impulse_component * (1 - plateau_mask)


def plateau_sine_impulse_clean(x: float, epsilon: float = 0) -> float:
    """
    Clean separation version - no frequency leakage.
    
    This version completely eliminates any oscillatory components outside the plateau
    region to provide the cleanest possible impulse behavior for polynomial approximation.
    
    Mathematical approach:
    - Plateau region: clean 1.0 + small localized ripples 
    - Outside plateau: pure exponential decay only
    - No global oscillatory components whatsoever
    """
    # Get parameters
    plateau_params = config.get_function_params("plateau_sine")
    impulse_params = config.get_function_params("impulse")
    
    sp_amplitude = plateau_params["amplitude"]
    sp_freq = plateau_params["freq"]
    sp_steepness = plateau_params["steepness"]
    sp_width = plateau_params["width"]
    
    imp_sigma = impulse_params["sigma"]
    imp_mu = impulse_params["mu"]
    imp_scaling = impulse_params["scaling"]
    
    # Domain setup
    if config.USE_RESCALED:
        mu = rescale_to_unit_interval(np.array([config.DESIRED_VALUE]))[0]
        width = sp_width / 4.0
        sigma_rescaled = imp_sigma / 4.0
    else:
        mu = config.DESIRED_VALUE
        width = sp_width
        sigma_rescaled = imp_sigma
    
    impulse_center = mu if imp_mu == 0 else (rescale_to_unit_interval(np.array([imp_mu]))[0] if config.USE_RESCALED else imp_mu)
    
    # Sharp plateau definition
    rise = 1 / (1 + np.exp(-sp_steepness * (x - (mu - width / 2))))
    fall = 1 / (1 + np.exp(-sp_steepness * (mu + width / 2 - x)))
    plateau_mask = rise * fall
    
    # CLEAN separation: oscillations ONLY exist inside plateau
    if plateau_mask > 0.01:  # Only add ripples where plateau is significant
        ripples = sp_amplitude * np.sin(sp_freq * x * np.pi)
        plateau_component = 1.0 + ripples
    else:
        plateau_component = 0.0  # Clean zero outside plateau
    
    # Pure impulse (no oscillations)
    x_shifted = x - impulse_center
    impulse_component = imp_scaling * np.exp(-(x_shifted**2) / (2 * sigma_rescaled**2))
    
    return plateau_component * plateau_mask + impulse_component * (1 - plateau_mask)




def get_indicator_function(function_name: str) -> Callable[[float, float], float]:
    """
    Get an indicator function by name.
    
    Args:
        function_name: Name of the function ("impulse", "plateau_sine", "plateau_reg")
        
    Returns:
        Function that takes (x, epsilon) and returns the indicator value
        
    Raises:
        ValueError: If function_name is not recognized
    """
    function_map = {
        "impulse": impulse,
        "plateau_sine": plateau_sine,
        "plateau_reg": plateau_reg,
        "plateau_sine_impulse": plateau_sine_impulse,
        "plateau_sine_impulse_clean": plateau_sine_impulse_clean,
    }
    
    if function_name not in function_map:
        raise ValueError(f"Unknown function: {function_name}. Available: {list(function_map.keys())}")
    
    return function_map[function_name]


def evaluate_indicator_function(function_name: str, x_values: list, epsilon: float = 0) -> list:
    """
    Evaluate an indicator function at multiple points.
    
    Args:
        function_name: Name of the indicator function
        x_values: List of x values to evaluate
        epsilon: Epsilon parameter for the function
        
    Returns:
        List of function values at the input points
    """
    func = get_indicator_function(function_name)
    return [func(x, epsilon) for x in x_values]


def create_function_summary() -> dict:
    """
    Create a summary of all indicator functions and their parameters.
    
    Returns:
        Dictionary with function information and current parameter values
    """
    summary = {
        "available_functions": ["impulse", "plateau_sine", "plateau_reg", "plateau_sine_impulse", 
                               "plateau_sine_impulse_clean"],
        "current_function": config.FUNCTION_TYPE,
        "domain_mode": "rescaled [-1,1]" if config.USE_RESCALED else f"original [{config.MIN_VAL},{config.MAX_VAL}]",
        "desired_value": config.DESIRED_VALUE,
        "function_parameters": {}
    }
    
    # Add parameters for each function type
    for func_name in summary["available_functions"]:
        try:
            params = config.get_function_params(func_name)
            summary["function_parameters"][func_name] = params
        except ValueError:
            summary["function_parameters"][func_name] = "Not configured"
    
    return summary


if __name__ == "__main__":
    print("=== INDICATOR FUNCTIONS TEST ===")
    
    # Print function summary
    summary = create_function_summary()
    print(f"Available functions: {summary['available_functions']}")
    print(f"Current function: {summary['current_function']}")
    print(f"Domain mode: {summary['domain_mode']}")
    print(f"Desired value: {summary['desired_value']}")
    
    # Test each function
    test_epsilon = 0.01
    
    if config.USE_RESCALED:
        # Calculate desired value location in rescaled domain
        desired_rescaled = rescale_to_unit_interval(np.array([config.DESIRED_VALUE]))[0]
        # Include points ±0.1 from desired value for better testing
        test_points = [-1.0, desired_rescaled - 0.1, desired_rescaled, 
                      desired_rescaled + 0.1, 0.0, 0.5, 1.0]
        domain_str = "[-1, 1]"
    else:
        # Include points ±0.1 from desired value for better testing
        test_points = [0, config.DESIRED_VALUE - 0.1, config.DESIRED_VALUE, 
                      config.DESIRED_VALUE + 0.1, 4, 6, 8]
        domain_str = f"[{config.MIN_VAL}, {config.MAX_VAL}]"
    
    print(f"\nTesting functions at points: {test_points}")
    print(f"Domain: {domain_str}")
    print(f"Epsilon: {test_epsilon}")
    
    for func_name in summary["available_functions"]:
        print(f"\n--- {func_name.upper()} FUNCTION ---")
        
        # Get function parameters
        try:
            params = config.get_function_params(func_name)
            print(f"Parameters: {params}")
        except ValueError:
            print("Parameters: Not configured for this function")
        
        # Evaluate function
        try:
            values = evaluate_indicator_function(func_name, test_points, test_epsilon)
            
            print("Point   | Function Value")
            print("-" * 25)
            for point, value in zip(test_points, values):
                print(f"{point:6.2f} | {value:12.6f}")
            
            # Show statistics
            stats = utils.calculate_statistics(values)
            print(f"Statistics: min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f}")
            
        except Exception as e:
            print(f"Error evaluating {func_name}: {e}")
    
    # Test the configured function
    print(f"\n--- CONFIGURED FUNCTION ({config.FUNCTION_TYPE}) ---")
    configured_func = get_indicator_function(config.FUNCTION_TYPE)
    
    # Test evaluation grid for plotting
    if config.USE_RESCALED:
        eval_grid = np.linspace(-1.2, 1.2, 100)
    else:
        eval_grid = np.linspace(-1, config.MAX_VAL + 1, 100)
    
    func_values = [configured_func(x, test_epsilon) for x in eval_grid]
    
    print(f"Evaluated on {len(eval_grid)} point grid")
    print(f"Value range: [{min(func_values):.6f}, {max(func_values):.6f}]")
    
    # Find peak location
    max_idx = np.argmax(func_values)
    peak_location = eval_grid[max_idx]
    peak_value = func_values[max_idx]
    
    print(f"Peak at x={peak_location:.6f} with value={peak_value:.6f}")
    
    # Expected peak location
    if config.USE_RESCALED:
        expected_peak = rescale_to_unit_interval(np.array([config.DESIRED_VALUE]))[0]
    else:
        expected_peak = config.DESIRED_VALUE
    
    print(f"Expected peak at x={expected_peak:.6f}")
    print(f"Peak location error: {abs(peak_location - expected_peak):.6f}")
    
    print(f"\n=== INDICATOR FUNCTIONS TEST COMPLETE ===")