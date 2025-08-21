"""
Configuration for Polynomial Approximation Analysis

This module contains all configurable parameters for the system.
Modify values directly in this file to customize analysis behavior.
"""

# === Polynomial Approximation Settings ===
CHEB_DEGREE = 247  # Degree of polynomial approximation 59 119 247

# === Output Configuration ===
DATE_FOLDER = f"aug21_deg{CHEB_DEGREE}"  # Change this to update ALL output paths
GRAPHS_BASE_PATH = f"graphs/{DATE_FOLDER}"

# === Test Execution Settings ===
FUNCTION_TYPE = "plateau_sine"  # Options: "impulse", "plateau_sine", "plateau_reg", "plateau_sine_impulse"
POINTS_PER_VALUE = 10000  # How many test points per integer value (0, 1, 2, etc.)
USE_RESCALED = True  # True: [-1,1] domain, False: [0,8] domain

# === Domain Settings ===
DESIRED_VALUE = 2.0  # Target value for indicator functions
MAX_VAL = 8.0  # Maximum domain value (when not rescaled)
MIN_VAL = 0.0  # Minimum domain value (when not rescaled)

# === Epsilon Testing ===
MIN_EPSILON = 0.001  # Minimum epsilon for testing
MAX_EPSILON = 0.45  # Maximum epsilon for testing  
NUM_EPSILON_VALUES = 50  # Number of epsilon values to test
EXACTLY_EPSILON = False  # True: generate points exactly epsilon away, False: random within epsilon distance

# === Function-Specific Parameters ===
IMPULSE = {
    "sigma": 0.04,  # Standard deviation of Gaussian (default width)
    "mu": 0.0,  # Mean offset (0 = use DESIRED_VALUE)
    "scaling": 1.0,  # Amplitude scaling factor
}


# DEGREE 247
# PLATEAU_SINE = {
#     "amplitude": 0.00001,  # Amplitude of internal ripples
#     "base_amp": 0.00001,  # Base wave amplitude  
#     "base_freq": 20,  # Base wave frequency
#     "freq": 20,  # Internal ripple frequency
#     "steepness": 100,  # Steepness of sigmoid transitions
#     "width": 1.0,  # Width of plateau region
# }

# DEGREE 119
PLATEAU_SINE = {
    "amplitude": 0.0001,  # Amplitude of internal ripples
    "base_amp": 0.0001,  # Base wave amplitude  
    "base_freq": 10,  # Base wave frequency
    "freq": 10,  # Internal ripple frequency
    "steepness": 100,  # Steepness of sigmoid transitions
    "width": 1.0,  # Width of plateau region
}

# DEGREE 59
# PLATEAU_SINE = {
#     "amplitude": 0.01,  # Amplitude of internal ripples
#     "base_amp": 0.001,  # Base wave amplitude  
#     "base_freq": 10,  # Base wave frequency
#     "freq": 50,  # Internal ripple frequency
#     "steepness": 75,  # Steepness of sigmoid transitions
#     "width": 1.0,  # Width of plateau region
# }

# === Output Settings ===
ROUND_PRECISION = 4  # Decimal precision for outputs


def get_function_params(function_type: str = None) -> dict:
    """Get parameters for the specified function type."""
    func_type = function_type or FUNCTION_TYPE
    
    if func_type == "impulse":
        return IMPULSE.copy()
    elif func_type in ["plateau_sine", "plateau_reg", "plateau_sine_impulse"]:
        return PLATEAU_SINE.copy()
    else:
        raise ValueError(f"Unknown function type: {func_type}")


def print_config():
    """Print current configuration."""
    print("=== Configuration ===")
    print(f"Function Type: {FUNCTION_TYPE}")
    print(f"Domain: {'[-1,1]' if USE_RESCALED else f'[{MIN_VAL},{MAX_VAL}]'}")
    print(f"Desired Value: {DESIRED_VALUE}")
    print(f"Epsilon Range: {MIN_EPSILON} to {MAX_EPSILON} ({NUM_EPSILON_VALUES} values)")
    print(f"Exactly Epsilon: {EXACTLY_EPSILON}")
    print(f"Chebyshev Degree: {CHEB_DEGREE}")
    print(f"Points per Value: {POINTS_PER_VALUE}")
    print(f"Output Path: {GRAPHS_BASE_PATH}")
    
    if FUNCTION_TYPE in ["impulse", "plateau_sine", "plateau_reg"]:
        params = get_function_params()
        print(f"{FUNCTION_TYPE.title()} Parameters: {params}")