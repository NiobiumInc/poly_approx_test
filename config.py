"""
Centralized Configuration for Polynomial Approximation Tests

This file contains ALL configurable parameters for the entire project.
Simply modify the values below and run main.py to execute tests with your desired settings.
"""

# === Output Configuration ===
DATE_FOLDER = "aug4_2"  # Change this to update ALL output paths
GRAPHS_BASE_PATH = f"graphs/{DATE_FOLDER}"

# === Test Execution Settings ===
FUNCTION_TYPE = "impulse"  # Options: "impulse", "plateau_sine", "plateau_reg"
POINTS_PER_VALUE = 1000  # How many test points per integer value (0, 1, 2, etc.)
USE_RESCALED = True  # True: [-1,1] domain, False: [0,8] domain
TEST_BOTH_DOMAINS = False  # True: test both domains, False: use USE_RESCALED only

# === Domain Settings ===
DESIRED_VALUE = 2.0  # Target value for indicator functions
MAX_VAL = 8.0  # Maximum domain value (when not rescaled)
MIN_VAL = 0.0  # Minimum domain value (when not rescaled)

# === Epsilon Testing Mode (Choose ONE) ===
# Option 1: Test single epsilon value
SINGLE_EPSILON = None  # Set to None to disable single epsilon mode

# Option 2: Test range of epsilon values (set SINGLE_EPSILON = None to use this)
MIN_EPSILON = 0.001  # Minimum epsilon for testing
MAX_EPSILON = 0.05   # Maximum epsilon for testing  
NUM_EPSILON_VALUES = 20  # Number of epsilon values to test

# === Chebyshev Approximation Settings ===
CHEB_DEGREE = 119  # Degree of Chebyshev polynomial approximation

# === Impulse Function Parameters ===
IMPULSE_SIGMA = 0.04  # Standard deviation of Gaussian
IMPULSE_MU = 0.0  # Mean offset (0 = use DESIRED_VALUE)
IMPULSE_SCALING = 1.0  # Amplitude scaling factor

# === Plateau Sine Function Parameters ===
SP_AMPLITUDE = 0.001  # Amplitude of internal ripples
SP_BASE_AMP = 0.001  # Base wave amplitude  
SP_BASE_FREQ = 10  # Base wave frequency
SP_FREQ = 25  # Internal ripple frequency
SP_STEEPNESS = 100  # Steepness of sigmoid transitions
SP_WIDTH = 1.0  # Width of plateau region

# === Output Settings ===
ROUND_PRECISION = 4  # Decimal precision for outputs