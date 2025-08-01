"""
Indicator Functions for Polynomial Approximation

This module implements various indicator and test functions for polynomial approximation,
including Gaussian impulses and plateau functions with configurable parameters.
All functions support parameter tracking for logging and analysis.
"""

import numpy as np
import math
import random
import string
import csv
import os
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from generate_approx_integers import rescale_to_unit_interval


# Default parameter constants
DEFAULT_DESIRED_VALUE = 2
DEFAULT_MAX_VAL = 8
DEFAULT_MIN_VAL = 0
DEFAULT_SP_BASE_AMP = 0.1
DEFAULT_SP_BASE_FREQ = 2.0
DEFAULT_SP_AMP = 0.3
DEFAULT_SP_FREQ = 8.0
DEFAULT_SP_STEEPNESS = 10.0


@dataclass
class FunctionConfig:
    """Configuration class to track all parameters for a function run."""
    id: str = ""
    function_name: str = ""
    epsilons: float = 0.1
    num_trials: int = 1
    cheb_degree: int = 10
    use_rescaled: bool = False
    batch_size: int = 80
    desired_value: float = DEFAULT_DESIRED_VALUE
    max_val: float = DEFAULT_MAX_VAL
    min_val: float = DEFAULT_MIN_VAL
    interval_type: str = "uniform"
    round_precision: int = 6
    
    # Impulse function parameters
    impulse_mu: float = DEFAULT_DESIRED_VALUE
    impulse_scaling: float = 1.0
    impulse_sigma: float = 0.5
    
    # Plateau function parameters
    sp_amplitude: float = DEFAULT_SP_AMP
    sp_base_amp: float = DEFAULT_SP_BASE_AMP
    sp_base_freq: float = DEFAULT_SP_BASE_FREQ
    sp_freq: float = DEFAULT_SP_FREQ
    sp_steepness: float = DEFAULT_SP_STEEPNESS
    sp_width: float = 1.0
    
    # Additional tracking fields
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging."""
        result = {}
        for key, value in self.__dict__.items():
            if key != 'additional_params':
                result[key] = value
            else:
                result.update(value)
        return result
    
    def print_config(self):
        """Print all configuration parameters."""
        print(f"=== Function Configuration (ID: {self.id}) ===")
        config_dict = self.to_dict()
        for key, value in sorted(config_dict.items()):
            print(f"{key}: {value}")
    
    def generate_unique_id(self) -> str:
        """Generate a unique 4-character lowercase alphanumeric ID."""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    
    def to_csv_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for CSV logging with n/a for unused parameters."""
        base_dict = {
            'id': self.id,
            'function_name': self.function_name,
            'epsilons': self.epsilons,
            'num_trials': self.num_trials,
            'cheb_degree': self.cheb_degree,
            'use_rescaled': self.use_rescaled,
            'batch_size': self.batch_size,
            'desired_value': self.desired_value,
            'max_val': self.max_val,
            'min_val': self.min_val,
            'interval_type': self.interval_type,
            'round_precision': self.round_precision,
        }
        
        # Add function-specific parameters with n/a for unused ones
        if self.function_name == 'impulse':
            base_dict.update({
                'impulse_mu': self.impulse_mu,
                'impulse_scaling': self.impulse_scaling,
                'impulse_sigma': self.impulse_sigma,
                'sp_amplitude': 'n/a',
                'sp_base_amp': 'n/a',  
                'sp_base_freq': 'n/a',
                'sp_freq': 'n/a',
                'sp_steepness': 'n/a',
                'sp_width': 'n/a',
            })
        elif self.function_name in ['plateau_sine', 'plateau_reg']:
            base_dict.update({
                'impulse_mu': 'n/a',
                'impulse_scaling': 'n/a',
                'impulse_sigma': 'n/a',
                'sp_amplitude': self.sp_amplitude,
                'sp_base_amp': self.sp_base_amp,
                'sp_base_freq': self.sp_base_freq,
                'sp_freq': self.sp_freq,
                'sp_steepness': self.sp_steepness,
                'sp_width': self.sp_width,
            })
        else:
            # Default: all n/a
            base_dict.update({
                'impulse_mu': 'n/a',
                'impulse_scaling': 'n/a',
                'impulse_sigma': 'n/a',
                'sp_amplitude': 'n/a',
                'sp_base_amp': 'n/a',
                'sp_base_freq': 'n/a',
                'sp_freq': 'n/a',
                'sp_steepness': 'n/a',
                'sp_width': 'n/a',
            })
        
        return base_dict
    
    def log_to_csv(self, csv_path: str = "graphs/aug1/run_log.csv"):
        """Log configuration to CSV file."""
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        csv_dict = self.to_csv_dict()
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = list(csv_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(csv_dict)
        
        print(f"Configuration logged to {csv_path}")


def gaussian_impulse(x: float, mu: float = DEFAULT_DESIRED_VALUE, 
                    sigma: float = 0.5, scaling: Optional[float] = None) -> float:
    """
    Normalized Gaussian impulse function (probability density function).
    
    Args:
        x: Input value
        mu: Mean (center) of the Gaussian
        sigma: Standard deviation
        scaling: Optional scaling factor. If None, uses normalized Gaussian (1/(sigma*sqrt(2π)))
    
    Returns:
        Gaussian impulse value at x
    """
    x_shifted = x - mu
    sigma2 = 2 * sigma**2
    
    if scaling is None:
        scaling = 1.0 / (sigma * math.sqrt(2 * math.pi))
    
    return scaling * np.exp(-(x_shifted**2) / sigma2)


def plateau_sine(x: float, config: FunctionConfig) -> float:
    """
    Plateau function with smooth sigmoid transitions and internal ripples,
    supporting optional rescaled inputs.

    Args:
        x: Input value
        config: Function configuration containing all parameters

    Returns:
        Output of the plateau function
    """
    epsilon = config.epsilons
    sp_base_amp = config.sp_base_amp
    sp_base_freq = config.sp_base_freq
    sp_amplitude = config.sp_amplitude
    sp_freq = config.sp_freq
    sp_steepness = config.sp_steepness
    sp_width = config.sp_width
    
    if config.use_rescaled:
        # Rescale DESIRED_VALUE and width to [-1, 1]
        mu = rescale_to_unit_interval(
            np.array([config.desired_value]), max_value=config.max_val
        )[0]
        width = sp_width * (2 / (config.max_val + epsilon - (config.min_val - epsilon)))
    else:
        mu = config.desired_value
        width = sp_width

    base_wave = sp_base_amp * np.sin(sp_base_freq * x * np.pi)

    rise = 1 / (1 + np.exp(-sp_steepness * (x - (mu - width / 2))))
    fall = 1 / (1 + np.exp(-sp_steepness * (mu + width / 2 - x)))

    plateau_mask = rise * fall
    ripple_in_plateau = sp_amplitude * np.sin(sp_freq * x * np.pi)

    return (1 + ripple_in_plateau) * plateau_mask + base_wave * (1 - plateau_mask)


def plateau_reg(x: float, config: FunctionConfig) -> float:
    """
    Hard cutoff function indicating whether x lies within DESIRED_VALUE ± 0.5,
    supporting optional rescaling.

    Args:
        x: Input value
        config: Function configuration containing all parameters

    Returns:
        1.0 if x is inside the hard interval, 0.0 otherwise
    """
    epsilon = config.epsilons
    
    if config.use_rescaled:
        low, high = rescale_to_unit_interval(
            np.array([config.desired_value - 0.5, config.desired_value + 0.5]),
            max_value=config.max_val
        )
        return 1.0 if low <= x <= high else 0.0
    else:
        return 1.0 if (config.desired_value - 0.5 <= x <= config.desired_value + 0.5) else 0.0


class FunctionWrapper:
    """Wrapper class to create function instances with tracked parameters."""
    
    def __init__(self, config: FunctionConfig):
        self.config = config
        
    def create_gaussian_impulse(self) -> Callable[[float], float]:
        """Create a Gaussian impulse function with tracked parameters."""
        def func(x: float) -> float:
            # Handle rescaling if needed
            if self.config.use_rescaled:
                # Rescale mu to [-1, 1] range
                mu_rescaled = rescale_to_unit_interval(
                    np.array([self.config.impulse_mu]), max_value=self.config.max_val
                )[0]
                # Adjust sigma for the rescaled domain
                sigma_rescaled = self.config.impulse_sigma * (2.0 / self.config.max_val)
            else:
                mu_rescaled = self.config.impulse_mu
                sigma_rescaled = self.config.impulse_sigma
                
            return gaussian_impulse(
                x, 
                mu=mu_rescaled,
                sigma=sigma_rescaled,
                scaling=self.config.impulse_scaling
            )
        return func
    
    def create_plateau_sine(self) -> Callable[[float], float]:
        """Create a plateau sine function with tracked parameters."""
        def func(x: float) -> float:
            return plateau_sine(x, self.config)
        return func
    
    def create_plateau_reg(self) -> Callable[[float], float]:
        """Create a plateau reg function with tracked parameters."""
        def func(x: float) -> float:
            return plateau_reg(x, self.config)
        return func


def get_function_by_name(name: str, config: FunctionConfig) -> Callable[[float], float]:
    """
    Get a function by name with the given configuration.
    
    Args:
        name: Function name ('impulse', 'plateau_sine', 'plateau_reg')
        config: Function configuration
        
    Returns:
        Configured function
    """
    wrapper = FunctionWrapper(config)
    
    function_map = {
        'impulse': wrapper.create_gaussian_impulse,
        'plateau_sine': wrapper.create_plateau_sine,
        'plateau_reg': wrapper.create_plateau_reg,
    }
    
    if name not in function_map:
        raise ValueError(f"Unknown function name: {name}. Available: {list(function_map.keys())}")
    
    config.function_name = name
    return function_map[name]()


def run_single_function_demo(function_name: str, use_rescaled: bool = False):
    """
    Run demo for a single function with rescaling support.
    
    Args:
        function_name: Name of function to test ('impulse', 'plateau_sine', 'plateau_reg')
        use_rescaled: Whether to use rescaled domain [-1,1] or original [0,8]
    """
    import matplotlib.pyplot as plt
    
    # Create test configuration
    config = FunctionConfig(
        id="",  # Will be generated
        function_name=function_name,
        desired_value=2.0,
        impulse_mu=2.0,
        impulse_sigma=0.3,
        use_rescaled=use_rescaled
    )
    
    # Generate unique ID
    config.id = config.generate_unique_id()
    
    # Set domain based on rescaling
    if use_rescaled:
        x_vals = np.linspace(-1, 1, 200)
        domain_str = "[-1,1]"
    else:
        x_vals = np.linspace(0, 8, 200)
        domain_str = "[0,8]"
    
    # Get function
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
        desired_rescaled = rescale_to_unit_interval(
            np.array([config.desired_value]), max_value=config.max_val
        )[0]
        ax.axvline(x=desired_rescaled, color='r', linestyle='--', alpha=0.7, label='Desired Value')
    else:
        ax.axvline(x=config.desired_value, color='r', linestyle='--', alpha=0.7, label='Desired Value')
    ax.legend()
    
    plt.tight_layout()
    
    # Create function-specific folder and save with ID at end
    rescaled_suffix = "rescaled" if use_rescaled else "original"
    function_folder = f'graphs/aug1/{function_name}'
    os.makedirs(function_folder, exist_ok=True)
    
    filename = f'{function_name}_{rescaled_suffix}_{config.id}.png'
    filepath = f'{function_folder}/{filename}'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {filepath}")
    config.print_config()
    
    # Log to CSV
    config.log_to_csv()
    
    return config


def demo_functions():
    """Demonstrate running individual functions separately."""
    print("=== Individual Function Demos ===")
    
    # Run each function individually
    functions_to_test = ['impulse', 'plateau_sine', 'plateau_reg']
    
    for func_name in functions_to_test:
        print(f"\n--- Testing {func_name} ---")
        
        # Test original domain
        print("Original domain [0,8]:")
        config1 = run_single_function_demo(func_name, use_rescaled=False)
        
        # Test rescaled domain  
        print("\nRescaled domain [-1,1]:")
        config2 = run_single_function_demo(func_name, use_rescaled=True)
        
        print("-" * 50)


if __name__ == "__main__":
    demo_functions()