#!/usr/bin/env python3
"""
Main Runner for Polynomial Approximation Tests

This script runs epsilon variation tests using parameters from config.py.
Simply modify config.py and run this script to execute tests with your desired settings.

Usage:
    python main.py
"""

import numpy as np
from indicator_functions import FunctionConfig
from epsilon_variation_test import run_single_function_epsilon_test
import config

def create_function_config():
    """Create FunctionConfig from centralized config parameters."""
    
    # Determine epsilon testing mode
    if config.SINGLE_EPSILON is not None:
        # Single epsilon mode
        epsilon_values = [config.SINGLE_EPSILON]
        test_mode = "single"
    else:
        # Epsilon range mode
        epsilon_values = np.linspace(
            config.MIN_EPSILON, 
            config.MAX_EPSILON, 
            config.NUM_EPSILON_VALUES
        ).tolist()
        test_mode = "range"
    
    # Calculate total batch size (points per value × number of values)
    # Domain has values 0 through MAX_VAL, so (MAX_VAL + 1) total values
    total_batch_size = int(config.POINTS_PER_VALUE * (config.MAX_VAL + 1))
    
    # Create configuration with all parameters from config.py
    function_config = FunctionConfig(
        function_name=config.FUNCTION_TYPE,
        epsilons=epsilon_values[0],  # Base epsilon for config
        num_trials=1,  # Simplified to always 1
        cheb_degree=config.CHEB_DEGREE,
        use_rescaled=config.USE_RESCALED,
        batch_size=total_batch_size,
        desired_value=config.DESIRED_VALUE,
        max_val=int(config.MAX_VAL),
        min_val=int(config.MIN_VAL),
        round_precision=config.ROUND_PRECISION,
        
        # Impulse function parameters
        impulse_mu=config.IMPULSE_MU if config.IMPULSE_MU != 0 else config.DESIRED_VALUE,
        impulse_scaling=config.IMPULSE_SCALING,
        impulse_sigma=config.IMPULSE_SIGMA,
        
        # Plateau function parameters
        sp_amplitude=config.SP_AMPLITUDE,
        sp_base_amp=config.SP_BASE_AMP,
        sp_base_freq=config.SP_BASE_FREQ,
        sp_freq=config.SP_FREQ,
        sp_steepness=config.SP_STEEPNESS,
        sp_width=config.SP_WIDTH,
    )
    
    return function_config, epsilon_values, test_mode

def main():
    """Main execution function."""
    
    print("=" * 80)
    print("Polynomial Approximation Test Runner")
    print("=" * 80)
    print(f"Function Type: {config.FUNCTION_TYPE}")
    print(f"Domain: {'[-1, 1] (rescaled)' if config.USE_RESCALED else '[0, 8] (original)'}")
    print(f"Test Points: {config.POINTS_PER_VALUE} per integer value")
    print(f"Chebyshev Degree: {config.CHEB_DEGREE}")
    print(f"Output Directory: {config.GRAPHS_BASE_PATH}")
    
    # Create function configuration and epsilon values
    function_config, epsilon_values, test_mode = create_function_config()
    
    # Display epsilon testing mode
    if test_mode == "single":
        print(f"Epsilon Mode: Single value ({config.SINGLE_EPSILON})")
    else:
        print(f"Epsilon Mode: Range from {config.MIN_EPSILON} to {config.MAX_EPSILON} ({len(epsilon_values)} values)")
    
    print("-" * 80)
    
    # Display function-specific parameters
    if config.FUNCTION_TYPE == "impulse":
        print("Impulse Function Parameters:")
        print(f"  Sigma: {config.IMPULSE_SIGMA}")
        print(f"  Mu: {config.IMPULSE_MU} (0 = use desired_value)")
        print(f"  Scaling: {config.IMPULSE_SCALING}")
    elif config.FUNCTION_TYPE.startswith("plateau"):
        print("Plateau Function Parameters:")
        print(f"  Amplitude: {config.SP_AMPLITUDE}")
        print(f"  Base Amp: {config.SP_BASE_AMP}")
        print(f"  Frequency: {config.SP_FREQ}")
        print(f"  Steepness: {config.SP_STEEPNESS}")
        print(f"  Width: {config.SP_WIDTH}")
    
    print("-" * 80)
    print("Starting test...")
    print()
    
    # Run the test
    if test_mode == "single":
        print(f"Testing single epsilon: {config.SINGLE_EPSILON}")
    else:
        print(f"Testing {len(epsilon_values)} epsilon values...")
    
    results = run_single_function_epsilon_test(
        function_name=config.FUNCTION_TYPE,
        base_config=function_config,
        epsilon_values=epsilon_values,
        cheb_degree=config.CHEB_DEGREE,
        test_both_domains=config.TEST_BOTH_DOMAINS
    )
    
    print()
    print("=" * 80)
    print("Test Complete!")
    print(f"Results: {len(results)} evaluations completed")
    print(f"Files saved to: {config.GRAPHS_BASE_PATH}/{config.FUNCTION_TYPE}/")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    main()