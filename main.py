#!/usr/bin/env python3
"""
Main Application Runner

This runs polynomial approximation analysis based on config.py settings.
It analyzes the configured function with the configured epsilon range.
"""

import sys
from error_analysis import run_config_based_analysis, save_results
import create_graphs


def main():
    """Main application entry point - orchestrates the complete analysis workflow."""
    try:
        # Run analysis based on config
        results = run_config_based_analysis()
        
        # Create comprehensive visualizations
        print(f"\n{'='*60}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*60}")
        
        # Create function approximation plot
        create_graphs.create_function_approximation_plot(results)
        
        # Create error analysis plots
        create_graphs.create_connected_dot_plots(results)
        
        # Save results
        save_results(results)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()