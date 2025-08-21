#!/usr/bin/env python3
"""
Manual Run Comparison Tool

This script allows manual comparison of specific run IDs or directories.
Use this when you want fine-grained control over which runs to compare.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
from create_graphs import create_comparison_plots


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare specific analysis runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare specific run IDs (searches in current date folder)
  python compare_runs.py --ids abc1 def2 ghi3
  
  # Compare specific run IDs with comma separation
  python compare_runs.py --ids abc1,def2,ghi3
  
  # Compare all runs in specific directories  
  python compare_runs.py --directories graphs/aug13_1 graphs/aug13_2
  
  # Compare specific IDs across multiple directories
  python compare_runs.py --ids abc1,def2 --directories graphs/aug13_1 graphs/aug13_2
  
  # Compare all runs in current date folder (same as automatic mode)
  python compare_runs.py --directory graphs/aug13_3 --all
        """
    )
    
    parser.add_argument(
        '--ids', 
        type=str, 
        help='Comma-separated list of run IDs to compare (e.g., abc1,def2,ghi3)'
    )
    
    parser.add_argument(
        '--directories', '--dirs',
        nargs='+',
        type=str,
        help='Directories to search for CSV files'
    )
    
    parser.add_argument(
        '--directory', '--dir',
        type=str,
        help='Single directory to search (use with --all for automatic mode)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Compare all available runs (automatic mode)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for comparison plots (default: <dir>/comparisons/)'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate argument combinations."""
    if not any([args.ids, args.directories, args.directory]):
        print("Error: Must specify at least one of --ids, --directories, or --directory")
        return False
    
    if args.directory and args.directories:
        print("Error: Cannot use both --directory and --directories")
        return False
    
    return True


def parse_run_ids(ids_string: str) -> List[str]:
    """Parse comma-separated run IDs."""
    if not ids_string:
        return []
    
    # Split by comma and clean up whitespace
    ids = [id_str.strip() for id_str in ids_string.split(',')]
    return [id_str for id_str in ids if id_str]  # Remove empty strings


def determine_base_directory(args) -> Path:
    """Determine the base directory for output."""
    if args.output:
        return Path(args.output).parent
    
    if args.directory:
        return Path(args.directory)
    elif args.directories:
        return Path(args.directories[0])
    else:
        # Default to current config directory
        import config
        return Path(config.GRAPHS_BASE_PATH)


def main():
    """Main entry point for manual comparison tool."""
    args = parse_arguments()
    
    if not validate_arguments(args):
        sys.exit(1)
    
    try:
        # Parse run IDs
        specific_run_ids = parse_run_ids(args.ids) if args.ids else None
        
        # Determine directories to search
        if args.directories:
            specific_directories = [Path(d) for d in args.directories]
        elif args.directory:
            specific_directories = [Path(args.directory)]
        else:
            specific_directories = None
        
        # Determine base directory for output
        base_directory = determine_base_directory(args)
        
        # Print configuration
        print("=" * 60)
        print("MANUAL RUN COMPARISON")
        print("=" * 60)
        print(f"Base directory: {base_directory}")
        
        if specific_run_ids:
            print(f"Target run IDs: {specific_run_ids}")
        else:
            print("Mode: Compare all available runs")
            
        if specific_directories:
            print(f"Search directories: {specific_directories}")
        
        # Create comparison plots
        create_comparison_plots(
            base_directory=base_directory,
            current_run_data=None,
            specific_run_ids=specific_run_ids,
            specific_directories=specific_directories
        )
        
        print("=" * 60)
        print("COMPARISON COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()