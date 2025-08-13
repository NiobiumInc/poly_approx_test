"""
Utility Functions

This module provides common utility functions used throughout the 
polynomial approximation analysis system.
"""

import os
import csv
import random
import string
import numpy as np
from typing import List, Tuple, Union, Optional
from pathlib import Path


def create_unique_id(length: int = 4) -> str:
    """
    Create a unique alphanumeric identifier.
    
    Args:
        length: Length of the ID (default: 4)
        
    Returns:
        Random lowercase alphanumeric string
    """
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object of the created/existing directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def safe_filename(name: str, max_length: int = 255) -> str:
    """
    Create a safe filename by removing/replacing problematic characters.
    
    Args:
        name: Original filename
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename
    """
    # Remove problematic characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    safe_name = ''.join(c if c in safe_chars else '_' for c in name)
    
    # Ensure it doesn't start with a dot or dash
    if safe_name.startswith(('.', '-')):
        safe_name = 'file_' + safe_name
    
    # Truncate if too long
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]
    
    return safe_name


def format_number(value: float, precision: int = 4) -> str:
    """
    Format a number with specified precision, using scientific notation for very small/large values.
    
    Args:
        value: Number to format
        precision: Decimal places for formatting
        
    Returns:
        Formatted number string
    """
    if abs(value) < 10**(-precision) or abs(value) >= 10**(precision + 1):
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"


def validate_array_shapes(*arrays: np.ndarray) -> bool:
    """
    Validate that all arrays have the same shape.
    
    Args:
        *arrays: Variable number of numpy arrays
        
    Returns:
        True if all arrays have the same shape
    """
    if len(arrays) < 2:
        return True
    
    reference_shape = arrays[0].shape
    return all(arr.shape == reference_shape for arr in arrays[1:])


def calculate_statistics(values: List[float]) -> dict:
    """
    Calculate basic statistics for a list of values.
    
    Args:
        values: List of numerical values
        
    Returns:
        Dictionary with statistical measures
    """
    if not values:
        return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0}
    
    values_array = np.array(values)
    return {
        "count": len(values),
        "mean": float(np.mean(values_array)),
        "std": float(np.std(values_array)),
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
        "median": float(np.median(values_array))
    }


def linspace_with_endpoints(start: float, stop: float, num: int, 
                           include_endpoints: bool = True) -> np.ndarray:
    """
    Create evenly spaced values with explicit endpoint control.
    
    Args:
        start: Start value
        stop: Stop value  
        num: Number of points
        include_endpoints: Whether to include start/stop points
        
    Returns:
        Array of evenly spaced values
    """
    if include_endpoints:
        return np.linspace(start, stop, num)
    else:
        # Exclude endpoints by creating num+2 points and taking the middle ones
        full_range = np.linspace(start, stop, num + 2)
        return full_range[1:-1]


def split_array_by_condition(array: np.ndarray, condition: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split an array into two parts based on a boolean condition.
    
    Args:
        array: Array to split
        condition: Boolean array of same length
        
    Returns:
        Tuple of (values_where_true, values_where_false)
    """
    if len(array) != len(condition):
        raise ValueError("Array and condition must have the same length")
    
    true_values = array[condition]
    false_values = array[~condition]
    
    return true_values, false_values


def clamp_value(value: float, min_val: float = None, max_val: float = None) -> float:
    """
    Clamp a value between minimum and maximum bounds.
    
    Args:
        value: Value to clamp
        min_val: Minimum allowed value (None = no minimum)
        max_val: Maximum allowed value (None = no maximum)
        
    Returns:
        Clamped value
    """
    if min_val is not None and value < min_val:
        return min_val
    if max_val is not None and value > max_val:
        return max_val
    return value


def get_file_extension(filename: str) -> str:
    """
    Get the file extension from a filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension (without the dot)
    """
    return Path(filename).suffix.lstrip('.')


def log_to_csv(log_path: Path, log_entry: dict) -> None:
    """
    Log a dictionary entry to CSV file, creating headers if file doesn't exist.
    
    Args:
        log_path: Path to the CSV log file
        log_entry: Dictionary to log as a CSV row
    """
    file_exists = log_path.exists()
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)


def create_output_filename(base_name: str, suffix: str = "", 
                          extension: str = "csv", unique_id: str = None) -> str:
    """
    Create a standardized output filename.
    
    Args:
        base_name: Base name of the file
        suffix: Optional suffix to add
        extension: File extension (without dot)
        unique_id: Optional unique identifier (generated if None)
        
    Returns:
        Complete filename with ID
    """
    if unique_id is None:
        unique_id = create_unique_id()
    
    parts = [base_name]
    if suffix:
        parts.append(suffix)
    parts.append(unique_id)
    
    filename = "_".join(parts) + f".{extension}"
    return safe_filename(filename)


def find_nearest_value(array: np.ndarray, target: float) -> Tuple[int, float]:
    """
    Find the index and value in array closest to target.
    
    Args:
        array: Array to search
        target: Target value
        
    Returns:
        Tuple of (index, value) of nearest element
    """
    array = np.asarray(array)
    idx = np.abs(array - target).argmin()
    return int(idx), array[idx]


def round_to_significant_digits(value: float, digits: int = 3) -> float:
    """
    Round a number to a specified number of significant digits.
    
    Args:
        value: Number to round
        digits: Number of significant digits
        
    Returns:
        Rounded value
    """
    if value == 0:
        return 0
    
    from math import log10, floor
    return round(value, -int(floor(log10(abs(value)))) + (digits - 1))


def batch_process(items: List, batch_size: int = 100, process_func=None):
    """
    Process items in batches to manage memory usage.
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        process_func: Function to apply to each batch
        
    Yields:
        Processed batches or raw batches if no process_func
    """
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        if process_func:
            yield process_func(batch)
        else:
            yield batch


def memory_usage_mb() -> float:
    """
    Get current memory usage in MB.
    
    Returns:
        Memory usage in megabytes
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0  # Return 0 if psutil not available


def vectorized_function_evaluation(func, test_points, *args, **kwargs):
    """
    Efficiently evaluate a function over array of points.
    
    Args:
        func: Function to evaluate  
        test_points: Array of evaluation points
        *args, **kwargs: Additional arguments to pass to func
        
    Returns:
        Array of function values
    """
    # Try vectorized evaluation first
    try:
        return np.vectorize(func)(test_points, *args, **kwargs)
    except:
        # Fall back to list comprehension if vectorization fails
        return np.array([func(x, *args, **kwargs) for x in test_points])


def batch_process_large_arrays(func, data_array, batch_size: int = 10000):
    """
    Process large arrays in batches to manage memory usage.
    
    Args:
        func: Function to apply to each batch
        data_array: Large array to process
        batch_size: Size of each processing batch
        
    Returns:
        Concatenated results from all batches
    """
    results = []
    for i in range(0, len(data_array), batch_size):
        batch = data_array[i:i + batch_size]
        batch_result = func(batch)
        results.append(batch_result)
    
    return np.concatenate(results) if results else np.array([])


def timer_context():
    """
    Context manager for timing code execution.
    
    Usage:
        with timer_context() as timer:
            # code to time
            pass
        print(f"Elapsed time: {timer.elapsed:.3f} seconds")
    """
    import time
    
    class Timer:
        def __init__(self):
            self.elapsed = 0
            
        def __enter__(self):
            self.start = time.time()
            return self
            
        def __exit__(self, *args):
            self.elapsed = time.time() - self.start
    
    return Timer()



if __name__ == "__main__":
    print("=== UTILITY FUNCTIONS TEST ===")
    
    # Test ID generation
    print(f"Unique ID: {create_unique_id()}")
    print(f"Long ID: {create_unique_id(8)}")
    
    # Test number formatting
    test_numbers = [0.000123, 1234.5678, 0.0000001, 999999.99]
    print(f"\nNumber formatting:")
    for num in test_numbers:
        print(f"  {num} → {format_number(num)}")
    
    # Test statistics
    test_data = [1.2, 3.4, 2.1, 4.8, 1.9, 3.7, 2.5]
    stats = calculate_statistics(test_data)
    print(f"\nStatistics for {test_data}:")
    for key, value in stats.items():
        print(f"  {key}: {format_number(value)}")
    
    # Test array splitting
    test_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    condition = test_array > 5
    true_vals, false_vals = split_array_by_condition(test_array, condition)
    print(f"\nArray splitting (>5): {true_vals} | {false_vals}")
    
    # Test filename creation
    print(f"\nFilename creation:")
    print(f"  Base: {create_output_filename('test', 'results', 'csv')}")
    print(f"  With suffix: {create_output_filename('experiment', 'epsilon_sweep', 'png')}")
    
    # Test nearest value
    test_array = np.array([0.1, 0.3, 0.7, 1.2, 1.8, 2.5])
    target = 1.0
    idx, val = find_nearest_value(test_array, target)
    print(f"\nNearest to {target} in {test_array}: index {idx}, value {val}")
    
    # Test significant digits
    test_vals = [0.123456, 1234.567, 0.00012345]
    print(f"\nSignificant digits (3):")
    for val in test_vals:
        rounded = round_to_significant_digits(val, 3)
        print(f"  {val} → {rounded}")
    
    # Test timing
    with timer_context() as timer:
        # Simulate some work
        sum(i**2 for i in range(10000))
    print(f"\nTiming test: {timer.elapsed:.6f} seconds")
    
    print("\n=== UTILITY FUNCTIONS TEST COMPLETE ===")