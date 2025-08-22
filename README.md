# Polynomial Approximation Test Suite

A comprehensive analysis framework for testing Chebyshev polynomial approximations of indicator functions, with focus on performance under input noise and domain variations.

## Overview
This project evaluates how well Chebyshev polynomial approximations of indicator functions perform when evaluated on "approximate integer" points with varying epsilon values. The system supports multiple indicator function types, domain rescaling between [0,8] and [-1,1], and comprehensive error analysis across different polynomial degrees.

**Key Features:**
- 4 different indicator function implementations
- Dual domain support (original [0,8] and rescaled [-1,1])  
- Configurable Chebyshev polynomial degrees (59, 119, 247+)
- Comprehensive error analysis for desired vs non-desired regions
- Automated visualization and comparison generation
- Extensive parameter sweep capabilities

## Quick Start
1. **Configure parameters**: Edit `config.py` (see Configuration section)
2. **Run tests**: `python main.py`
3. **Verify system**: `python verify_tests.py`
4. **View results**: Check `graphs/{DATE_FOLDER}/{FUNCTION_TYPE}/`

## Configuration
All parameters are centralized in `config.py`:

### Key Settings
- **Line 13**: `FUNCTION_TYPE` - Choose function: `"impulse"`, `"plateau_sine"`, `"plateau_reg"`
- **Line 15**: `USE_RESCALED` - Domain: `True` for [-1,1], `False` for [0,8]
- **Line 25**: `SINGLE_EPSILON` - Single value or `None` for range testing

### Test Data
- **Line 14**: `POINTS_PER_VALUE` - How many test points per integer (0,1,2,...,8)

## Indicator Function Types

### 1. **Impulse Function** (`"impulse"`)
- **Description**: Gaussian probability density centered at desired value
- **Output**: Peak of 1.0 at desired value, decays smoothly
- **Parameters**: `IMPULSE_SIGMA`, `IMPULSE_MU`, `IMPULSE_SCALING`
- **Use case**: Smooth, differentiable indicator

### 2. **Plateau Sine Function** (`"plateau_sine"`)  
- **Description**: Smooth plateau with internal ripples and sigmoid edges
- **Output**: Plateau region with sine wave ripples, smooth transitions
- **Parameters**: `SP_AMPLITUDE`, `SP_STEEPNESS`, `SP_FREQ`, `SP_WIDTH`
- **Use case**: Complex function with multiple frequency components

### 3. **Plateau Regular Function** (`"plateau_reg"`)
- **Description**: Hard cutoff indicator function
- **Output**: `1` for x ∈ [desired_value-0.5, desired_value+0.5], `0` elsewhere
- **Parameters**: Uses `DESIRED_VALUE` only
- **Use case**: Traditional step function indicator

## Domain Rescaling
The system supports two domains:

### Original Domain [0, 8]
- **When**: `USE_RESCALED = False`
- **Function evaluation**: Direct on [0,8] 
- **Test points**: Approximate integers 0,1,2,3,4,5,6,7,8 ± epsilon

### Rescaled Domain [-1, 1]
- **When**: `USE_RESCALED = True`
- **Mapping**: `x_rescaled = x_original / 4.0 - 1.0`
- **Key mappings**: 0→-1, 2→-0.5, 4→0, 8→1
- **Function evaluation**: Functions adapted to work on [-1,1]

## Test Methodology

### 1. Generate Test Points
- **Total points**: `POINTS_PER_VALUE × 9` (for integers 0-8)
- **Point distribution**: Equal number of points near each integer
- **Noise**: Each point = `integer + uniform_noise(-epsilon, +epsilon)`
- **Categories**:
  - **Desired points**: Near integer 2 (should map to target 1)
  - **Excluding points**: Near other integers (should map to target 0)

### 2. Train Chebyshev Approximation
- **Domain**: [0,8] or [-1,1] depending on `USE_RESCALED`
- **Degree**: Configurable via `CHEB_DEGREE` (default: 119)
- **Method**: Discrete cosine transform approach with Clenshaw evaluation

### 3. Evaluate & Measure Error
- **Point Error**: |Chebyshev_output - expected_target|
- **Green Error**: Error for points that should equal 1 
- **Red Error**: Error for points that should equal 0
- **Metrics**: Max error, RMS error, separate tracking by category

## Epsilon Variation Testing
Test how approximation accuracy changes with input noise:

### Single Epsilon Mode
```python
SINGLE_EPSILON = 0.02  # Test just this epsilon value
```

### Range Mode  
```python
SINGLE_EPSILON = None   # Enable range mode
MIN_EPSILON = 0.001     # Start of range
MAX_EPSILON = 0.05      # End of range  
NUM_EPSILON_VALUES = 20 # Number of values to test
```

## Output Files
Results are saved to `graphs/{DATE_FOLDER}/{FUNCTION_TYPE}/`:

- **Function plots**: Visual representation of the indicator function
- **Error plots**: Epsilon vs error curves (3 types: desired, excluding, combined)
- **CSV logs**: Detailed numerical results for each epsilon value
- **Summary files**: Aggregated statistics

## Verification
Run `python verify_tests.py` to verify the system works correctly:

- ✓ **Rescaling**: Confirms [0,8] ↔ [-1,1] mapping
- ✓ **Functions**: Verifies different functions produce different outputs  
- ✓ **Epsilon**: Confirms epsilon affects point generation
- ✓ **Domains**: Tests rescaled vs original domain consistency

## File Structure

### Core System Files
- **`main.py`** - Main application entry point that orchestrates the complete analysis workflow
- **`config.py`** - Centralized configuration file containing all adjustable parameters for the system
- **`utils.py`** - Common utility functions including statistics, file handling, and ID generation

### Mathematical Components  
- **`indicator_functions.py`** - Implementation of various indicator functions (impulse, plateau_sine, plateau_reg, plateau_sine_impulse)
- **`chebyshev_approximation.py`** - Chebyshev polynomial coefficient computation and evaluation using optimized NumPy implementation
- **`data_generation.py`** - Generation of approximate integer test data with domain rescaling capabilities

### Analysis & Visualization
- **`error_analysis.py`** - Comprehensive error calculation and analysis between polynomial approximations and true function values
- **`create_graphs.py`** - All plotting and visualization functionality including function plots, error analysis, and comparison charts
- **`compare_runs.py`** - Cross-run comparison utilities and batch analysis tools

### Output Structure
- **`graphs/`** - Organized output directory structure:
  - `{DATE_FOLDER}/` - Date-based run organization (e.g., `aug21_deg119/`)
    - `{FUNCTION_TYPE}/` - Function-specific results (impulse, plateau_sine, plateau_reg)
      - `*_function_approximation_*.png` - Function vs approximation comparison
      - `*_function_*_errors_*.png` - Function-based error analysis plots  
      - `*_chebyshev_*_errors_*.png` - Chebyshev-based error analysis plots
      - `*_detailed_results_*.csv` - Raw numerical results data
      - `analysis_summary_*.txt` - Statistical analysis summaries
    - `comparisons/` - Cross-function comparison plots
    - `verification/` - System verification and validation plots
- **`archive/`** - Historical versions and experimental code

## Dependencies
- Python 3.7+
- NumPy - Mathematical operations and polynomial computations  
- Matplotlib - Plotting and visualization
- Pathlib - File system operations

## Usage Examples

### Basic Single Run
```python
# Configure in config.py:
FUNCTION_TYPE = "impulse"
CHEB_DEGREE = 119
USE_RESCALED = True

# Run analysis
python main.py
```

### Parameter Sweep
```python  
# Configure in config.py:
MIN_EPSILON = 0.001
MAX_EPSILON = 0.45
NUM_EPSILON_VALUES = 50

# Results automatically saved to graphs/{DATE_FOLDER}/
```

### Function Comparison
```python
# Run multiple function types with compare_runs.py
# Automatically generates comparison visualizations
```

## Key Findings
1. **Rescaling works correctly**: Both domains produce consistent results
2. **Functions are distinct**: Each function type has unique characteristics
3. **Epsilon sensitivity**: Larger epsilon → larger point deviations as expected
4. **Constant errors**: High-degree polynomials (119) may saturate accuracy

