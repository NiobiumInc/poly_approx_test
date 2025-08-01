# poly_approx_test

## Goal
Test how well Chebyshev polynomial approximations of indicator functions perform when evaluated on "approximate integer" points with varying epsilon values.

## Indicator Function Definition
- **Desired value**: 2
- **Indicator function should return**:
  - `1` for points in range [1.5, 2.5] (desired_value ± 0.5)  
  - `0` for points outside this range

## Test Methodology
1. **Generate Test Points**:
   - "Approximate integers": Points within epsilon of true integers 0,1,2,3,4,5,6,7,8
   - Split into two categories:
     - **Excluding points**: Near integers ≠ 2 (should map to target 0)
     - **Desired points**: Near integer 2 (should map to target 1)

2. **Train Chebyshev Approximation**:
   - Train on indicator function over domain [0,8]
   - Use degree N (currently testing different values)

3. **Evaluate & Measure Error**:
   - Evaluate Chebyshev approximation on test points
   - **Green error**: |Chebyshev_output - 1| for points that should = 1
   - **Red error**: |Chebyshev_output - 0| for points that should = 0

