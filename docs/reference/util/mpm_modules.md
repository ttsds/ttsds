---
layout: default
title: MPM Modules
---

# MPM Modules

This module provides implementation modules for the McLeod Pitch Method (MPM), which are used by the MPM pitch detection algorithm.

## Functions

```python
from ttsds.util.mpm_modules import nsdf, peak_picking, parabolic_interpolation
```

### `nsdf(signal)`

Computes the Normalized Square Difference Function of a signal.

```python
import numpy as np
from ttsds.util.mpm_modules import nsdf

# Example signal
signal = np.random.randn(1024)

nsdf_result = nsdf(signal)
# Returns: array of NSDF values
```

### `peak_picking(nsdf_values, threshold=0.1)`

Identifies peaks in the NSDF that exceed a given threshold.

```python
from ttsds.util.mpm_modules import peak_picking

peaks = peak_picking(nsdf_result)
# Returns: list of peak indices
```

### `parabolic_interpolation(array, peak_index)`

Refines a peak location using parabolic interpolation.

```python
from ttsds.util.mpm_modules import parabolic_interpolation

refined_peak = parabolic_interpolation(nsdf_result, peaks[0])
# Returns: refined peak position and value
```

## Implementation Details

These modules implement specific components of the McLeod Pitch Method:

1. **NSDF**: The normalized square difference function measures the similarity of a signal to itself at different time lags, which helps identify periodicity
2. **Peak Picking**: Identifies local maxima in the NSDF that exceed a given threshold
3. **Parabolic Interpolation**: Refines peak locations by fitting a parabola to the peak and its adjacent points

These modules are typically used internally by the main `pitch_mpm` function but can also be used separately for custom pitch detection implementations. 