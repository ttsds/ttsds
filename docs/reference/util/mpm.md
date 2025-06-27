---
layout: default
title: MPM Utilities
---

# MPM Utilities

This module provides functions related to the McLeod Pitch Method (MPM), which is used for pitch detection.

## Functions

```python
from ttsds.util.mpm import pitch_mpm
```

### `pitch_mpm(waveform, sample_rate)`

Extracts pitch using the McLeod Pitch Method.

```python
import numpy as np
from ttsds.util.mpm import pitch_mpm

# Example waveform (1 second of audio at 16kHz)
waveform = np.random.randn(16000)
sample_rate = 16000

times, pitches = pitch_mpm(waveform, sample_rate)
# Returns: arrays of time points and corresponding pitch values
```

## Algorithm Details

The McLeod Pitch Method is a time-domain pitch detection algorithm that works by:

1. Computing the normalized square difference function (NSDF)
2. Finding positive zero-crossings in the NSDF
3. Finding the highest peak among these zero-crossings
4. Estimating the period based on the position of this peak
5. Converting the period to frequency

This method is particularly effective for speech and musical sounds, providing accurate pitch estimation even in the presence of noise. 