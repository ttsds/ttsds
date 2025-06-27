---
layout: default
title: Measures Utilities
---

# Measures Utilities

This module provides functions for calculating various audio measurements used in benchmarks.

## Functions

### Prosody Measures

```python
from ttsds.util.measures import pitch_measures, energy_measures, duration_measures
```

#### `pitch_measures(waveform, sample_rate)`

Extracts pitch-related measures from an audio waveform.

```python
import numpy as np
from ttsds.util.measures import pitch_measures

# Example waveform (1 second of audio at 16kHz)
waveform = np.random.randn(16000)
sample_rate = 16000

measures = pitch_measures(waveform, sample_rate)
# Returns: mean, std, min, max, range of pitch
```

#### `energy_measures(waveform, sample_rate)`

Extracts energy-related measures from an audio waveform.

```python
from ttsds.util.measures import energy_measures

measures = energy_measures(waveform, sample_rate)
# Returns: mean, std, min, max, range of energy
```

#### `duration_measures(phoneme_durations)`

Calculates statistics on phoneme durations.

```python
from ttsds.util.measures import duration_measures

phoneme_durations = [0.05, 0.1, 0.2, 0.15, 0.08]
measures = duration_measures(phoneme_durations)
# Returns: mean, std, min, max, range of durations
```

### Spectral Measures

```python
from ttsds.util.measures import spectral_centroid, spectral_bandwidth, spectral_rolloff
```

#### `spectral_centroid(waveform, sample_rate)`

Calculates the spectral centroid of an audio waveform.

```python
from ttsds.util.measures import spectral_centroid

centroid = spectral_centroid(waveform, sample_rate)
```

#### `spectral_bandwidth(waveform, sample_rate)`

Calculates the spectral bandwidth of an audio waveform.

```python
from ttsds.util.measures import spectral_bandwidth

bandwidth = spectral_bandwidth(waveform, sample_rate)
```

#### `spectral_rolloff(waveform, sample_rate, percentile=0.85)`

Calculates the spectral rolloff point of an audio waveform.

```python
from ttsds.util.measures import spectral_rolloff

rolloff = spectral_rolloff(waveform, sample_rate)
``` 