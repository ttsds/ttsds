---
layout: default
title: Installation
---

# Installation

This guide will help you install TTSDS and its dependencies.

## System Requirements

Before installing TTSDS, ensure your system has the following requirements:

```bash
# Required system packages
sudo apt-get install ffmpeg automake autoconf espeak unzip sox gfortran subversion libtool
```

For macOS users, you can install these dependencies using Homebrew:

```bash
brew install ffmpeg sox automake autoconf libtool gfortran subversion
```

## Python Installation

TTSDS requires Python 3.7 or higher. You can install it using pip:

```bash
# Basic installation
pip install ttsds
```

## Environment Variables

You can configure TTSDS using environment variables:

```bash
# Set cache directory (default: ~/.cache/ttsds)
export TTSDS_CACHE_DIR=/path/to/cache
```

## Verifying Installation

To verify that TTSDS is installed correctly, run:

```python
import ttsds
print(ttsds.__version__)
```

## Troubleshooting

### Common Issues

## Fairseq Installation

If you encounter dependency conflicts with fairseq, use this fork:

```bash
pip install git+https://github.com/MiniXC/fairseq-noconf
```

#### PyTorch Installation

If you encounter issues with PyTorch installation, you might need to install it separately:

```bash
# For CPU-only version
pip install torch torchvision torchaudio

# For CUDA version (example for CUDA 11.7)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

### Getting Help

If you encounter any issues during installation, please:

1. Check the [GitHub issues](https://github.com/MiniXC/ttsds/issues) to see if your problem has been reported
2. Open a new issue if necessary, providing detailed information about your environment and the error messages

## Next Steps

Once you have successfully installed TTSDS, proceed to the [Quick Start Guide](quickstart.md) to learn how to use it. 