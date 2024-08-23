# ttsds

[![PyPI - Version](https://img.shields.io/pypi/v/ttsds.svg)](https://pypi.org/project/ttsds)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ttsds.svg)](https://pypi.org/project/ttsds)
[![Hugginface Space](https://img.shields.io/badge/%F0%9F%A4%97-ttsds%2Fbenchmark-blue)](https://huggingface.co/spaces/ttsds/benchmark)

## Installation

### Requirements

- Python 3.8+
- System packages: ffmpeg, automake, autoconf, unzip, sox, gfortran, subversion, libtool
- Simple_hifigan, wvmos and wespeaker are not available on PyPi, so you need to install them manually.
    - https://github.com/wenet-e2e/wespeaker
    - https://github.com/AndreevP/wvmos
    - https://github.com/MiniXC/simple_hifigan
- On some systems, the fairseq installation may fail due to conflicting dependencies. In this case, you can install this fork of fairseq https://github.com/MiniXC/fairseq-noconf

### Pip

```console
pip install ttsds
```

### Caching

Please set ``TTSDS_CACHE_DIR`` environment variable to a directory where you want to cache the downloaded models and data.

[![Website](https://ttsdsbenchmark.com/logo-dark.png)](https://ttsdsbenchmark.com)

## License

`ttsds` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
