# ttsds

[![PyPI - Version](https://img.shields.io/pypi/v/ttsds.svg)](https://pypi.org/project/ttsds)
[![Hugginface Space](https://img.shields.io/badge/%F0%9F%A4%97-ttsds%2Fbenchmark-blue)](https://huggingface.co/spaces/ttsds/benchmark)

As many recent Text-to-Speech (TTS) models have shown, synthetic audio can be close to real human speech. However, traditional evaluation methods for TTS systems need an update to keep pace with these new developments. Our TTSDS benchmark assesses the quality of synthetic speech by considering factors like prosody, speaker identity, and intelligibility. By comparing these factors with both real speech and noise datasets, we can better understand how close synthetic speech is to human speech.

For the current benchmark results, see https://huggingface.co/spaces/ttsds/benchmark.

For other details, see our paper: https://arxiv.org/abs/2407.12707

## Installation

### Pip

```console
pip install ttsds
```

### Requirements

- Python 3.8+
- System packages: ffmpeg, automake, autoconf, unzip, sox, gfortran, subversion, libtool
- On some systems, the fairseq installation may fail due to conflicting dependencies. In this case, you can install this fork of fairseq https://github.com/MiniXC/fairseq-noconf

### Caching

Please set ``TTSDS_CACHE_DIR`` environment variable to a directory where you want to cache the downloaded models and data.

[![Website](https://ttsdsbenchmark.com/logo-dark.png)](https://ttsdsbenchmark.com)

## License

`ttsds` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Citation
```bibtex
@misc{minixhofer2024ttsdstexttospeechdistribution,
      title={TTSDS -- Text-to-Speech Distribution Score}, 
      author={Christoph Minixhofer and Ondřej Klejch and Peter Bell},
      year={2024},
      eprint={2407.12707},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2407.12707}, 
}
```
