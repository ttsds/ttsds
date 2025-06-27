---
layout: default
title: Citation
---

# Citation

If you use TTSDS in your research, please cite our paper:

```bibtex
@inproceedings{minixhofer2024ttsds,
  title={TTSDS-Text-to-Speech Distribution Score},
  author={Minixhofer, Christoph and Klejch, Ond{\v{r}}ej and Bell, Peter},
  booktitle={SLT},
  year={2024},
}
```

## Paper Abstract

> Automatic evaluation of synthetic speech is becoming increasingly important as text-to-speech (TTS) systems continue to improve in quality and become more widely deployed. Traditional approaches like Mean Opinion Score (MOS) and its variants rely on human evaluation, which is time-consuming, expensive, and not easily reproducible. Existing automatic metrics often focus on specific aspects of speech quality or rely on reference audio, limiting their applicability. In this paper, we introduce TTSDS, a comprehensive evaluation framework that assesses synthetic speech quality by measuring distributional similarity to real speech across multiple dimensions. TTSDS evaluates prosody, speaker identity, and intelligibility, providing a holistic view of TTS performance without requiring parallel data. We validate our approach through correlation analysis with human judgments and demonstrate that TTSDS offers a more nuanced and complete evaluation than existing metrics. Our framework is open-source and designed to be easily extensible, enabling researchers and developers to better understand and improve their TTS systems.

## Additional References

If you're using specific components of TTSDS, you might also want to cite the following papers that influenced our work:

### Speaker Verification

```bibtex
@article{chen2022wavlm,
  title={WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing},
  author={Chen, Sanyuan and Wang, Chengyi and Chen, Zhengyang and Wu, Yu and Liu, Shujie and Chen, Zhuo and Li, Jinyu and Kanda, Naoyuki and Yoshioka, Takuya and Xiao, Xiong and others},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  volume={16},
  number={6},
  pages={1505--1518},
  year={2022},
  publisher={IEEE}
}
```

### Speech Recognition

```bibtex
@article{baevski2020wav2vec,
  title={wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations},
  author={Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={12449--12460},
  year={2020}
}
```

### Prosody Analysis

```bibtex
@inproceedings{wagner2017speech,
  title={Speech Rhythm Analysis with Decomposition of the Amplitude Envelope: Characterizing Rhythmic Patterns within and across Languages},
  author={Wagner, Petra and Windmann, Andreas},
  booktitle={Journal of the Acoustical Society of America},
  volume={142},
  number={3},
  pages={1159--1169},
  year={2017}
}
```

## Related Projects

If you're interested in TTS evaluation, you might also want to explore these related projects:

1. **NISQA**: Non-Intrusive Speech Quality Assessment
   ```bibtex
   @inproceedings{mittag2021nisqa,
     title={NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced MOS Annotations},
     author={Mittag, Gabriel and Naderi, Babak and Chehadi, Ahmad and M{\"o}ller, Sebastian},
     booktitle={INTERSPEECH},
     pages={2127--2131},
     year={2021}
   }
   ```

2. **MOSNet**: Mean Opinion Score Prediction Network
   ```bibtex
   @inproceedings{lo2019mosnet,
     title={MOSNet: Deep Learning based Objective Assessment for Voice Conversion},
     author={Lo, Chen-Chou and Fu, Szu-Wei and Huang, Wen-Chin and Wang, Xin and Yamagishi, Junichi and Tsao, Yu and Wang, Hsin-Min},
     booktitle={INTERSPEECH},
     pages={1541--1545},
     year={2019}
   }
   ```

3. **UTMOS**: Universal Text-to-speech MOS Prediction
   ```bibtex
   @inproceedings{saeki2022utmos,
     title={UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022},
     author={Saeki, Takaaki and Ando, Atsunori and Ueno, Ryo and Xin, Yuki and Takamichi, Shinnosuke and Yamagishi, Junichi and Mochizuki, Hideyuki},
     booktitle={Proc. Interspeech 2022},
     pages={973--977},
     year={2022}
   }
   ```

## Leaderboard

For the latest results and benchmarks using TTSDS, please visit our [leaderboard](https://ttsdsbenchmark.com). 