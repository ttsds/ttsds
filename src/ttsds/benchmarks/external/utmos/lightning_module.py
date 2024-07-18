import os
from pathlib import Path

import requests
import numpy as np
import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn

from ttsds.benchmarks.external.utmos.model import (
    load_ssl_model,
    PhonemeEncoder,
    DomainEmbedding,
    LDConditioner,
    Projection,
)
from ttsds.util.cache import CACHE_DIR


class BaselineLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.construct_model()
        self.save_hyperparameters()

    def construct_model(self):
        wav2vec_path = Path(CACHE_DIR) / "wav2vec_small.pt"
        url = "https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/resolve/main/wav2vec_small.pt?download=true"
        if not wav2vec_path.exists():
            print("Downloading wav2vec_small.pt...")
            response = requests.get(url)
            with open(wav2vec_path, "wb") as f:
                f.write(response.content)
        wav2vec_path = str(wav2vec_path)
        self.feature_extractors = nn.ModuleList(
            [
                load_ssl_model(cp_path=wav2vec_path),
                DomainEmbedding(3, 128),
            ]
        )
        output_dim = sum(
            [
                feature_extractor.get_output_dim()
                for feature_extractor in self.feature_extractors
            ]
        )
        output_layers = [
            LDConditioner(judge_dim=128, num_judges=3000, input_dim=output_dim)
        ]
        output_dim = output_layers[-1].get_output_dim()
        output_layers.append(
            Projection(
                hidden_dim=2048,
                activation=torch.nn.ReLU(),
                range_clipping=False,
                input_dim=output_dim,
            )
        )

        self.output_layers = nn.ModuleList(output_layers)

    def forward(self, inputs):
        outputs = {}
        for feature_extractor in self.feature_extractors:
            outputs.update(feature_extractor(inputs))
        x = outputs
        for output_layer in self.output_layers:
            x = output_layer(x, inputs)
        return x
