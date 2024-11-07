from pathlib import Path
import tarfile
import importlib.resources
from typing import List, Union

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from scipy.stats import hmean
import math

from ttsds import BenchmarkSuite
from ttsds.util.dataset import DirectoryDataset, TarDataset

noise_datasets = sorted(list(Path("noise-reference").rglob("*.tar.gz")))
noise_datasets = [TarDataset(x, text_suffix=".txt") for x in noise_datasets]

# reference_datasets = [
#     DirectoryDataset(
#         "v2-evaluation/librilatest", text_suffix=".txt", name="LibriLatest"
#     ),
# ]

# test_datasets = []

# for directory in Path("v2-evaluation/tts").iterdir():
#     if directory.is_dir():
#         test_datasets.append(
#             DirectoryDataset(
#                 directory / "librilatest", text_suffix=".txt", name=directory.name
#             )
#         )

# ttsds = BenchmarkSuite(
#     test_datasets,
#     noise_datasets,
#     reference_datasets,
#     write_to_file="results_librilatest_all.csv",
# )

# ttsds.run()

reference_datasets = [
    DirectoryDataset("v2-evaluation/myst/A", text_suffix=".txt", name="LibriTTSR A"),
]

test_datasets = [
    DirectoryDataset("v2-evaluation/myst/B", text_suffix=".txt", name="LibriTTSR B"),
]

ttsds = BenchmarkSuite(
    test_datasets,
    noise_datasets,
    reference_datasets,
    write_to_file="results_myst_new.csv",
)

ttsds.run()


def get_datasets(split: str) -> List[Union[DirectoryDataset]]:
    """
    Get the datasets for the given split.
    """
    split = split.lower()
    datasets = []
    for directory in Path("v2-evaluation/tts").iterdir():
        for subdirectory in directory.iterdir():
            if subdirectory.is_dir() and subdirectory.name == split:
                datasets.append(
                    DirectoryDataset(
                        subdirectory,
                        text_suffix=".txt",
                        name=directory.name,
                    )
                )
    return datasets


# myst
myst_reference_datasets = [
    DirectoryDataset("v2-evaluation/myst/A", text_suffix=".txt", name="Myst Reference"),
]
myst_test_datasets = [
    DirectoryDataset("v2-evaluation/myst/B", text_suffix=".txt", name="Myst Test"),
]
myst_test_datasets.extend(get_datasets("myst"))

myst_ttsds = BenchmarkSuite(
    myst_test_datasets,
    noise_datasets,
    myst_reference_datasets,
    write_to_file="results_myst_new.csv",
)

myst_ttsds.run()
