from pathlib import Path
import tarfile
import importlib.resources
from typing import List, Union
from argparse import ArgumentParser, BooleanOptionalAction

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from scipy.stats import hmean
import math

from ttsds import BenchmarkSuite, BENCHMARKS_V1, BENCHMARKS_V2
from ttsds.util.dataset import DirectoryDataset, TarDataset, WavListDataset

args = ArgumentParser()
args.add_argument("--test-dataset-dir", type=str, default="parler_tts_large")
args.add_argument("--reference-dataset-dir", type=str, default="speech_libritts_r_test")
args.add_argument("--benchmark-filter", type=str, default="all")
args.add_argument("--output", type=str, default="results")
args.add_argument("--device", type=str, default="cpu")
args.add_argument("--multiprocessing", action=BooleanOptionalAction)
args.add_argument("--no-text", action=BooleanOptionalAction)

noise_datasets = sorted(list(Path("noise-reference").rglob("*.tar.gz")))
noise_datasets = [TarDataset(x, text_suffix=".txt") for x in noise_datasets]


if __name__ == "__main__":
    args = args.parse_args()

    if args.no_text:
        test_datasets = [DirectoryDataset(args.test_dataset_dir, has_text=False)]
        reference_datasets = [
            DirectoryDataset(args.reference_dataset_dir, has_text=False)
        ]
    else:
        test_datasets = [DirectoryDataset(args.test_dataset_dir, text_suffix=".txt")]
        reference_datasets = [
            DirectoryDataset(args.reference_dataset_dir, text_suffix=".txt")
        ]

    if args.benchmark_filter == "all":
        BENCHMARKS = dict(BENCHMARKS_V1, **BENCHMARKS_V2)
    else:
        BENCHMARKS = {
            k: v
            for k, v in dict(BENCHMARKS_V1, **BENCHMARKS_V2).items()
            if args.benchmark_filter in k
        }

    if args.no_text:
        BENCHMARKS = {k: v for k, v in BENCHMARKS.items() if "wer" not in k.lower()}

    print("Initializing benchmark suite...", BENCHMARKS)

    ttsds = BenchmarkSuite(
        test_datasets,
        noise_datasets,
        reference_datasets,
        write_to_file=f"{args.output}_{args.benchmark_filter}.csv",
        benchmarks=BENCHMARKS,
        multiprocessing=args.multiprocessing,
        device=args.device,
    )

    print("Running benchmarks...")
    ttsds.run()
