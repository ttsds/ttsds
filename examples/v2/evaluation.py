from pathlib import Path
import tarfile
import importlib.resources
from typing import List, Union
from argparse import ArgumentParser

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from scipy.stats import hmean
import math

from ttsds import BenchmarkSuite, BENCHMARKS_V1, BENCHMARKS_V2
from ttsds.util.dataset import DirectoryDataset, TarDataset, WavListDataset

args = ArgumentParser()
args.add_argument("--dataset", type=str, default="myst")
args.add_argument("--benchmark-filter", type=str, default="allosaurus")
args.add_argument("--output", type=str, default="results")

noise_datasets = sorted(list(Path("noise-reference").rglob("*.tar.gz")))
noise_datasets = [TarDataset(x, text_suffix=".txt") for x in noise_datasets]

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

if __name__ == "__main__":
    args = args.parse_args()
    if "torgo" in args.dataset:
        dataset = "torgo"
        if "ctr" in args.dataset:
            reference_datasets = [
                DirectoryDataset(f"v2-evaluation/{dataset}/A/ctr", text_suffix=".txt", name="Torgo Reference"),
            ]
            test_datasets = [
                DirectoryDataset(f"v2-evaluation/{dataset}/B/ctr", text_suffix=".txt", name="Torgo Test"),
            ]
            for directory in Path("v2-evaluation/tts").iterdir():
                for subdirectory in directory.iterdir():
                    if subdirectory.is_dir() and args.dataset in subdirectory.name:
                        ctr_wavs = [
                            file for file in subdirectory.rglob("*.wav") if int(file.name[:3]) >= 54
                        ]
                        ctr_txts = [
                            file.with_suffix(".txt") for file in ctr_wavs
                        ]
                        test_datasets.append(WavListDataset(ctr_wavs, ctr_txts, name="Torgo CTR"))
        elif "dys" in args.dataset:
            reference_datasets = [
                DirectoryDataset(f"v2-evaluation/{dataset}/A/dys", text_suffix=".txt", name="Torgo Reference"),
            ]
            test_datasets = [
                DirectoryDataset(f"v2-evaluation/{dataset}/B/dys", text_suffix=".txt", name="Torgo Test"),
            ]
            for directory in Path("v2-evaluation/tts").iterdir():
                for subdirectory in directory.iterdir():
                    if subdirectory.is_dir() and args.dataset in subdirectory.name:
                        dys_wavs = [
                            file for file in subdirectory.rglob("*.wav") if int(file.name[:3]) < 54
                        ]
                        dys_txts = [
                            file.with_suffix(".txt") for file in dys_wavs
                        ]
                        test_datasets.append(WavListDataset(dys_wavs, dys_txts, name="Torgo DYS"))
    else:
        reference_datasets = [
            DirectoryDataset(f"v2-evaluation/{args.dataset}/A", text_suffix=".txt", name=f"{args.dataset} Reference"),
        ]
        test_datasets = [
            DirectoryDataset(f"v2-evaluation/{args.dataset}/B", text_suffix=".txt", name=f"{args.dataset} Test"),
        ]
        test_datasets.extend(get_datasets(args.dataset))

    BENCHMARKS = {
        k: v for k, v in dict(BENCHMARKS_V1, **BENCHMARKS_V2).items() if args.benchmark_filter in k
    }

    print("Initializing benchmark suite...", BENCHMARKS)


    ttsds = BenchmarkSuite(
        test_datasets,
        noise_datasets,
        reference_datasets,
        write_to_file=f"{args.output}_{args.dataset}_{args.benchmark_filter}.csv",
        benchmarks=BENCHMARKS,
        multiprocessing=True,
    )

    print("Running benchmarks...")
    ttsds.run()
