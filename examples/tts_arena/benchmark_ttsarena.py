from pathlib import Path
import tarfile
import importlib.resources

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from scipy.stats import hmean
import math

from ttsdb import BenchmarkSuite
from ttsdb.util.dataset import DirectoryDataset, TarDataset
from ttsdb.benchmarks.external.pesq import PESQBenchmark
from ttsdb.benchmarks.external.wv_mos import WVMOSBenchmark
from ttsdb.benchmarks.external.utmos import UTMOSBenchmark
from ttsdb.benchmarks.benchmark import Benchmark

datasets = sorted(list(Path("data").rglob("*.tar.gz")))
datasets = [TarDataset(x) for x in datasets]

benchmarks = [
    "hubert",
    "wav2vec2",
    "wavlm",
    "wav2vec2_wer",
    "whisper_wer",
    "mpm",
    "pitch",
    "wespeaker",
    "dvector",
    "hubert_token",
    "voicefixer",
    "wada_snr",
]

benchmark_suite = BenchmarkSuite(
    datasets,
    benchmarks=benchmarks,
    write_to_file="results.csv",
)

benchmark_suite.run()
df = benchmark_suite.get_aggregated_results()

datasets = sorted(datasets, key=lambda x: x.name)


def run_external_benchmark(benchmark: Benchmark, datasets: list):
    if Path(f"csv/{benchmark.name.lower()}.csv").exists():
        return pd.read_csv(f"csv/{benchmark.name.lower()}.csv")
    df = pd.DataFrame()
    names = []
    scores = []
    for d in datasets:
        score = np.mean(benchmark._get_distribution(d))
        names.append(d.name)
        scores.append(score)
    df["dataset"] = names
    df["score"] = scores
    df.to_csv(f"csv/{benchmark.name.lower()}.csv", index=False)
    return df


wvmos_df = run_external_benchmark(WVMOSBenchmark(), datasets)
wvmos_df["benchmark_category"] = "wvmos"
utmos_df = run_external_benchmark(UTMOSBenchmark(), datasets)
utmos_df["benchmark_category"] = "utmos"

gt_score_df = pd.read_csv("csv/gt_score.csv")
gt_score_df["benchmark_category"] = "gt_score"
# normalize the scores
gt_score_df["score"] = (gt_score_df["score"] - gt_score_df["score"].min()) / (
    gt_score_df["score"].max() - gt_score_df["score"].min()
)
gt_score_df["score"] = np.log10(gt_score_df["score"] + 1)
gt_score_df["score"] = (gt_score_df["score"] - gt_score_df["score"].min()) / (
    gt_score_df["score"].max() - gt_score_df["score"].min()
)

# print systems ordered by score
print(gt_score_df.sort_values("score"))

# merge the dataframes
df["benchmark_type"] = "ttsdb"
wvmos_df["benchmark_type"] = "external"
utmos_df["benchmark_type"] = "external"
gt_score_df["benchmark_type"] = "mos"
df = pd.concat([df, wvmos_df, utmos_df, gt_score_df])

# compute the correlations
corrs = []

# compute the correlations with statsmodels
X = df[df["benchmark_type"] == "ttsdb"]
X = X.pivot(index="dataset", columns="benchmark_category", values="score")
X = X.sort_values("dataset")
X = X.reset_index()

# apply to all columns except dataset
x_ds = X["dataset"]
X["dataset"] = x_ds

# print systems ordered by harmonic mean
X_mean = X
# mean of all columns except dataset
X_mean["mean"] = X_mean.drop("dataset", axis=1).apply(np.mean, axis=1)
print(X_mean.sort_values("mean"))

X = X.drop("dataset", axis=1)

y = df[df["benchmark_category"] == "gt_score"]
# remove parlertts and vokan
y = y.sort_values("dataset")
y = y.reset_index()
y = y["score"]


X_mean = X.apply(np.mean, axis=1)
# min_max normalize
X_mean = (X_mean - X_mean.min()) / (X_mean.max() - X_mean.min())
# get correlation with mean
print(y.shape, X_mean.shape)
corr, p = spearmanr(y, X_mean)
# print systems ordered by mean
print(f"mean: {corr:.3f} ({p:.3f})")

for b in df["benchmark_category"].unique():
    bdf = df[df["benchmark_category"] == b]
    mosdf = df[df["benchmark_category"] == "gt_score"]
    hmean_score = X_mean
    # sort both dataframes by dataset name
    mosdf = mosdf.sort_values("dataset")
    bdf = bdf.sort_values("dataset")
    assert (mosdf["dataset"].values == bdf["dataset"].values).all()
    if b == "gt_score":
        continue
    bdf_score = bdf["score"]
    bdf["score"] = (bdf_score - bdf_score.min()) / (bdf_score.max() - bdf_score.min())
    corr, p = spearmanr(mosdf["score"], bdf["score"])
    corrs.append((b, corr, p))
    print(f"{b}: {corr:.3f} ({p:.3f})")
    # get correlation with mean
    hmean_corr, hmean_p = spearmanr(bdf["score"], hmean_score)
    # print(f"{b} mean: {hmean_corr:.3f} ({hmean_p:.3f})")
