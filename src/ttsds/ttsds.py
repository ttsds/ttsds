from typing import List, Optional, Dict
import pickle
import hashlib
import os
import warnings
from subprocess import run
from pathlib import Path
from time import time
from sklearn.exceptions import ConvergenceWarning

# Suppress specific (annoying) warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pyannote.audio.core.inference"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="huggingface_hub.file_download"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="transformers.configuration_utils"
)
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.base")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torch.nn.utils.weight_norm"
)

import pandas as pd
from transformers import logging
import numpy as np
from multiprocessing import cpu_count
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout

from ttsds.benchmarks.benchmark import DeviceSupport
from ttsds.benchmarks.environment.voicerestore import VoiceRestoreBenchmark
from ttsds.benchmarks.environment.wada_snr import WadaSNRBenchmark
from ttsds.benchmarks.general.hubert import HubertBenchmark
from ttsds.benchmarks.general.wav2vec2 import Wav2Vec2Benchmark
from ttsds.benchmarks.general.wavlm import WavLMBenchmark
from ttsds.benchmarks.intelligibility.w2v2_activations import (
    Wav2Vec2ActivationsBenchmark,
)
from ttsds.benchmarks.intelligibility.whisper_activations import (
    WhisperActivationsBenchmark,
)
from ttsds.benchmarks.prosody.mpm import MPMBenchmark
from ttsds.benchmarks.prosody.pitch import PitchBenchmark
from ttsds.benchmarks.prosody.hubert_token import (
    HubertTokenSRBenchmark,
)
from ttsds.benchmarks.prosody.allosaurus import (
    AllosaurusSRBenchmark,
)
from ttsds.benchmarks.speaker.wespeaker import WeSpeakerBenchmark
from ttsds.benchmarks.speaker.dvector import DVectorBenchmark
from ttsds.benchmarks.benchmark import BenchmarkCategory
from ttsds.util.dataset import Dataset, TarDataset
from ttsds.util.cache import CACHE_DIR

# we do this to avoid "some weights of the model checkpoint at ... were not used when initializing" warnings
logging.set_verbosity_error()

BENCHMARKS = {
    # generic
    "hubert": HubertBenchmark,
    "wav2vec2": Wav2Vec2Benchmark,
    "wavlm": WavLMBenchmark,
    # prosody
    "mpm": MPMBenchmark,
    "pitch": PitchBenchmark,
    "hubert_token_sr": HubertTokenSRBenchmark,
    "allosaurus_sr": AllosaurusSRBenchmark,
    # speaker
    "wespeaker": WeSpeakerBenchmark,
    "dvector": DVectorBenchmark,
    # environment
    "voicerestore": VoiceRestoreBenchmark,
    "wada_snr": WadaSNRBenchmark,
    # intelligibility
    "wav2vec2_activations": Wav2Vec2ActivationsBenchmark,
    "whisper_activations": WhisperActivationsBenchmark,
}


# save https://huggingface.co/datasets/ttsds/noise-reference to cache_dir/noise-reference
if not (CACHE_DIR / "noise-reference").exists():
    run(
        [
            "git",
            "clone",
            "https://huggingface.co/datasets/ttsds/noise-reference",
            str(CACHE_DIR / "noise-reference"),
        ],
        check=True,
    )

tar_files = Path(CACHE_DIR / "noise-reference").glob("*.tar.gz")

NOISE_DATASETS = [
    TarDataset(
        str(CACHE_DIR / "noise-reference" / "noise_all_ones.tar.gz"),
        name="noise_all_ones",
    ),
    TarDataset(
        str(CACHE_DIR / "noise-reference" / "noise_all_zeros.tar.gz"),
        name="noise_all_zeros",
    ),
    TarDataset(
        str(CACHE_DIR / "noise-reference" / "noise_normal_distribution.tar.gz"),
        name="noise_normal_distribution",
    ),
    TarDataset(
        str(CACHE_DIR / "noise-reference" / "noise_uniform_distribution.tar.gz"),
        name="noise_uniform_distribution",
    ),
]


class BenchmarkSuite:

    def __init__(
        self,
        datasets: List[Dataset],
        reference_datasets: List[Dataset],
        noise_datasets: List[Dataset] = NOISE_DATASETS,
        benchmarks: Dict[str, "Benchmark"] = BENCHMARKS,
        category_weights: Dict[BenchmarkCategory, float] = {
            BenchmarkCategory.SPEAKER: 0.25,
            BenchmarkCategory.INTELLIGIBILITY: 0.25,
            BenchmarkCategory.PROSODY: 0.25,
            BenchmarkCategory.GENERIC: 0.25,
            BenchmarkCategory.ENVIRONMENT: 0.0,
        },
        skip_errors: bool = False,
        write_to_file: str = None,
        benchmark_kwargs: dict = {},
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        include_environment: bool = False,
    ):
        if (
            "hubert_token" not in benchmark_kwargs
            or "cluster_datasets" not in benchmark_kwargs["hubert_token"]
        ):
            if "hubert_token" not in benchmark_kwargs:
                benchmark_kwargs["hubert_token"] = {}
            benchmark_kwargs["hubert_token"]["cluster_datasets"] = [
                reference_datasets[0].sample(min(100, len(reference_datasets[0])))
            ]
        if (
            "hubert_token_sr" not in benchmark_kwargs
            or "cluster_datasets" not in benchmark_kwargs["hubert_token_sr"]
        ):
            if "hubert_token_sr" not in benchmark_kwargs:
                benchmark_kwargs["hubert_token_sr"] = {}
            benchmark_kwargs["hubert_token_sr"]["cluster_datasets"] = [
                reference_datasets[0].sample(min(100, len(reference_datasets[0])))
            ]
        self.benchmarks = {
            k: v(**benchmark_kwargs.get(k, {})) for k, v in benchmarks.items()
        }
        if not include_environment:
            self.benchmarks = {
                k: v
                for k, v in self.benchmarks.items()
                if v.category != BenchmarkCategory.ENVIRONMENT
            }
        # if gpu is available, move the benchmarks that support it to the gpu
        for benchmark in self.benchmarks.values():
            if DeviceSupport.GPU in benchmark.supported_devices and device == "cuda":
                benchmark.to_device(device)
        self.datasets = datasets
        self.datasets = sorted(self.datasets, key=lambda x: x.name)
        self.database = pd.DataFrame(
            columns=[
                "benchmark_name",
                "benchmark_category",
                "dataset",
                "score",
                "time_taken",
                "noise_dataset",
                "reference_dataset",
            ]
        )
        self.category_weights = category_weights
        self.skip_errors = skip_errors
        self.noise_datasets = noise_datasets
        self.reference_datasets = reference_datasets
        self.cache_dir = cache_dir or os.getenv(
            "TTSDS_CACHE_DIR", os.path.expanduser("~/.ttsds_cache")
        )
        self.write_to_file = write_to_file
        os.makedirs(self.cache_dir, exist_ok=True)
        self._setup_caching()
        self._progress = None
        self._progress_table = None
        self._main_task = None
        self._live = None
        self._layout = None
        if self.write_to_file is not None and Path(self.write_to_file).exists():
            self.database = pd.read_csv(self.write_to_file)

    def _setup_caching(self):
        """Setup caching for benchmark distributions."""
        self.distribution_cache = {}
        self.cache_file = os.path.join(self.cache_dir, "distribution_cache.pkl")
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self.distribution_cache = pickle.load(f)
            except Exception:
                self.distribution_cache = {}

    def _get_cache_key(self, benchmark: "Benchmark", dataset: Dataset) -> str:
        """Generate a unique cache key for a benchmark and dataset combination."""
        key_data = f"{benchmark.name}_{dataset.name}_{dataset.root_dir}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_distribution(self, benchmark: "Benchmark", dataset: Dataset) -> np.ndarray:
        """Get distribution with caching."""
        cache_key = self._get_cache_key(benchmark, dataset)
        if cache_key in self.distribution_cache:
            return self.distribution_cache[cache_key]

        distribution = benchmark.get_distribution(dataset)
        self.distribution_cache[cache_key] = distribution

        # Save cache to disk
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.distribution_cache, f)

        return distribution

    def _setup_progress(self, total_tasks: int):
        """Setup the progress display with both table and progress bars."""
        console = Console()

        # Create layout
        self._layout = Layout()
        self._layout.split_column(
            Layout(name="table", size=20), Layout(name="progress", size=3)
        )

        # Create the progress table
        self._progress_table = Table(show_header=True, header_style="bold magenta")
        self._progress_table.add_column("Benchmark", style="cyan")
        self._progress_table.add_column("Category", style="blue")
        self._progress_table.add_column("Completed", style="green")
        self._progress_table.add_column("Total", style="yellow")
        self._progress_table.add_column("Progress", style="magenta")

        # Initialize benchmark progress tracking
        for benchmark in sorted(self.benchmarks.values(), key=lambda x: x.name):
            self._progress_table.add_row(
                benchmark.name,
                benchmark.category.name,
                "0",
                str(total_tasks),
                "[yellow]0%[/yellow]",
            )

        # Create the main progress bar
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        self._main_task = self._progress.add_task(
            "[cyan]Running benchmarks...", total=total_tasks
        )

        # Update layout with components
        self._layout["table"].update(
            Panel(self._progress_table, title="Benchmark Progress", border_style="blue")
        )
        self._layout["progress"].update(self._progress)

        # Start the live display
        self._live = Live(
            self._layout,
            console=console,
            refresh_per_second=4,  # Limit refresh rate to reduce flickering
            vertical_overflow="visible",
        )
        self._live.start()

    def _update_progress(self):
        """Update the progress display based on the current database state."""
        if self._progress and self._progress_table:
            # Update main progress
            self._progress.advance(self._main_task)

            # Rebuild the table from scratch
            new_table = Table(show_header=True, header_style="bold magenta")
            new_table.add_column("Benchmark", style="cyan")
            new_table.add_column("Category", style="blue")
            new_table.add_column("Completed", style="green")
            new_table.add_column("Total", style="yellow")
            new_table.add_column("Progress", style="magenta")

            # Add rows for each benchmark
            for benchmark in sorted(self.benchmarks.values(), key=lambda x: x.name):
                # Count completed datasets for this benchmark
                completed = len(
                    self.database[
                        (self.database["benchmark_name"] == benchmark.name)
                        & (~self.database["score"].isna())
                    ]
                )
                total = len(self.datasets)
                percentage = (completed / total) * 100

                new_table.add_row(
                    benchmark.name,
                    benchmark.category.name,
                    str(completed),
                    str(total),
                    f"[{'green' if percentage == 100 else 'yellow'}]{percentage:.1f}%[/{'green' if percentage == 100 else 'yellow'}]",
                )

            # Update the layout with the new table
            self._layout["table"].update(
                Panel(new_table, title="Benchmark Progress", border_style="blue")
            )
            self._layout["progress"].update(self._progress)

    def _stop_progress(self):
        """Stop and cleanup the progress display."""
        if self._live:
            # Clear the live display
            self._live.stop()
            self._live = None

            # Clear the console to remove the progress display
            console = Console()
            console.clear()

            # Print final status
            completed = len(self.database[~self.database["score"].isna()])
            total = len(self.database)
            console.print(f"[green]Completed {completed}/{total} benchmarks[/green]")

        # Clean up other components
        if self._progress:
            self._progress.stop()
            self._progress = None
        self._progress_table = None
        self._main_task = None
        self._layout = None

    def _run_benchmark(self, benchmark: "Benchmark", dataset: Dataset) -> dict:
        """Run a single benchmark and return its results."""
        console = Console()

        try:
            start = time()
            score = benchmark.compute_score(
                dataset, self.reference_datasets, self.noise_datasets
            )
            time_taken = time() - start

            result = {
                "benchmark_name": [benchmark.name],
                "benchmark_category": [benchmark.category.name],
                "dataset": [dataset.name],
                "score": [score[0]],
                "time_taken": [time_taken],
                "noise_dataset": [score[1][0]],
                "reference_dataset": [score[1][1]],
            }
            return result

        except Exception as e:
            if self.skip_errors:
                console.print(
                    f"[red]Error in {benchmark.name} on {dataset.name}: {str(e)}[/red]"
                )
                return {
                    "benchmark_name": [benchmark.name],
                    "benchmark_category": [benchmark.category.name],
                    "dataset": [dataset.name],
                    "score": [np.nan],
                    "time_taken": [np.nan],
                    "noise_dataset": [""],
                    "reference_dataset": [""],
                }
            else:
                raise e

    def run(self) -> pd.DataFrame:
        console = Console()
        tasks = []
        num_tasks = 0
        for benchmark in sorted(self.benchmarks.values(), key=lambda x: x.name):
            for dataset in self.datasets:
                num_tasks += 1
                if (
                    (self.database["benchmark_name"] == benchmark.name)
                    & (self.database["dataset"] == dataset.name)
                ).any():
                    console.print(
                        f"[yellow]Skipping {benchmark.name} on {dataset.name} as it's already in the database[/yellow]"
                    )
                    continue
                tasks.append((benchmark, dataset))

        # Setup progress tracking
        self._setup_progress(num_tasks)
        self._update_progress()

        if num_tasks > len(tasks):
            for i in range(num_tasks - len(tasks)):
                self._progress.advance(self._main_task)

        if not tasks:
            console.print("[green]All benchmarks have already been run![/green]")
            return self.database

        try:
            # Process tasks sequentially
            for benchmark, dataset in tasks:
                result = self._run_benchmark(benchmark, dataset)
                self._update_database(result)

        finally:
            self._stop_progress()

        return self.database

    def _update_database(self, result: dict) -> None:
        """Helper method to update the database with new results."""
        self.database = pd.concat(
            [self.database, pd.DataFrame(result)],
            ignore_index=True,
        )
        if self.write_to_file is not None:
            self.database["score"] = self.database["score"].astype(float)
            self.database = self.database.sort_values(
                ["benchmark_category", "benchmark_name", "score"],
                ascending=[True, True, False],
            )
            self.database.to_csv(self.write_to_file, index=False)

        # Update progress display after database update
        self._update_progress()

    def aggregate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        def concat_text(x):
            return ", ".join(x)

        # First aggregate the benchmark results
        df_agg = df.groupby(
            [
                "benchmark_category",
                "dataset",
            ]
        ).agg(
            {
                "score": ["mean", "std"],
                "time_taken": ["sum"],
                "noise_dataset": [concat_text],
                "reference_dataset": [concat_text],
                "benchmark_name": [concat_text],
            }
        )

        # Flatten the multiindex columns
        df_agg.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_agg.columns
        ]

        # Reset index to ensure unique indices
        df_agg = df_agg.reset_index()

        # Calculate weighted scores for each dataset
        unique_datasets = df_agg["dataset"].unique()
        overall_data = []

        for dataset in unique_datasets:
            df_dataset = df_agg[df_agg["dataset"] == dataset]
            weighted_score = 0
            total_time = 0

            for category in self.category_weights.keys():
                category_data = df_dataset[
                    df_dataset["benchmark_category"] == category.name
                ]
                if not category_data.empty:
                    weighted_score += (
                        category_data["score_mean"].iloc[0]  # Use mean score
                        * self.category_weights[category]
                    )
                    total_time += category_data["time_taken_sum"].iloc[
                        0
                    ]  # Use sum of time

            overall_data.append(
                {
                    "benchmark_category": "OVERALL",
                    "dataset": dataset,
                    "score_mean": weighted_score,
                    "score_std": 0,  # No standard deviation for overall score
                    "time_taken_sum": total_time,
                    "noise_dataset": None,
                    "reference_dataset": None,
                }
            )

        # Create the overall DataFrame
        df_overall = pd.DataFrame(overall_data)

        # Ensure both DataFrames have the same columns in the same order
        columns = [
            "benchmark_category",
            "dataset",
            "score_mean",
            "score_std",
            "time_taken_sum",
            "noise_dataset",
            "reference_dataset",
        ]

        # Reorder columns and ensure they exist
        for col in columns:
            if col not in df_agg.columns:
                df_agg[col] = None
            if col not in df_overall.columns:
                df_overall[col] = None

        df_agg = df_agg[columns]
        df_overall = df_overall[columns]

        # Concatenate and sort the results
        df_final = pd.concat([df_agg, df_overall], ignore_index=True)
        df_final = df_final.sort_values(
            ["benchmark_category", "score_mean"], ascending=False
        )

        return df_final

    def get_aggregated_results(self) -> pd.DataFrame:
        df = self.database.copy()
        return self.aggregate_df(df)
