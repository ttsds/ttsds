from typing import List, Optional, Dict
import pickle
import hashlib
import os
import warnings
from subprocess import run
from pathlib import Path
from time import time
from sklearn.exceptions import ConvergenceWarning
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

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
from ttsds.benchmarks.generic.hubert import HubertBenchmark
from ttsds.benchmarks.generic.mhubert import MHubert147Benchmark
from ttsds.benchmarks.generic.wav2vec2 import Wav2Vec2Benchmark
from ttsds.benchmarks.generic.wav2vec2_xlsr import Wav2Vec2XLSRBenchmark
from ttsds.benchmarks.generic.wavlm import WavLMBenchmark
from ttsds.benchmarks.intelligibility.w2v2_activations import (
    Wav2Vec2ActivationsBenchmark,
)
from ttsds.benchmarks.intelligibility.w2v2_xlsr_activations import (
    Wav2Vec2XLSRActivationsBenchmark,
)
from ttsds.benchmarks.intelligibility.whisper_activations import (
    WhisperActivationsBenchmark,
)
from ttsds.benchmarks.intelligibility.mwhisper_activations import (
    MWhisperActivationsBenchmark,
)
from ttsds.benchmarks.prosody.mpm import MPMBenchmark
from ttsds.benchmarks.prosody.pitch import PitchBenchmark
from ttsds.benchmarks.prosody.hubert_token import (
    HubertTokenSRBenchmark,
)
from ttsds.benchmarks.prosody.mhubert_token import (
    MHubert147TokenSRBenchmark,
)
from ttsds.benchmarks.prosody.allosaurus import (
    AllosaurusSRBenchmark,
)
from ttsds.benchmarks.speaker.wespeaker import WeSpeakerBenchmark
from ttsds.benchmarks.speaker.dvector import DVectorBenchmark
from ttsds.benchmarks.benchmark import BenchmarkCategory
from ttsds.util.dataset import Dataset, TarDataset
from ttsds.util.cache import CACHE_DIR
from ttsds.util.parallel_distances import DistanceCalculator

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

BENCHMARKS_ML = {
    # generic
    "mhubert": MHubert147Benchmark,
    "wav2vec2_xlsr": Wav2Vec2XLSRBenchmark,
    "wavlm": WavLMBenchmark,
    # prosody
    "mpm": MPMBenchmark,
    "pitch": PitchBenchmark,
    "hubert_token_sr": MHubert147TokenSRBenchmark,
    "allosaurus_sr": AllosaurusSRBenchmark,
    # speaker
    "wespeaker": WeSpeakerBenchmark,
    "dvector": DVectorBenchmark,
    # environment
    "voicerestore": VoiceRestoreBenchmark,
    "wada_snr": WadaSNRBenchmark,
    # intelligibility
    "wav2vec2_activations": Wav2Vec2XLSRActivationsBenchmark,
    "whisper_activations": MWhisperActivationsBenchmark,
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
    """
    A suite for running multiple benchmarks on multiple datasets.

    This class manages the execution of benchmarks, tracks progress, and
    computes aggregate scores. It supports parallel computation of distances
    for improved performance.

    Attributes:
        benchmarks (Dict[str, "Benchmark"]): Dictionary of benchmark instances to run
        datasets (List[Dataset]): List of datasets to evaluate
        reference_datasets (List[Dataset]): List of reference datasets for comparison
        noise_datasets (List[Dataset]): List of noise datasets for normalization
        category_weights (Dict[BenchmarkCategory, float]): Weights for each benchmark category
        skip_errors (bool): Whether to skip errors during benchmark execution
        write_to_file (Optional[str]): Path to write results to
        device (str): Device to run benchmarks on ('cpu' or 'cuda')
        cache_dir (Optional[str]): Directory to store cached results
        n_workers (int): Number of parallel workers for distance computation

    Examples:
        # Basic usage
        suite = BenchmarkSuite(datasets, reference_datasets)
        results = suite.run()

        # With parallel computation
        suite = BenchmarkSuite(datasets, reference_datasets, n_workers=4)
        results = suite.run()
    """

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
        write_to_file: Optional[str] = None,
        benchmark_kwargs: Dict[str, Dict[str, any]] = {},
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        include_environment: bool = False,
        multilingual: bool = False,
        n_workers: int = cpu_count(),
    ):
        """
        Initialize the BenchmarkSuite.

        Args:
            datasets: List of datasets to evaluate
            reference_datasets: List of reference datasets to compare against
            noise_datasets: List of noise datasets for normalization
            benchmarks: Dictionary of benchmark instances to run
            category_weights: Weights for each benchmark category
            skip_errors: Whether to skip errors during benchmark execution
            write_to_file: Path to write results to
            benchmark_kwargs: Additional keyword arguments for benchmarks
            device: Device to run benchmarks on ('cpu' or 'cuda')
            cache_dir: Directory to store cached results
            include_environment: Whether to include environment benchmarks
            multilingual: Whether to use multilingual benchmarks
            n_workers: Number of parallel workers for distance computation
        """
        if multilingual and benchmarks == BENCHMARKS:
            benchmarks = BENCHMARKS_ML
            print("Using multilingual benchmarks")
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
            k: v(**benchmark_kwargs.get(k, {})) if isinstance(v, type) else v
            for k, v in benchmarks.items()
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
            # Set the logger for each benchmark
            benchmark.set_logger(self.log_message)
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
        self.reference_datasets = reference_datasets
        self.noise_datasets = noise_datasets
        self.category_weights = category_weights
        self.skip_errors = skip_errors
        self.write_to_file = write_to_file
        self.cache_dir = cache_dir

        # Create a distance calculator with n_workers
        self.distance_calculator = DistanceCalculator(n_workers=n_workers)
        self.n_workers = n_workers

        # Live display setup for rich console output
        self.live = None
        self.layout = None
        self._progress = None
        self._progress_task = None
        self._task_info = {}
        self._finished_tasks = 0
        self._total_tasks = 0
        self._setup_caching()

    def _setup_caching(self) -> None:
        """
        Set up caching for the benchmark suite.
        Creates necessary directories if they don't exist.
        """
        self.distribution_cache = {}
        self.cache_file = os.path.join(self.cache_dir, "distribution_cache.pkl")
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self.distribution_cache = pickle.load(f)
            except Exception:
                self.distribution_cache = {}

    def _get_cache_key(self, benchmark: "Benchmark", dataset: Dataset) -> str:
        """
        Generate a cache key for a benchmark-dataset pair.

        Args:
            benchmark: The benchmark instance
            dataset: The dataset instance

        Returns:
            str: The cache key string
        """
        key_data = f"{benchmark.name}_{dataset.name}_{dataset.root_dir}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_distribution(self, benchmark: "Benchmark", dataset: Dataset) -> np.ndarray:
        """
        Get the distribution of a benchmark on a dataset.
        Uses caching when available.

        Args:
            benchmark: The benchmark instance
            dataset: The dataset instance

        Returns:
            np.ndarray: The distribution
        """
        cache_key = self._get_cache_key(benchmark, dataset)
        if cache_key in self.distribution_cache:
            return self.distribution_cache[cache_key]

        distribution = benchmark.get_distribution(dataset)
        self.distribution_cache[cache_key] = distribution

        # Save cache to disk
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.distribution_cache, f)

        return distribution

    def _setup_progress(self, total_tasks: int) -> None:
        """
        Set up the progress tracking for the benchmark suite.

        Args:
            total_tasks: Total number of tasks to track
        """
        console = Console()

        # Create layout
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="table", size=20),
            Layout(name="progress", size=3),
            Layout(name="log", size=10),  # Add a new section for log messages
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
        self._progress_task = self._progress.add_task(
            "[cyan]Running benchmarks...", total=total_tasks
        )

        # Initialize log storage
        self._log_messages = []

        # Initialize log panel with empty text
        log_panel = Panel("", title="Log Messages", border_style="green")

        # Update layout with components
        self.layout["table"].update(
            Panel(self._progress_table, title="Benchmark Progress", border_style="blue")
        )
        self.layout["progress"].update(self._progress)
        self.layout["log"].update(log_panel)

        # Store reference to log panel for updates
        self._log_panel = log_panel

        # Start the live display
        self.live = Live(
            self.layout,
            console=console,
            refresh_per_second=4,  # Limit refresh rate to reduce flickering
            vertical_overflow="visible",
        )
        self.live.start()

    def _update_progress(self) -> None:
        """
        Update the progress display with current task information.
        Updates the rich live display with benchmark status.
        """
        if self._progress and self._progress_table:
            # Update main progress
            self._progress.advance(self._progress_task)

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
            self.layout["table"].update(
                Panel(new_table, title="Benchmark Progress", border_style="blue")
            )
            self.layout["progress"].update(self._progress)

    def _stop_progress(self) -> None:
        """
        Stop the progress tracking and clean up resources.
        """
        if hasattr(self, "live") and self.live:
            # Clear the live display
            self.live.stop()

            # Clear the console to remove the progress display
            console = Console()
            console.clear()

            # Print final status
            completed = len(self.database[~self.database["score"].isna()])
            total = len(self.database)
            console.print(f"[green]Completed {completed}/{total} benchmarks[/green]")

        # Clean up other components
        if hasattr(self, "_progress") and self._progress:
            self._progress.stop()
            self._progress = None
        self._progress_table = None
        self._progress_task = None
        self._log_panel = None  # Clean up log panel reference
        if hasattr(self, "_log_messages"):
            self._log_messages = []  # Clean up log messages
        self.layout = None

    def log_message(self, message: str) -> None:
        """
        Log a message to the console.

        Args:
            message: The message to log
        """
        if self.live and hasattr(self, "_log_panel"):
            # Add new message to the history
            if not hasattr(self, "_log_messages"):
                self._log_messages = []

            # Add timestamp to message
            from datetime import datetime

            timestamp = datetime.now().strftime("%H:%M:%S")
            timestamped_message = f"[{timestamp}] {message}"

            # Add to message history
            self._log_messages.append(timestamped_message)

            # Keep only the last 10 messages for display
            if len(self._log_messages) > 10:
                self._log_messages = self._log_messages[-10:]

            # Join messages for display
            display_text = "\n".join(self._log_messages)

            # Update the log panel with combined messages
            self._log_panel.renderable = display_text

            # Update the layout
            self.layout["log"].update(self._log_panel)

    def _run_benchmark(
        self, benchmark: "Benchmark", dataset: Dataset
    ) -> Dict[str, any]:
        """
        Run a benchmark on a dataset.

        Args:
            benchmark: The benchmark to run
            dataset: The dataset to evaluate

        Returns:
            Dict[str, any]: Result data including scores and timing information
        """
        console = Console()

        try:
            # Log starting the benchmark
            self.log_message(
                f"[cyan]Running {benchmark.name} on {dataset.name}...[/cyan]"
            )

            start = time()
            score = benchmark.compute_score(
                dataset,
                self.reference_datasets,
                self.noise_datasets,
            )
            time_taken = time() - start

            # Log successful completion
            self.log_message(
                f"[green]Completed {benchmark.name} on {dataset.name} in {time_taken:.2f}s[/green]"
            )

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
            # Log error
            error_message = (
                f"[red]Error in {benchmark.name} on {dataset.name}: {str(e)}[/red]"
            )
            self.log_message(error_message)

            if self.skip_errors:
                console.print(error_message)
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
        """
        Run all benchmarks on all datasets.

        This method runs all configured benchmarks on all datasets, computes scores
        by comparing with reference datasets, and normalizes scores using noise datasets.

        Returns:
            pd.DataFrame: DataFrame containing benchmark results
        """
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
                self._progress.advance(self._progress_task)

        if not tasks:
            console.print("[green]All benchmarks have already been run![/green]")
            return self.database

        try:
            # Process tasks in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit all tasks to the executor
                future_to_task = {
                    executor.submit(self._run_benchmark, benchmark, dataset): (
                        benchmark,
                        dataset,
                    )
                    for benchmark, dataset in tasks
                }

                # Process results as they complete
                for future in future_to_task:
                    try:
                        result = future.result()
                        self._update_database(result)
                    except Exception as exc:
                        benchmark, dataset = future_to_task[future]
                        console.print(
                            f"[red]Task {benchmark.name} on {dataset.name} generated an exception: {exc}[/red]"
                        )
                        if not self.skip_errors:
                            raise exc

        finally:
            self._stop_progress()

        return self.database

    def _update_database(self, result: Dict[str, any]) -> None:
        """
        Update the results database with a new benchmark result.

        Args:
            result: Benchmark result data to add to the database
        """
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
        """
        Aggregate benchmark results by category and dataset.

        Args:
            df: DataFrame containing benchmark results

        Returns:
            pd.DataFrame: Aggregated results
        """

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
        """
        Get the aggregated results from the current database.

        Returns:
            pd.DataFrame: Aggregated benchmark results
        """
        df = self.database.copy()
        return self.aggregate_df(df)

    def stop(self) -> None:
        """
        Stop the benchmark suite and clean up resources.
        """
        self._stop_progress()

    def __del__(self) -> None:
        """
        Clean up resources when the benchmark suite is deleted.
        """
        self._stop_progress()

    def _setup_layout(self) -> None:
        """
        Set up the layout for the progress display.
        """
        pass  # This will be implemented when the suite is run
