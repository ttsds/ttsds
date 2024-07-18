import platform
from subprocess import run
import subprocess
from typing import List
from multiprocessing import cpu_count
import os

from ttsdb.util.cache import CACHE_DIR

CPUS = cpu_count()
KALDI_PATH = os.getenv("TTSDB_KALDI_PATH", CACHE_DIR / "kaldi")


def run_commands(
    commands: List[str], directory: str = None, suppress_output: bool = False
):
    """
    Run a list of commands.
    """
    for command in commands:
        if directory:
            run_command(command, directory, suppress_output)
        else:
            run_command(command, suppress_output=suppress_output)


def run_command(command: str, directory: str = None, suppress_output: bool = False):
    """
    Run a command.
    """
    if suppress_output:
        if directory:
            run(
                command,
                shell=True,
                check=True,
                cwd=directory,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    elif directory:
        run(command, shell=True, check=True, cwd=directory)
    else:
        run(command, shell=True, check=True)


def install_kaldi(kaldi_path: str = KALDI_PATH, verbose: bool = False):
    """
    Install Kaldi.
    """
    try:
        run_command(
            f"{kaldi_path}/src/featbin/compute-mfcc-feats --help",
            suppress_output=True,
        )
        os.environ["KALDI_ROOT"] = kaldi_path
        return
    except Exception as e:
        pass
    if not kaldi_path.exists():
        yn_install_kaldi = input(
            f"TTSDB_KALDI_PATH is not set. Do you want to install Kaldi to {kaldi_path}? (y/n) "
        )
    else:
        yn_install_kaldi = input(f"Overwrite Kaldi at {kaldi_path}? (y/n) ")
        if yn_install_kaldi.lower() == "n":
            return
        run_command(f"rm -rf {kaldi_path}", suppress_output=not verbose)
    kaldi_path = kaldi_path.resolve()
    if yn_install_kaldi.lower() == "y":
        run_command(
            f"git clone https://github.com/kaldi-asr/kaldi.git {KALDI_PATH}",
            suppress_output=not verbose,
        )
        run_command(
            f"cd {kaldi_path} && git checkout 26b9f648",
            suppress_output=not verbose,
        )
        try:
            is_osx = platform.system() == "Darwin"
            if not is_osx:
                run_commands(
                    [
                        f"cd {kaldi_path}/tools && \
                                sed -i 's/python2.7/python3/' extras/check_dependencies.sh"
                    ],
                    suppress_output=not verbose,
                )
            else:
                run_commands(
                    [
                        f"cd {kaldi_path}/tools && \
                                sed -i '' 's/python2.7/python3/' extras/check_dependencies.sh"
                    ],
                    suppress_output=not verbose,
                )
            run_commands(
                [
                    f"cd {kaldi_path}/tools && ./extras/check_dependencies.sh",
                    f"cd {kaldi_path}/tools && make -j {CPUS}",
                    f"cd {kaldi_path}/src && ./configure --shared",
                    f"cd {kaldi_path}/src && make depend -j {CPUS}",
                    f"cd {kaldi_path}/src && make -j {CPUS}",
                ],
                suppress_output=not verbose,
            )
        except Exception as e:
            print(f"Error installing Kaldi: {e}")
            # remove kaldi
            # run_command(f"rm -rf {kaldi_path}", suppress_output=not verbose)
            raise e
        os.environ["KALDI_ROOT"] = kaldi_path
