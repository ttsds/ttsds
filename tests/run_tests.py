#!/usr/bin/env python
"""
Test runner script for TTSDS.

This script runs the test suite for TTSDS with proper configuration.
"""

import os
import sys
import pytest

if __name__ == "__main__":
    # Add the project root to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, project_root)

    # Add src to sys.path specifically
    src_path = os.path.join(project_root, "src")
    if os.path.exists(src_path):
        sys.path.insert(0, src_path)

    print(f"Python path: {sys.path}")
    print(f"Looking for ttsds at: {os.path.join(src_path, 'ttsds')}")

    # Parse command line arguments
    test_args = sys.argv[1:]

    # Define default exclude patterns for unstable or WIP tests
    default_excludes = [
        "tests/unit/benchmarks/speaker",  # Speaker tests need extensive mocking
    ]

    # Check if we should include all tests (including unstable ones)
    include_all = "--include-all" in test_args
    if include_all:
        test_args.remove("--include-all")
        exclude_patterns = []
    else:
        exclude_patterns = default_excludes

    # Define test arguments
    args = [
        "--verbose",
        "--color=yes",
        # Explicitly set the PYTHONPATH for subprocess tests
        "-o",
        f"pythonpath={src_path}:{project_root}",
        # Generate coverage data
        "--cov=src/ttsds",
        "--cov-report=term",
        "--cov-report=html:coverage_html",
    ]

    # Add exclude patterns
    for pattern in exclude_patterns:
        args.extend(["--ignore", pattern])

    # Add test paths or other arguments
    args.extend(test_args)

    print(f"Running pytest with args: {args}")

    # Run pytest
    exit_code = pytest.main(args)

    # Exit with the pytest exit code
    sys.exit(exit_code)
