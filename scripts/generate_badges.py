#!/usr/bin/env python
"""
Generate badges for test coverage and test status.

This script:
1. Runs the test suite to collect coverage data
2. Generates a coverage badge using coverage-badge
3. Generates a test status badge based on test results
4. Saves badges to the docs/assets/img directory
"""

import os
import sys
import subprocess
import re
from pathlib import Path


def run_tests_with_coverage():
    """Run tests with coverage and return pass/fail status."""
    print("Running tests with coverage...")

    # Create the coverage directory if it doesn't exist
    coverage_dir = Path("coverage_html")
    coverage_dir.mkdir(exist_ok=True)

    # Run tests with coverage
    cmd = [
        "python",
        "tests/run_tests.py",
        "tests/unit",
        "--ignore",
        "tests/unit/benchmarks/speaker",  # Skip speaker tests that require extensive mocking
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout

    # Extract coverage percentage
    coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)%", stdout)
    coverage_percent = int(coverage_match.group(1)) if coverage_match else 0

    # Check if tests passed
    tests_passed = result.returncode == 0

    return tests_passed, coverage_percent


def generate_coverage_badge(coverage_percent):
    """Generate a coverage badge with the given percentage."""
    print(f"Generating coverage badge for {coverage_percent}%...")

    # Create directories if they don't exist
    img_dir = Path("docs/assets/img")
    img_dir.mkdir(exist_ok=True, parents=True)

    # Create badge SVG
    badge_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="104" height="20">
    <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <mask id="a">
        <rect width="104" height="20" rx="3" fill="#fff"/>
    </mask>
    <g mask="url(#a)">
        <path fill="#555" d="M0 0h61v20H0z"/>
        <path fill="{get_color_for_coverage(coverage_percent)}" d="M61 0h43v20H61z"/>
        <path fill="url(#b)" d="M0 0h104v20H0z"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="30.5" y="15" fill="#010101" fill-opacity=".3">coverage</text>
        <text x="30.5" y="14">coverage</text>
        <text x="81.5" y="15" fill="#010101" fill-opacity=".3">{coverage_percent}%</text>
        <text x="81.5" y="14">{coverage_percent}%</text>
    </g>
</svg>"""

    # Write badge to file
    badge_path = img_dir / "coverage.svg"
    with open(badge_path, "w") as f:
        f.write(badge_content)

    print(f"Coverage badge generated at {badge_path}")
    return badge_path


def get_color_for_coverage(percent):
    """Get color based on coverage percentage."""
    if percent >= 90:
        return "#4c1"  # Bright green
    elif percent >= 75:
        return "#97CA00"  # Green-yellow
    elif percent >= 50:
        return "#dfb317"  # Yellow
    elif percent >= 25:
        return "#fe7d37"  # Orange
    else:
        return "#e05d44"  # Red


def generate_test_status_badge(passed):
    """Generate a test status badge."""
    print(f"Generating test status badge (passed={passed})...")

    # Create directories if they don't exist
    img_dir = Path("docs/assets/img")
    img_dir.mkdir(exist_ok=True, parents=True)

    # Set badge color and text based on test status
    color = "#4c1" if passed else "#e05d44"  # Green if passed, red if failed
    status_text = "passing" if passed else "failing"

    # Create badge SVG
    badge_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="78" height="20">
    <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <mask id="a">
        <rect width="78" height="20" rx="3" fill="#fff"/>
    </mask>
    <g mask="url(#a)">
        <path fill="#555" d="M0 0h37v20H0z"/>
        <path fill="{color}" d="M37 0h41v20H37z"/>
        <path fill="url(#b)" d="M0 0h78v20H0z"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="18.5" y="15" fill="#010101" fill-opacity=".3">tests</text>
        <text x="18.5" y="14">tests</text>
        <text x="56.5" y="15" fill="#010101" fill-opacity=".3">{status_text}</text>
        <text x="56.5" y="14">{status_text}</text>
    </g>
</svg>"""

    # Write badge to file
    badge_path = img_dir / "tests.svg"
    with open(badge_path, "w") as f:
        f.write(badge_content)

    print(f"Test status badge generated at {badge_path}")
    return badge_path


def update_readme_with_badges():
    """Update the README.md with badge links."""
    print("Updating README with badges...")
    readme_path = Path("README.md")

    if not readme_path.exists():
        print("README.md not found. Skipping update.")
        return

    with open(readme_path, "r") as f:
        content = f.read()

    # Check if badges are already in the README
    if "[![Tests]" in content and "[![Coverage]" in content:
        print("README already has badges. No update needed.")
        return

    # Add badges after the title and any existing badges
    lines = content.split("\n")

    # Find the title and any existing badges
    title_index = -1
    for i, line in enumerate(lines):
        if line.startswith("# "):
            title_index = i
            break

    if title_index == -1:
        print("Could not find title in README. Skipping update.")
        return

    # Find the blank line after the title/badges
    blank_line_index = title_index + 1
    while blank_line_index < len(lines) and lines[blank_line_index].strip():
        blank_line_index += 1

    # Insert badges
    badge_line = "[![PyPI - Version](https://img.shields.io/pypi/v/ttsds.svg)](https://pypi.org/project/ttsds) [![Tests](https://raw.githubusercontent.com/ttsds/ttsds/main/docs/assets/img/tests.svg)](https://github.com/ttsds/ttsds/actions) [![Coverage](https://raw.githubusercontent.com/ttsds/ttsds/main/docs/assets/img/coverage.svg)](https://github.com/ttsds/ttsds/actions)"

    # Check if the PyPI badge already exists
    if "[![PyPI" in "\n".join(lines[title_index:blank_line_index]):
        # Replace the line with the PyPI badge
        for i in range(title_index + 1, blank_line_index):
            if "[![PyPI" in lines[i]:
                lines[i] = badge_line
                break
    else:
        # Insert after the title
        lines.insert(title_index + 1, badge_line)

    # Write back to the file
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))

    print("README updated with badges.")


def main():
    """Main function to generate badges."""
    # Run tests and get status
    tests_passed, coverage_percent = run_tests_with_coverage()

    # Generate badges
    generate_coverage_badge(coverage_percent)
    generate_test_status_badge(tests_passed)

    # Update README
    update_readme_with_badges()

    print("Badge generation complete!")


if __name__ == "__main__":
    main()
