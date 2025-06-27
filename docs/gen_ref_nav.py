"""Generate the code reference documentation for mkdocs.

This script creates a navigation structure for the API reference section of the
documentation based on the module structure of the project. It also generates
the actual API documentation files using mkdocstrings directives.
"""

import pathlib
import os
from typing import Dict, List, Optional, Union

# Get the project root directory
root = pathlib.Path(__file__).parent.parent

# Main module mapping
MAIN_MODULES = {
    "ttsds": "TTSDS Main Module",
    "benchmarks": "Benchmark Classes",
    "util": "Utility Functions",
}

# Navigation structure
nav: List[Dict[str, Union[str, List]]] = []

# Create API reference directory structure if it doesn't exist
api_dir = root / "docs" / "reference" / "api"
api_dir.mkdir(exist_ok=True)


# Function to create mkdocstrings files
def create_api_file(module_path, output_path, title, show_root_heading=False):
    """Create an API documentation file with mkdocstrings directive."""

    # Create directories if they don't exist
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Write the file with mkdocstrings directive
    with open(output_path, "w") as f:
        f.write(f"---\ntitle: {title}\n---\n\n")
        f.write(f"# {title}\n\n")
        f.write(f"::: {module_path}\n")
        f.write("    options:\n")
        if not show_root_heading:
            f.write("      show_root_heading: false\n")
        f.write("      show_source: true\n")
        f.write("      members_order: source\n")
        f.write("      show_category_heading: true\n")
        f.write("      show_if_no_docstring: true\n")


# Create top-level module entries
for module, title in MAIN_MODULES.items():
    module_nav: Dict[str, Union[str, List]] = {title: {}}

    # Handle the case of the main module
    if module == "ttsds":
        # Create the main module file
        main_output = api_dir / "ttsds.md"
        create_api_file("ttsds.BenchmarkSuite", main_output, "BenchmarkSuite")

        module_nav[title] = "api/ttsds.md"
        nav.append(module_nav)
        continue

    # Get all Python files in the module directory
    module_path = root / "src" / "ttsds" / module
    if not module_path.exists():
        continue

    # Collect all Python files and subdirectories
    module_files = sorted(module_path.glob("*.py"))
    subdirs = sorted(
        [
            p
            for p in module_path.iterdir()
            if p.is_dir() and (p / "__init__.py").exists()
        ]
    )

    # Create entries for Python files in the module
    files_nav = []
    for py_file in module_files:
        if py_file.name == "__init__.py":
            # Create index file for the module
            module_index = api_dir / module / "index.md"
            create_api_file(f"ttsds.{module}", module_index, f"{title} Overview", True)
            files_nav.append({"Overview": f"api/{module}/index.md"})
        else:
            name = py_file.stem
            if name.startswith("_"):
                continue

            title_name = name.replace("_", " ").title()
            output_file = api_dir / module / f"{name}.md"

            # Create API file
            create_api_file(f"ttsds.{module}.{name}", output_file, title_name)

            files_nav.append({title_name: f"api/{module}/{name}.md"})

    # Create entries for subdirectories
    for subdir in subdirs:
        subdir_name = subdir.name
        if subdir_name.startswith("_"):
            continue

        subdir_title = subdir_name.replace("_", " ").title()
        subdir_files = sorted(subdir.glob("*.py"))

        subdir_nav = []
        for py_file in subdir_files:
            if py_file.name == "__init__.py":
                # Create index file for the submodule
                submodule_index = api_dir / module / subdir_name / "index.md"
                create_api_file(
                    f"ttsds.{module}.{subdir_name}",
                    submodule_index,
                    f"{subdir_title} Overview",
                    True,
                )
                subdir_nav.append({"Overview": f"api/{module}/{subdir_name}/index.md"})
            else:
                name = py_file.stem
                if name.startswith("_"):
                    continue

                title_name = name.replace("_", " ").title()
                output_file = api_dir / module / subdir_name / f"{name}.md"

                # Create API file
                create_api_file(
                    f"ttsds.{module}.{subdir_name}.{name}", output_file, title_name
                )

                subdir_nav.append({title_name: f"api/{module}/{subdir_name}/{name}.md"})

        if subdir_nav:
            files_nav.append({subdir_title: subdir_nav})

    if files_nav:
        module_nav[title] = files_nav
        nav.append(module_nav)

# Create the reference section navigation
with open(root / "docs" / "reference" / "SUMMARY.md", "w") as f:
    f.write("# API Reference\n\n")
    f.write("This documentation is automatically generated from the source code.\n\n")

    for section in nav:
        for title, content in section.items():
            f.write(f"## {title}\n\n")

            if isinstance(content, str):
                # Direct link
                f.write(f"* [{title}]({content})\n")
            else:
                # List of items
                for item in content:
                    for item_title, item_link in item.items():
                        if isinstance(item_link, str):
                            f.write(f"* [{item_title}]({item_link})\n")
                        else:
                            # Nested list
                            f.write(f"* {item_title}\n")
                            for subitem in item_link:
                                for subitem_title, subitem_link in subitem.items():
                                    f.write(
                                        f"    * [{subitem_title}]({subitem_link})\n"
                                    )

            f.write("\n")

print(f"Reference documentation generated in {api_dir}")
print(f"Reference navigation generated in {root / 'docs' / 'reference' / 'SUMMARY.md'}")
