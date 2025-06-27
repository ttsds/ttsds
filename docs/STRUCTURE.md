# TTSDS Documentation Structure

This file provides an overview of the documentation structure for TTSDS.

## Documentation Sections

### Home
- [index.md](index.md): Main landing page and overview of TTSDS

### User Guide
- [installation.md](user-guide/installation.md): Installation instructions
- [quickstart.md](user-guide/quickstart.md): Getting started guide
- [configuration.md](user-guide/configuration.md): Configuration options
- [advanced.md](user-guide/advanced.md): Advanced usage examples

### API Reference
- [benchmarks.md](reference/benchmarks.md): Benchmark classes documentation
- [utils.md](reference/utils.md): Utility functions documentation
- [ttsds.md](reference/ttsds.md): Main module documentation
- [SUMMARY.md](reference/SUMMARY.md): Reference navigation

### Contributing
- [development.md](contributing/development.md): Development guide
- [code-style.md](contributing/code-style.md): Code style guidelines
- [testing.md](contributing/testing.md): Testing guidelines

### About
- [license.md](about/license.md): License information
- [citation.md](about/citation.md): Citation information

## Build System

The documentation is built using MkDocs with the Material theme. The configuration is defined in `mkdocs.yml` at the root of the project.

To build the documentation:

```bash
# Install mkdocs and dependencies
pip install mkdocs mkdocs-material mkdocstrings

# Build the documentation
mkdocs build

# Serve the documentation locally
mkdocs serve
```

## Auto-generation

Some parts of the documentation are auto-generated:

- API reference navigation is generated using [gen_ref_nav.py](gen_ref_nav.py)
- API reference content can be automatically generated from docstrings using mkdocstrings

## Deployment

The documentation is deployed to GitHub Pages when changes are pushed to the main branch. 