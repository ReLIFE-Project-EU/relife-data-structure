# ReLIFE Data Structure

A small toolkit used in ReLIFE Task 2.1 to profile sample datasets and consolidate findings. It helps validate the data storage architecture by producing per-dataset exploratory data analysis (EDA) reports and a single consolidated summary with data quality insights.

## What this repository contains

> [!NOTE]
> The `reports/` and `consolidated_reports/` directories are generated locally and are gitignored, so they may not exist until you run the tools.

- **Samples and example outputs**: Data samples in `samples/` and example profiling outputs in `reports/`.
- **Report generator (`main.py`)**: Scans `samples/` for supported files and creates reports (HTML + JSON) under `reports/`, mirroring the folder structure.
- **Consolidator (`consolidate_reports.py`)**: Reads the JSON profiling outputs and produces a concise Markdown report with prioritized issues and an optional appendix.
- **Consolidation engine (`consolidator/`)**: Scanner, parser, analyzer, quality checks, templates, and orchestrator that power the consolidation workflow.
- **Automation**: `Taskfile.yml` tasks for setup and running; `tests/` for basic coverage.

## Key features

- **Multi-format ingestion**: `.csv`, `.sqlite`, `.parquet`, `.h5/.hdf5`, `.xlsx`.
- **Automatic discovery**: Recursively scans `samples/` and mirrors paths under `reports/`.
- **Rich output**: Per-table HTML + JSON profiling; consolidated Markdown with quality thresholds.
- **Configurable**: Flags for progress display, timestamps, appendix, and quality thresholds.
- **Simple setup**: Python 3.11+, managed via `uv` (recommended).

## Prerequisites

- Python 3.11 or newer
- `uv` CLI for environment management
- `task` (Taskfile runner) to execute the predefined tasks

## Quick start

1) Set up the environment:
```
task virtualenv
```
2) Generate profiling reports from `samples/`:
```
task run
```
3) Consolidate profiling JSONs into a single Markdown report:
```
task consolidate
```
For advanced consolidator options, run:
```
uv run consolidate_reports.py --help
```

## Repository structure

- `samples/`: Input data samples used for validation.
- `reports/`: Generated per-dataset profiling outputs (HTML + JSON).
- `consolidate_reports.py`: CLI to produce the consolidated Markdown report.
- `main.py`: Creates profiling reports from discovered data files.
- `consolidator/`: Core modules (scanner, parser, analyzer, quality, generator, orchestrator, templates).
- `Taskfile.yml`: Common tasks (`uv sync`, run, clean).
- `tests/`: Pytest-based tests for core functionality.

## Citations

This Exploratory Data Analysis pipeline uses the [YData Profiling package](https://docs.profiling.ydata.ai/latest/), which automates and standardizes the creation of detailed reports.