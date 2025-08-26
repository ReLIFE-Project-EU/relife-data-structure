"""
A script to sample CSV files so entire datasets do not need to be uploaded
to Git for profiling purposes.

Mirrors the CLI and logging style of the SQLite sampler, supporting:
- Fixed ratio sampling or target output size with iterative adjustment
- Per-file minimum and maximum rows
- Directory or single-file inputs
- Final overview logging of sampled outputs
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from rich.console import Console

from logging_setup import configure_logging as _configure_logging

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    """Configure logging for the CLI using shared setup."""

    _configure_logging(verbose=verbose, console=Console())


@dataclass
class FilePlan:
    """Planning information for a single CSV file."""

    source_path: Path
    output_path: Path
    total_rows: int
    planned_take: int


def _list_csv_files(source: Path, recursive: bool) -> List[Path]:
    """Return a sorted list of CSV files under ``source``.

    If ``source`` is a file, returns [source] if it has a .csv suffix.
    If ``source`` is a directory, searches for files with .csv suffix.
    """

    if source.is_file():
        return [source] if source.suffix.lower() == ".csv" else []

    if not source.exists():
        return []

    results: List[Path] = []
    if recursive:
        for p in source.rglob("*.csv"):
            if p.is_file():
                results.append(p)
    else:
        for p in source.glob("*.csv"):
            if p.is_file():
                results.append(p)

    return sorted(results)


def _ensure_output_location(source: Path, output: Path) -> Tuple[Path, bool]:
    """Normalize output path and indicate whether it represents a directory.

    - If ``source`` is a file:
        - If ``output`` is an existing directory, returns (output/source.name, False)
        - Else, returns (output, False)
    - If ``source`` is a directory:
        - Returns (output, True). The directory will be created by the caller.
    """

    if source.is_file():
        if output.exists() and output.is_dir():
            return (output / source.name, False)
        return (output, False)

    # Directory input
    return (output, True)


def _count_rows(file_path: Path, *, has_header: bool, encoding: str) -> int:
    """Return the number of data rows in a CSV file.

    Assumes the first row is a header when ``has_header`` is True.
    """

    try:
        with file_path.open("r", newline="", encoding=encoding) as f:
            reader = csv.reader(f)
            count = 0
            if has_header:
                next(reader, None)
            for _ in reader:
                count += 1
            return count
    except Exception as exc:
        logger.warning("Failed to count rows in %s: %s", file_path, exc)
        return 0


def _reservoir_sample_rows(
    file_path: Path,
    *,
    k: int,
    has_header: bool,
    seed: Optional[int],
    encoding: str,
) -> Tuple[Optional[List[str]], List[List[str]]]:
    """Sample ``k`` rows uniformly from CSV using reservoir sampling.

    Returns (header, sampled_rows). ``header`` is None when ``has_header`` is False.
    """

    if k <= 0:
        # Return just header if present; no rows
        header: Optional[List[str]] = None
        try:
            if has_header:
                with file_path.open("r", newline="", encoding=encoding) as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
        except Exception as exc:
            logger.warning("Failed reading header from %s: %s", file_path, exc)
        return (header, [])

    rng = random.Random(seed)
    reservoir: List[List[str]] = []
    header_row: Optional[List[str]] = None

    try:
        with file_path.open("r", newline="", encoding=encoding) as f:
            reader = csv.reader(f)
            if has_header:
                header_row = next(reader, None)

            # Fill the reservoir with the first k rows
            for _ in range(k):
                try:
                    row = next(reader)
                except StopIteration:
                    break
                reservoir.append(row)

            # Continue with reservoir sampling for the rest
            # i is the index of the next item after the initial reservoir
            i = k
            for row in reader:
                j = rng.randint(0, i)
                if j < k:
                    reservoir[j] = row
                i += 1
    except Exception as exc:
        logger.warning("Failed to sample rows from %s: %s", file_path, exc)

    return (header_row, reservoir)


def _copy_file(src: Path, dst: Path) -> None:
    """Copy a file from ``src`` to ``dst``."""

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_csv(
    dst: Path, header: Optional[List[str]], rows: Iterable[List[str]], *, encoding: str
) -> None:
    """Write a CSV file to ``dst`` with ``header`` and ``rows``.

    If ``header`` is None, the file will be created without a header row.
    """

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding=encoding) as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        for r in rows:
            writer.writerow(r)


def _sum_file_sizes(paths: Iterable[Path]) -> int:
    """Return the total size of all files in ``paths``."""

    total = 0
    for p in paths:
        try:
            total += p.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _remove_output_tree(path: Path) -> None:
    """Remove a file or directory at ``path``."""

    if path.is_file():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def _plan_per_file(
    files: List[Path],
    ratio_est: float,
    *,
    has_header: bool,
    min_rows_per_file: int,
    max_rows_per_file: Optional[int],
    encoding: str,
) -> Dict[Path, Tuple[int, int]]:
    """Return a map of file -> (total_rows, planned_take)."""

    plan: Dict[Path, Tuple[int, int]] = {}
    for fp in files:
        total = _count_rows(fp, has_header=has_header, encoding=encoding)
        take = min(
            total,
            max(
                min_rows_per_file if total > 0 else 0,
                math.ceil(total * ratio_est),
            ),
        )
        if max_rows_per_file is not None:
            take = min(take, max_rows_per_file)
        plan[fp] = (total, take)
    return plan


def _log_output_overview(output_root: Path, *, has_header: bool, encoding: str) -> None:
    """Log overview of sampled CSV outputs (files, rows, sizes)."""

    try:
        if output_root.is_file():
            files = [output_root]
        elif output_root.is_dir():
            files = sorted([p for p in output_root.rglob("*.csv") if p.is_file()])
        else:
            logger.info("No output produced at %s", output_root)
            return

        sizes = _sum_file_sizes(files)
        total_mb = sizes / (1024 * 1024) if sizes else 0.0
        logger.info(
            "Overview of output sample: %d file(s), total size %.2f MB",
            len(files),
            total_mb,
        )

        for f in files:
            rows = _count_rows(f, has_header=has_header, encoding=encoding)
            size_mb = (f.stat().st_size / (1024 * 1024)) if f.exists() else 0.0
            logger.info(
                "  - %s: %d rows, ~%.2f MB",
                f.relative_to(output_root.parent),
                rows,
                size_mb,
            )
    except Exception as exc:
        logger.warning("Failed to build CSV output overview: %s", exc)


def sample_csv(
    source: Path,
    output: Path,
    *,
    target_size_mb: Optional[float],
    ratio: Optional[float],
    min_rows_per_file: int,
    max_rows_per_file: Optional[int],
    max_iterations: int,
    has_header: bool,
    recursive: bool,
    encoding: str,
    seed: Optional[int],
) -> None:
    """Create sampled copies of CSV file(s).

    If ``source`` is a directory, recursively samples all ``.csv`` files (unless
    ``recursive`` is False). If ``source`` is a file, samples only that file.
    Either samples by a fixed ratio or iteratively adjusts the ratio to reach
    an approximate target output size in megabytes.
    """

    if not source.exists():
        raise FileNotFoundError(f"Source path not found: {source}")
    if target_size_mb is None and ratio is None:
        raise ValueError("Either --target-size-mb or --ratio must be provided")
    if ratio is not None and not (0.0 < ratio <= 1.0):
        raise ValueError("--ratio must be in (0, 1]")
    if max_rows_per_file is not None and max_rows_per_file < 1:
        raise ValueError("--max-rows-per-file must be >= 1 when provided")

    files = _list_csv_files(source, recursive=recursive)
    if not files:
        raise ValueError("No CSV files found in source")

    normalized_output, output_is_dir = _ensure_output_location(source, output)

    # Branch: per-file target-size control when sampling a directory
    # In this mode, each source file is sampled iteratively to respect the
    # provided target size for its corresponding output file.
    if target_size_mb is not None and output_is_dir:
        desired_bytes = int(target_size_mb * 1024 * 1024)

        # Clean previous outputs and prepare directory
        _remove_output_tree(normalized_output)
        normalized_output.mkdir(parents=True, exist_ok=True)

        for src_file in files:
            # Output path mirrors directory structure under source
            rel = src_file.relative_to(source) if source.is_dir() else src_file.name
            dst_file = normalized_output / rel  # type: ignore[operator]
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            # Initial ratio guess per file (bytes-based heuristic)
            if ratio is None:
                try:
                    src_bytes = src_file.stat().st_size
                except FileNotFoundError:
                    src_bytes = 0
                if src_bytes == 0:
                    ratio_est_file = 1.0
                else:
                    ratio_est_file = min(1.0, max(0.0001, desired_bytes / src_bytes))
            else:
                ratio_est_file = ratio

            # Determine total rows once per file for planning
            total_rows = _count_rows(src_file, has_header=has_header, encoding=encoding)

            attempt = 0
            while True:
                attempt += 1

                logger.info(
                    "Creating CSV sample for %s (attempt %d) with ratio ~ %.4f",
                    src_file,
                    attempt,
                    ratio_est_file,
                )

                # Compute planned rows for this attempt
                take = min(
                    total_rows,
                    max(
                        min_rows_per_file if total_rows > 0 else 0,
                        math.ceil(total_rows * ratio_est_file),
                    ),
                )
                if max_rows_per_file is not None:
                    take = min(take, max_rows_per_file)

                # Produce sampled output
                if total_rows == 0 or take == 0:
                    if has_header and total_rows >= 0:
                        header, _rows = _reservoir_sample_rows(
                            src_file,
                            k=0,
                            has_header=has_header,
                            seed=seed,
                            encoding=encoding,
                        )
                        _write_csv(dst_file, header, [], encoding=encoding)
                    else:
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        dst_file.write_text("", encoding=encoding)
                    logger.debug("%s: no rows to copy", src_file)
                elif take >= total_rows:
                    _copy_file(src_file, dst_file)
                    logger.info("%s: copied all %d rows", src_file, total_rows)
                else:
                    header, rows = _reservoir_sample_rows(
                        src_file,
                        k=take,
                        has_header=has_header,
                        seed=seed,
                        encoding=encoding,
                    )
                    _write_csv(
                        dst_file,
                        header if has_header else None,
                        rows,
                        encoding=encoding,
                    )
                    logger.info(
                        "%s: inserted %d / planned %d (source %d)",
                        src_file,
                        len(rows),
                        take,
                        total_rows,
                    )

                # Evaluate size and decide whether to iterate further
                try:
                    out_size = dst_file.stat().st_size
                except FileNotFoundError:
                    out_size = 0
                logger.info(
                    "%s output size: %.2f MB (target %.2f MB)",
                    dst_file,
                    out_size / (1024 * 1024),
                    desired_bytes / (1024 * 1024),
                )

                if out_size <= desired_bytes or attempt >= max_iterations:
                    if out_size > desired_bytes:
                        logger.warning(
                            "%s remains above target after %d attempts (%.2f MB > %.2f MB). Keeping latest sample.",
                            dst_file,
                            attempt,
                            out_size / (1024 * 1024),
                            desired_bytes / (1024 * 1024),
                        )
                    break

                # Adjust ratio downward conservatively and try again
                shrink_factor = desired_bytes / out_size if out_size > 0 else 0.5
                ratio_est_file = max(
                    0.00005, min(1.0, ratio_est_file * shrink_factor * 0.95)
                )

        # Final overview for directory sampling
        _log_output_overview(
            normalized_output, has_header=has_header, encoding=encoding
        )
        return

    # Global aggregated target-size mode (single file output or no per-file target needed)
    # Keep previous behavior when:
    # - target_size_mb is None (ratio-based), OR
    # - source is a single file (we target that file size), OR
    # - output is not a directory (single output file requested)
    src_total_bytes = _sum_file_sizes(files)
    desired_bytes = (
        None if target_size_mb is None else int(target_size_mb * 1024 * 1024)
    )

    if ratio is None:
        ratio_est = 1.0 if src_total_bytes == 0 else min(1.0, max(0.0001, src_total_bytes and desired_bytes / src_total_bytes))  # type: ignore[operator]
    else:
        ratio_est = ratio

    attempt = 0
    while True:
        attempt += 1

        # Clean previous outputs
        _remove_output_tree(normalized_output)
        if output_is_dir:
            normalized_output.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Creating CSV sample (attempt %d) with ratio ~ %.4f", attempt, ratio_est
        )

        # Plan per file based on current ratio estimate
        plan_map = _plan_per_file(
            files,
            ratio_est,
            has_header=has_header,
            min_rows_per_file=min_rows_per_file,
            max_rows_per_file=max_rows_per_file,
            encoding=encoding,
        )

        # Execute sampling for each file
        for src_file in files:
            total, take = plan_map[src_file]

            # Determine output path for this file
            if output_is_dir:
                rel = (
                    src_file.name if source.is_file() else src_file.relative_to(source)
                )
                dst_file = normalized_output / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                dst_file = normalized_output

            if total == 0 or take == 0:
                if has_header and total >= 0:
                    header, _rows = _reservoir_sample_rows(
                        src_file,
                        k=0,
                        has_header=has_header,
                        seed=seed,
                        encoding=encoding,
                    )
                    _write_csv(dst_file, header, [], encoding=encoding)
                else:
                    # Create empty file
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    dst_file.write_text("", encoding=encoding)
                logger.debug("%s: no rows to copy", src_file)
                # If a single-file input producing a single output, stop after first
                if not output_is_dir:
                    break
                continue

            if take >= total:
                # Copy as-is
                _copy_file(src_file, dst_file)
                logger.info("%s: copied all %d rows", src_file, total)
            else:
                header, rows = _reservoir_sample_rows(
                    src_file,
                    k=take,
                    has_header=has_header,
                    seed=seed,
                    encoding=encoding,
                )
                _write_csv(
                    dst_file, header if has_header else None, rows, encoding=encoding
                )
                logger.info(
                    "%s: inserted %d / planned %d (source %d)",
                    src_file,
                    len(rows),
                    take,
                    total,
                )

            # For single-file scenario, we produced the output file already
            if not output_is_dir:
                break

        # If no target size, we are done
        if desired_bytes is None:
            break

        # Compute total output size
        out_paths: List[Path]
        if output_is_dir:
            out_paths = [p for p in normalized_output.rglob("*.csv") if p.is_file()]
        else:
            out_paths = [normalized_output]

        out_size = _sum_file_sizes(out_paths)
        logger.info(
            "Output CSV size: %.2f MB (target %.2f MB)",
            out_size / (1024 * 1024),
            desired_bytes / (1024 * 1024),
        )

        if out_size <= desired_bytes or attempt >= max_iterations:
            if out_size > desired_bytes:
                logger.warning(
                    "Sample remains above target after %d attempts (%.2f MB > %.2f MB). Keeping latest sample.",
                    attempt,
                    out_size / (1024 * 1024),
                    desired_bytes / (1024 * 1024),
                )
            break

        # Adjust ratio downward conservatively
        shrink_factor = desired_bytes / out_size if out_size > 0 else 0.5
        ratio_est = max(0.00005, min(1.0, ratio_est * shrink_factor * 0.95))

    # Final overview
    _log_output_overview(
        normalized_output if output_is_dir else normalized_output,
        has_header=has_header,
        encoding=encoding,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the CSV sampler CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Create randomly-sampled copies of CSV file(s) while preserving headers and directory structure."
        )
    )

    parser.add_argument(
        "source", type=Path, help="Path to source .csv file or directory"
    )
    parser.add_argument(
        "output", type=Path, help="Path to output .csv file or directory"
    )

    size_group = parser.add_mutually_exclusive_group(required=False)
    size_group.add_argument(
        "--target-size-mb",
        type=float,
        help="Approximate max size of output in MB (across all files)",
    )
    size_group.add_argument("--ratio", type=float, help="Sampling ratio in (0,1]")

    parser.add_argument(
        "--min-rows-per-file",
        type=int,
        default=1,
        help="Minimum rows to include per non-empty CSV file",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=int(1e6),
        help="Approximate cap of rows to include per CSV (overrides minimum when smaller)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max attempts to adjust ratio to reach target size",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Treat CSVs as having no header row (default assumes a header)",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="When source is a directory, do not search recursively",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="File encoding for reading/writing CSV",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for creating sampled CSV file(s)."""

    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        sample_csv(
            source=args.source,
            output=args.output,
            target_size_mb=args.target_size_mb,
            ratio=args.ratio,
            min_rows_per_file=args.min_rows_per_file,
            max_rows_per_file=args.max_rows_per_file,
            max_iterations=args.max_iterations,
            has_header=not args.no_header,
            recursive=not args.non_recursive,
            encoding=args.encoding,
            seed=args.seed,
        )
    except Exception as exc:
        logger.exception("Failed to sample CSV: %s", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
