"""
A script to sample HDF5 files so entire datasets do not need to be uploaded
to Git for profiling purposes.

Style and CLI are inspired by the CSV and SQLite samplers in this project.

Capabilities:
- Sample a single HDF5 file or all HDF5 files in a directory
- Choose fixed ratio or target output size with iterative adjustment
- Preserve group structure and copy node attributes
- Sample along the first dimension for tables and arrays
- Per-node minimum and maximum rows
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import tables as tb
from rich.console import Console

from logging_setup import configure_logging as _configure_logging

logger = logging.getLogger(__name__)

# Suppress benign PyTables warnings about non-identifier names in groups/attributes.
# We do not rely on natural naming anywhere; we access nodes via explicit paths.
warnings.filterwarnings("ignore", category=tb.NaturalNameWarning)


def configure_logging(verbose: bool) -> None:
    """Configure logging for the CLI using shared setup."""

    _configure_logging(verbose=verbose, console=Console())


@dataclass
class NodePlan:
    """Planned sampling information for a single HDF5 leaf node."""

    node_path: str
    total_rows: int
    planned_take: int


def _list_hdf5_files(source: Path, recursive: bool) -> List[Path]:
    """Return a sorted list of HDF5 files under ``source``.

    Recognizes common suffixes: .h5, .hdf5, .hdf
    """

    suffixes = {".h5", ".hdf5", ".hdf"}

    if source.is_file():
        return [source] if source.suffix.lower() in suffixes else []

    if not source.exists():
        return []

    results: List[Path] = []
    pattern = "**/*" if recursive else "*"
    for p in source.glob(pattern):
        if p.is_file() and p.suffix.lower() in suffixes:
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
        # Remove all files within while preserving the directory if desired
        for child in path.glob("**/*"):
            if child.is_file():
                try:
                    child.unlink()
                except FileNotFoundError:
                    pass


def _copy_attrs(dst_node: tb.Node, src_node: tb.Node) -> None:
    """Copy all user attributes from ``src_node`` to ``dst_node``."""

    try:
        src_attrs = src_node._v_attrs
        dst_attrs = dst_node._v_attrs
        for name in src_attrs._v_attrnamesuser:  # type: ignore[attr-defined]
            try:
                dst_attrs[name] = src_attrs[name]
            except Exception:
                # Be tolerant with exotic attribute types
                continue
    except Exception:
        # Attributes are optional; ignore failures
        return


def _ensure_group(dst: tb.File, src: tb.File, path: str) -> tb.Group:
    """Ensure group ``path`` exists in ``dst`` mirroring from ``src``."""

    # Root group
    if path in ("/", ""):
        return dst.root

    # Walk and create as needed
    parts = [p for p in path.split("/") if p]
    where = "/"
    current_group: tb.Group = dst.root
    src_group: tb.Group = src.get_node(path)  # type: ignore[assignment]

    for idx, name in enumerate(parts):
        # Compute absolute path for this level
        where = "/" + "/".join(parts[: idx + 1])
        try:
            current_group = dst.get_node(where)  # type: ignore[assignment]
            if isinstance(current_group, tb.Group):
                pass
            else:
                raise tb.NodeError(f"Non-group node exists at {where}")
        except tb.NoSuchNodeError:
            parent_where = "/" + "/".join(parts[:idx]) if idx > 0 else "/"
            parent_group: tb.Group = dst.get_node(parent_where)  # type: ignore[assignment]
            # Mirror title if available
            src_current: tb.Group = src.get_node(where)  # type: ignore[assignment]
            title = getattr(src_current, "_v_title", "")
            current_group = dst.create_group(parent_group, name, title=title)
            _copy_attrs(current_group, src_current)

    return current_group


def _leaf_total_rows(node: tb.Leaf) -> Optional[int]:
    """Return the total number of rows/items along the first dimension for a leaf.

    - For Table: uses ``nrows``
    - For Arrays (Array/CArray/EArray): uses ``shape[0]``
    Returns None for unsupported leaf types.
    """

    if isinstance(node, tb.Table):
        return int(node.nrows)
    if isinstance(node, (tb.Array, tb.CArray, tb.EArray)):
        shape_obj = getattr(node, "shape", None)
        if not shape_obj:
            return 0
        shape = tuple(shape_obj)
        if len(shape) == 0:
            return 0
        return int(shape[0])
    return None


def _plan_nodes(
    src_h5: tb.File,
    ratio_est: float,
    *,
    min_rows_per_node: int,
    max_rows_per_node: Optional[int],
) -> Dict[str, NodePlan]:
    """Return a plan mapping leaf node paths to row counts to take."""

    plan: Dict[str, NodePlan] = {}
    for leaf in src_h5.walk_nodes("/", classname="Leaf"):
        total_opt = _leaf_total_rows(leaf)  # type: ignore[arg-type]
        if total_opt is None:
            continue
        total = total_opt
        take = min(
            total,
            max(min_rows_per_node if total > 0 else 0, math.ceil(total * ratio_est)),
        )
        if max_rows_per_node is not None:
            take = min(take, max_rows_per_node)
        path_str = cast(str, leaf._v_pathname)
        plan[path_str] = NodePlan(path_str, total, take)
    return plan


def _sample_indices(total: int, k: int, *, seed: Optional[int]) -> List[int]:
    """Return sorted indices for a uniform sample of size ``k`` from range(total)."""

    if k <= 0:
        return []
    if k >= total:
        return list(range(total))
    rng = random.Random(seed)
    # Use random.sample for simplicity and stable reproducibility
    idxs = rng.sample(range(total), k)
    idxs.sort()
    return idxs


def _copy_or_sample_table(
    dst: tb.File,
    src: tb.File,
    src_table: tb.Table,
    dst_parent: tb.Group,
    *,
    take: int,
) -> None:
    """Copy or sample a PyTables Table into ``dst_parent``."""

    total = int(src_table.nrows)
    title = src_table._v_title
    name = src_table._v_name
    filters = src_table.filters

    if take >= total:
        # Full copy
        dst_table = dst.create_table(
            dst_parent,
            name,
            src_table.description,
            title=title,
            expectedrows=total,
            filters=filters,
        )
        # Append in chunks for performance
        chunk = 100000
        start = 0
        while start < total:
            stop = min(start + chunk, total)
            recs = src_table.read(start, stop)
            dst_table.append(recs)
            start = stop
        dst_table.flush()
        _copy_attrs(dst_table, src_table)
        return

    # Sampled copy
    indices = _sample_indices(total, take, seed=None)
    dst_table = dst.create_table(
        dst_parent,
        name,
        src_table.description,
        title=title,
        expectedrows=take,
        filters=filters,
    )
    # Read selected rows in coordinate order
    if indices:
        recs = src_table.read_coordinates(np.asarray(indices, dtype=np.int64))
        dst_table.append(recs)
    dst_table.flush()
    _copy_attrs(dst_table, src_table)


def _copy_or_sample_array(
    dst: tb.File,
    src: tb.File,
    src_array: tb.Leaf,
    dst_parent: tb.Group,
    *,
    take: int,
) -> None:
    """Copy or sample a 1D+ array-like leaf along axis 0 into ``dst_parent``."""

    assert isinstance(src_array, (tb.Array, tb.CArray, tb.EArray))
    shape_obj = getattr(src_array, "shape", None)
    if not shape_obj or len(shape_obj) == 0:
        total = 0
    else:
        total = int(shape_obj[0])
    name = src_array._v_name
    title = src_array._v_title
    filters = getattr(src_array, "filters", None)
    atom = getattr(src_array, "atom", None)

    if take >= total:
        # Full copy using CArray with same shape
        shape_tuple: Tuple[int, ...] | None
        try:
            shape_tuple = tuple(shape_obj) if shape_obj is not None else tuple()
        except TypeError:
            shape_tuple = tuple()
        dst_carray = dst.create_carray(
            dst_parent, name, atom=atom, shape=shape_tuple, title=title, filters=filters
        )
        # Copy by chunks along axis 0
        chunk = getattr(src_array, "chunkshape", None)
        step = (
            int(chunk[0])
            if chunk is not None and len(chunk) > 0 and chunk[0]
            else 65536
        )
        start = 0
        while start < total:
            stop = min(start + step, total)
            dst_carray[start:stop] = src_array[start:stop]
            start = stop
        _copy_attrs(dst_carray, src_array)
        return

    # Sampled copy creates CArray of reduced length along axis 0
    indices = _sample_indices(total, take, seed=None)
    rest_shape: Tuple[int, ...]
    try:
        rest_shape = tuple(shape_obj[1:]) if shape_obj is not None else tuple()
    except TypeError:
        rest_shape = tuple()
    dst_shape = (take,) + rest_shape
    dst_carray = dst.create_carray(
        dst_parent, name, atom=atom, shape=dst_shape, title=title, filters=filters
    )
    for out_idx, src_idx in enumerate(indices):
        dst_carray[out_idx] = src_array[src_idx]
    _copy_attrs(dst_carray, src_array)


def _copy_leaf(
    dst: tb.File,
    src: tb.File,
    leaf: tb.Leaf,
    dst_parent: tb.Group,
    *,
    take: int,
) -> None:
    """Dispatch copy/sampling for a single leaf node based on its type."""

    if isinstance(leaf, tb.Table):
        _copy_or_sample_table(dst, src, leaf, dst_parent, take=take)
        return

    if isinstance(leaf, (tb.Array, tb.CArray, tb.EArray)):
        # 0D arrays: just copy as scalar attribute-like value into a 0D CArray
        shape_obj = getattr(leaf, "shape", None)
        if not shape_obj or len(shape_obj) == 0:
            name = leaf._v_name
            title = leaf._v_title
            atom = getattr(leaf, "atom", None)
            filters = getattr(leaf, "filters", None)
            dst_carray = dst.create_carray(
                dst_parent, name, atom=atom, shape=(), title=title, filters=filters
            )
            dst_carray[()] = leaf.read()
            _copy_attrs(dst_carray, leaf)
            return
        _copy_or_sample_array(dst, src, leaf, dst_parent, take=take)
        return

    # Unsupported leaf types: skip with a warning
    logger.warning("Skipping unsupported leaf node type at %s", leaf._v_pathname)


def _execute_sampling_for_file(
    src_file: Path,
    dst_file: Path,
    *,
    ratio_est: float,
    min_rows_per_node: int,
    max_rows_per_node: Optional[int],
    seed: Optional[int],
) -> None:
    """Produce a sampled HDF5 copy from ``src_file`` into ``dst_file``."""

    random_seed = seed

    # Open files
    with tb.open_file(str(src_file), mode="r") as src:
        # Prepare dst
        if dst_file.exists():
            dst_file.unlink()
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        with tb.open_file(str(dst_file), mode="w") as dst:
            # Copy root attributes
            _copy_attrs(dst.root, src.root)

            # Plan nodes for this ratio
            plan = _plan_nodes(
                src,
                ratio_est,
                min_rows_per_node=min_rows_per_node,
                max_rows_per_node=max_rows_per_node,
            )

            # Traverse leaves and copy/sample
            for leaf in src.walk_nodes("/", classname="Leaf"):
                total_opt = _leaf_total_rows(leaf)  # type: ignore[arg-type]
                if total_opt is None:
                    # For truly unsupported node types, attempt a raw copy via Array if possible
                    logger.debug(
                        "Skipping non-array/table leaf at %s", leaf._v_pathname
                    )
                    continue

                # Resolve parent group in destination
                parent_path = cast(str, leaf._v_parent._v_pathname)
                dst_group = _ensure_group(dst, src, parent_path)

                # Determine take
                node_plan = plan.get(cast(str, leaf._v_pathname))
                take = node_plan.planned_take if node_plan else 0

                # Seed per leaf for deterministic variety when provided
                if random_seed is not None:
                    random.seed((hash(leaf._v_pathname) ^ random_seed) & 0xFFFFFFFF)

                _copy_leaf(dst, src, cast(tb.Leaf, leaf), dst_group, take=take)


def _sample_file_to_target(
    src_file: Path,
    dst_file: Path,
    *,
    target_bytes: Optional[int],
    ratio: Optional[float],
    min_rows_per_node: int,
    max_rows_per_node: Optional[int],
    max_iterations: int,
    seed: Optional[int],
) -> None:
    """Sample a single HDF5 file with an approximate per-file size target.

    When ``target_bytes`` is provided and ``ratio`` is None, adjusts the sampling
    ratio iteratively for this file until the output size is <= target (or until
    ``max_iterations`` attempts have been made). If ``ratio`` is provided, the
    file is sampled once with that fixed ratio.
    """

    # Initial ratio estimate
    if ratio is not None:
        ratio_est = ratio
    else:
        try:
            src_size = src_file.stat().st_size
        except FileNotFoundError:
            src_size = 0
        if target_bytes is None:
            ratio_est = 1.0
        elif src_size == 0:
            ratio_est = 1.0
        else:
            ratio_est = min(1.0, max(0.0001, target_bytes / src_size))

    attempt = 0
    while True:
        attempt += 1

        # Ensure fresh destination for this attempt
        if dst_file.exists():
            try:
                dst_file.unlink()
            except FileNotFoundError:
                pass
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Creating HDF5 sample for %s (attempt %d) with ratio ~ %.4f",
            src_file,
            attempt,
            ratio_est,
        )

        _execute_sampling_for_file(
            src_file,
            dst_file,
            ratio_est=ratio_est,
            min_rows_per_node=min_rows_per_node,
            max_rows_per_node=max_rows_per_node,
            seed=seed,
        )

        # If using fixed ratio or no target, we are done
        if target_bytes is None or ratio is not None:
            break

        # Evaluate output size for this file
        try:
            out_size = dst_file.stat().st_size
        except FileNotFoundError:
            out_size = 0

        logger.info(
            "Output HDF5 size for %s: %.2f MB (target %.2f MB)",
            src_file,
            out_size / (1024 * 1024),
            (target_bytes or 0) / (1024 * 1024),
        )

        if out_size <= (target_bytes or 0) or attempt >= max_iterations:
            if target_bytes is not None and out_size > target_bytes:
                logger.warning(
                    "Sample for %s remains above target after %d attempts (%.2f MB > %.2f MB). Keeping latest sample.",
                    src_file,
                    attempt,
                    out_size / (1024 * 1024),
                    (target_bytes or 0) / (1024 * 1024),
                )
            break

        # Adjust ratio downward conservatively based on this file's result
        shrink_factor = (target_bytes or 0) / out_size if out_size > 0 else 0.5
        ratio_est = max(0.00005, min(1.0, ratio_est * shrink_factor * 0.95))


def _log_output_overview(output_root: Path) -> None:
    """Log overview of sampled HDF5 outputs (files and sizes)."""

    try:
        if output_root.is_file():
            files = [output_root]
        elif output_root.is_dir():
            files = sorted(
                [
                    p
                    for p in output_root.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".h5", ".hdf5", ".hdf"}
                ]
            )
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
            size_mb = (f.stat().st_size / (1024 * 1024)) if f.exists() else 0.0
            logger.info("  - %s: ~%.2f MB", f.relative_to(output_root.parent), size_mb)
    except Exception as exc:
        logger.warning("Failed to build HDF5 output overview: %s", exc)


def sample_hdf5(
    source: Path,
    output: Path,
    *,
    target_size_mb: Optional[float],
    ratio: Optional[float],
    min_rows_per_node: int,
    max_rows_per_node: Optional[int],
    max_iterations: int,
    recursive: bool,
    seed: Optional[int],
) -> None:
    """Create sampled copies of HDF5 file(s).

    If ``source`` is a directory, recursively samples all HDF5 files (unless
    ``recursive`` is False). Either samples by a fixed ratio or iteratively
    adjusts the ratio to reach an approximate target output size in megabytes.
    """

    if not source.exists():
        raise FileNotFoundError(f"Source path not found: {source}")
    if target_size_mb is None and ratio is None:
        raise ValueError("Either --target-size-mb or --ratio must be provided")
    if ratio is not None and not (0.0 < ratio <= 1.0):
        raise ValueError("--ratio must be in (0, 1]")
    if max_rows_per_node is not None and max_rows_per_node < 1:
        raise ValueError("--max-rows-per-node must be >= 1 when provided")

    files = _list_hdf5_files(source, recursive=recursive)
    if not files:
        raise ValueError("No HDF5 files found in source")

    normalized_output, output_is_dir = _ensure_output_location(source, output)

    desired_bytes = (
        None if target_size_mb is None else int(target_size_mb * 1024 * 1024)
    )

    # Per-file target mode: when a target size is provided and no fixed ratio
    if desired_bytes is not None and ratio is None:
        # Clean previous outputs
        _remove_output_tree(normalized_output)
        if output_is_dir:
            normalized_output.mkdir(parents=True, exist_ok=True)

        for src_file in files:
            if output_is_dir:
                rel = (
                    src_file.name if source.is_file() else src_file.relative_to(source)
                )
                dst_file = normalized_output / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                dst_file = normalized_output

            _sample_file_to_target(
                src_file,
                dst_file,
                target_bytes=desired_bytes,
                ratio=None,
                min_rows_per_node=min_rows_per_node,
                max_rows_per_node=max_rows_per_node,
                max_iterations=max_iterations,
                seed=seed,
            )

            if not output_is_dir:
                break
    else:
        # Fixed ratio (or no target) mode
        ratio_est = ratio if ratio is not None else 1.0

        # Clean previous outputs
        _remove_output_tree(normalized_output)
        if output_is_dir:
            normalized_output.mkdir(parents=True, exist_ok=True)

        logger.info("Creating HDF5 sample with fixed ratio %.4f", ratio_est)

        for src_file in files:
            if output_is_dir:
                rel = (
                    src_file.name if source.is_file() else src_file.relative_to(source)
                )
                dst_file = normalized_output / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                dst_file = normalized_output

            _execute_sampling_for_file(
                src_file,
                dst_file,
                ratio_est=ratio_est,
                min_rows_per_node=min_rows_per_node,
                max_rows_per_node=max_rows_per_node,
                seed=seed,
            )

            if not output_is_dir:
                break

    # Final overview
    _log_output_overview(normalized_output if output_is_dir else normalized_output)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the HDF5 sampler CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Create randomly-sampled copies of HDF5 file(s) while preserving group structure and attributes."
        )
    )

    parser.add_argument(
        "source", type=Path, help="Path to source .h5/.hdf5/.hdf file or directory"
    )
    parser.add_argument(
        "output", type=Path, help="Path to output .h5 file or directory"
    )

    size_group = parser.add_mutually_exclusive_group(required=False)
    size_group.add_argument(
        "--target-size-mb",
        type=float,
        help="Approximate per-file max size of output in MB",
    )
    size_group.add_argument("--ratio", type=float, help="Sampling ratio in (0,1]")

    parser.add_argument(
        "--min-rows-per-node",
        type=int,
        default=1,
        help="Minimum rows/items to include per non-empty leaf node",
    )
    parser.add_argument(
        "--max-rows-per-node",
        type=int,
        default=int(1e6),
        help="Approximate cap of rows/items to include per node (overrides minimum when smaller)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max attempts to adjust ratio to reach target size",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="When source is a directory, do not search recursively",
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
    """CLI entrypoint for creating sampled HDF5 file(s)."""

    args = parse_args(argv)
    configure_logging(args.verbose)

    # Normalize defaults consistent with parse_args
    max_rows_per_node_opt: Optional[int] = args.max_rows_per_node
    if max_rows_per_node_opt is not None and max_rows_per_node_opt <= 0:
        max_rows_per_node_opt = None

    try:
        sample_hdf5(
            source=args.source,
            output=args.output,
            target_size_mb=args.target_size_mb,
            ratio=args.ratio,
            min_rows_per_node=args.min_rows_per_node,
            max_rows_per_node=max_rows_per_node_opt,
            max_iterations=args.max_iterations,
            recursive=not args.non_recursive,
            seed=args.seed,
        )
    except Exception as exc:
        logger.exception("Failed to sample HDF5: %s", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
