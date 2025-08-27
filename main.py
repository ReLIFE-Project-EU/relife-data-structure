"""
A script to generate data profiling reports from various types of data files.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from ydata_profiling import ProfileReport

from logging_setup import configure_logging

console = Console()

configure_logging(verbose=False, console=console)

logger = logging.getLogger(__name__)


def read_sqlite_file(file_path: Path) -> Dict[str, pd.DataFrame]:
    """Read all tables from a SQLite database into a dictionary of DataFrames."""

    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Read each table into a DataFrame
    dataframes = {}

    for table in tables:
        table_name = table[0]
        dataframes[table_name] = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    conn.close()

    return dataframes


def read_csv_file(file_path: Path) -> Dict[str, pd.DataFrame]:
    """Read a CSV file into a DataFrame."""

    return {"data": pd.read_csv(file_path)}


def read_parquet_file(file_path: Path) -> Dict[str, pd.DataFrame]:
    """Read a Parquet file into a DataFrame."""

    return {"data": pd.read_parquet(file_path)}


def read_hdf5_file(file_path: Path) -> Dict[str, pd.DataFrame]:
    """Read an HDF5 file into a dictionary of DataFrames.

    Strategy:
    1) Try treating the file as a pandas HDFStore and read each key.
    2) Fallback to PyTables traversal, converting Table and 1D/2D Arrays into DataFrames.
    """

    dataframes: Dict[str, pd.DataFrame] = {}

    # Attempt pandas HDFStore first (handles files created with pandas.to_hdf)
    try:
        with pd.HDFStore(str(file_path), mode="r") as store:  # type: ignore[attr-defined]
            keys = list(store.keys())
            for key in keys:
                try:
                    df = store.select(key)
                except Exception:
                    df = pd.read_hdf(str(file_path), key)
                if isinstance(df, pd.DataFrame):
                    safe_key = key.strip("/").replace("/", "__") or "root"
                    dataframes[safe_key] = df
        if dataframes:
            return dataframes
    except Exception as e:  # Not a pandas HDFStore or failed to read via pandas
        logger.debug("HDF5 not readable via pandas HDFStore: %s", e)

    # Fallback: use PyTables to traverse generic HDF5 structure
    try:
        import tables as tb  # Lazy import to avoid hard dependency at module import

        with tb.open_file(str(file_path), mode="r") as h5:
            for leaf in h5.walk_nodes("/", classname="Leaf"):
                path = str(leaf._v_pathname)
                safe_key = path.strip("/").replace("/", "__") or getattr(
                    leaf, "_v_name", "dataset"
                )

                if isinstance(leaf, tb.Table):
                    try:
                        arr = leaf.read()
                        df = pd.DataFrame.from_records(arr)
                        dataframes[safe_key] = df
                    except Exception as exc:
                        logger.warning("Skipping HDF5 Table %s: %s", path, exc)
                    continue

                if isinstance(leaf, (tb.Array, tb.CArray, tb.EArray)):
                    try:
                        data = leaf.read()
                        # Only convert 1D/2D arrays to DataFrame
                        if getattr(data, "ndim", 0) == 1:
                            col_name = getattr(leaf, "_v_name", "value")
                            df = pd.DataFrame({str(col_name): data})
                            dataframes[safe_key] = df
                        elif getattr(data, "ndim", 0) == 2:
                            ncols = (
                                int(data.shape[1])
                                if data.shape and len(data.shape) > 1
                                else 1
                            )
                            cols = [f"col_{i}" for i in range(ncols)]
                            df = pd.DataFrame(data, columns=cols)
                            dataframes[safe_key] = df
                        else:
                            logger.debug(
                                "Skipping array with ndim=%s at %s",
                                getattr(data, "ndim", None),
                                path,
                            )
                    except Exception as exc:
                        logger.warning("Skipping HDF5 Array %s: %s", path, exc)
                    continue

                # Other leaf types are skipped
        return dataframes
    except Exception as e:
        logger.error("Failed to read HDF5 file %s: %s", file_path, e)
        return {}


def read_excel_file(file_path: Path) -> Dict[str, pd.DataFrame]:
    """Read an Excel (.xlsx) file into a dict of DataFrames keyed by sheet name."""

    dataframes: Dict[str, pd.DataFrame] = {}

    # Use ExcelFile for efficient multi-sheet access and explicit engine
    xls = pd.ExcelFile(file_path, engine="openpyxl")
    for sheet_name in xls.sheet_names:
        dataframes[str(sheet_name)] = pd.read_excel(
            xls, sheet_name=sheet_name, engine="openpyxl"
        )

    # When the workbook is empty (no sheets), return an empty dict
    return dataframes


def read_data_file(file_path: Path) -> Dict[str, pd.DataFrame]:
    """Read a data file based on its extension."""

    file_extension = file_path.suffix.lower()

    readers = {
        ".sqlite": read_sqlite_file,
        ".csv": read_csv_file,
        ".parquet": read_parquet_file,
        ".h5": read_hdf5_file,
        ".hdf5": read_hdf5_file,
        ".hdf": read_hdf5_file,
        ".xlsx": read_excel_file,
    }

    if file_extension not in readers:
        raise ValueError("Unsupported file type: %s" % file_extension)

    logger.debug("Reading %s using %s", file_path, readers[file_extension].__name__)

    return readers[file_extension](file_path)


def generate_profiling_reports(
    dataframes: Dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Generate profiling reports for each DataFrame and save them to files."""

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for table_name, df in dataframes.items():
            task = progress.add_task(
                f"Generating profile report for table: {table_name}", total=None
            )

            # Skip empty DataFrames to avoid profiling errors
            if df is None or getattr(df, "empty", False):
                progress.update(task, description=f"Skipped empty table {table_name}")
                logger.warning("Skipping %s: empty DataFrame", table_name)
                continue

            profile = ProfileReport(df, title="Profile Report - %s" % table_name)
            output_file_html = output_dir / f"{table_name}_profile.html"
            output_file_json = output_dir / f"{table_name}_profile.json"
            profile.to_file(output_file_html)
            profile.to_file(output_file_json)
            progress.update(task, description=f"Generated report for {table_name}")
            logger.info("Report saved to: %s", output_file_html)
            logger.info("JSON report saved to: %s", output_file_json)


def find_data_files(directory: Path, extensions: List[str]) -> List[Path]:
    """Find all data files with the specified extensions.

    Searches for files located directly under the provided directory and
    recursively within any subdirectories.
    """

    files = []

    # Files directly under the directory
    for ext in extensions:
        files.extend(directory.glob("*%s" % ext))

    # Files under any subdirectories (recursive)
    for ext in extensions:
        files.extend(directory.glob("**/*%s" % ext))

    return files


def main():
    # Define paths
    samples_dir = Path("samples")
    output_dir = Path("reports")

    # Supported file extensions
    supported_extensions = [
        ".sqlite",
        ".csv",
        ".parquet",
        ".h5",
        ".hdf5",
        ".hdf",
        ".xlsx",
    ]

    # Find all supported data files
    data_files = find_data_files(samples_dir, supported_extensions)

    if not data_files:
        logger.warning(
            "No supported files found in %s. Supported extensions: %s",
            samples_dir,
            ", ".join(supported_extensions),
        )

        return

    for data_file in data_files:
        logger.info("Processing file: %s", data_file)

        try:
            # Read the data file
            dataframes = read_data_file(data_file)

            # Create a subdirectory for this file's reports
            # Mirror the directory structure under samples/, e.g.:
            # samples/london/CC.csv -> reports/london/CC/
            try:
                rel_path = data_file.relative_to(samples_dir)
                parent_rel = rel_path.parent
            except ValueError:
                # Fallback when the file is not under samples_dir
                parent_rel = Path()

            file_output_dir = output_dir / parent_rel / data_file.stem
            generate_profiling_reports(dataframes, file_output_dir)

            logger.info("Successfully processed %s", data_file)

        except Exception as e:
            logger.error("Error processing %s: %s", data_file, str(e), exc_info=True)


if __name__ == "__main__":
    main()
