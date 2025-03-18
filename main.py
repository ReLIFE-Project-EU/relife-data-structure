import logging
import sqlite3
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from ydata_profiling import ProfileReport

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            markup=True,
            console=console,
        )
    ],
)

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


def read_data_file(file_path: Path) -> Dict[str, pd.DataFrame]:
    """Read a data file based on its extension."""

    file_extension = file_path.suffix.lower()

    readers = {
        ".sqlite": read_sqlite_file,
        ".csv": read_csv_file,
        ".parquet": read_parquet_file,
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

            profile = ProfileReport(df, title="Profile Report - %s" % table_name)
            output_file = output_dir / f"{table_name}_profile.html"
            profile.to_file(output_file)
            progress.update(task, description=f"Generated report for {table_name}")
            logger.info("Report saved to: %s", output_file)


def find_data_files(directory: Path, extensions: List[str]) -> List[Path]:
    """Find all data files with the specified extensions in the directory."""

    files = []

    for ext in extensions:
        files.extend(list(directory.glob("*%s" % ext)))

    return files


def main():
    # Define paths
    samples_dir = Path("samples")
    output_dir = Path("reports")

    # Supported file extensions
    supported_extensions = [".sqlite", ".csv", ".parquet"]

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
            file_output_dir = output_dir / data_file.stem
            generate_profiling_reports(dataframes, file_output_dir)

            logger.info("Successfully processed %s", data_file)

        except Exception as e:
            logger.error("Error processing %s: %s", data_file, str(e), exc_info=True)


if __name__ == "__main__":
    main()
