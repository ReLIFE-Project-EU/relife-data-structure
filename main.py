import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List

import pandas as pd
from mdutils.mdutils import MdUtils
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
            output_file_html = output_dir / f"{table_name}_profile.html"
            output_file_json = output_dir / f"{table_name}_profile.json"
            profile.to_file(output_file_html)
            profile.to_file(output_file_json)
            progress.update(task, description=f"Generated report for {table_name}")
            logger.info("Report saved to: %s", output_file_html)
            logger.info("JSON report saved to: %s", output_file_json)


def find_data_files(directory: Path, extensions: List[str]) -> List[Path]:
    """Find all data files with the specified extensions in the directory."""

    files = []

    for ext in extensions:
        files.extend(list(directory.glob("*%s" % ext)))

    return files


def generate_markdown_summary_report(output_dir: Path) -> None:
    """Generate a user-friendly Markdown report summarizing overviews and alerts from all JSON profiling reports using mdutils."""

    summary_path = output_dir / "SUMMARY.md"

    md_file = MdUtils(
        file_name=str(summary_path.with_suffix("")), title="Data Profiling Summary"
    )

    md_file.new_header(level=1, title="Data Profiling Summary")
    md_file.new_line(
        "This report provides a summary of all data profiling results. Each section includes key statistics and alerts for a dataset. [HTML reports](#html-reports) are available for detailed exploration."
    )

    # Collect summary stats for high-level overview
    dataset_summaries = []
    dataset_links = []
    total_tables = 0
    total_rows = 0
    total_columns = 0

    # Gather all datasets and their stats
    for dataset_dir in output_dir.iterdir():
        if dataset_dir.is_dir():
            for json_file in dataset_dir.glob("*_profile.json"):
                with open(json_file, "r") as f:
                    try:
                        data = json.load(f)
                    except Exception as e:
                        logger.error(f"Failed to load {json_file}: {e}")
                        continue

                table = data.get("table", {})
                analysis = data.get("analysis", {})
                title = analysis.get("title", json_file.stem)
                n = table.get("n", 0)
                n_var = table.get("n_var", 0)
                dataset_summaries.append(
                    [
                        title,
                        n,
                        n_var,
                        table.get("n_cells_missing", "N/A"),
                        table.get("n_vars_with_missing", "N/A"),
                        table.get("n_duplicates", "N/A"),
                    ]
                )
                total_tables += 1
                total_rows += n if isinstance(n, int) else 0
                total_columns += n_var if isinstance(n_var, int) else 0
                # For TOC and HTML links
                anchor = md_file.new_inline_link(
                    f"#{title.lower().replace(' ', '-')}", title
                )
                dataset_links.append(anchor)

    # High-level summary
    md_file.new_header(level=2, title="High-Level Summary")
    md_file.new_list(
        [
            f"**Total datasets:** {total_tables}",
            f"**Total rows:** {total_rows}",
            f"**Total columns:** {total_columns}",
        ]
    )
    md_file.new_line()

    # Table of Contents
    md_file.new_header(level=2, title="Table of Contents")
    md_file.new_list(dataset_links)
    md_file.new_line()

    # Table for key stats using new_table
    md_file.new_header(level=2, title="Dataset Overview Table")
    overview_headers = [
        "Dataset",
        "Rows",
        "Columns",
        "Missing Cells",
        "Columns w/ Missing",
        "Duplicate Rows",
    ]
    overview_table_flat = []
    for row in dataset_summaries:
        overview_table_flat.extend([str(cell) for cell in row])
    md_file.new_table(
        columns=len(overview_headers),
        rows=len(dataset_summaries) + 1,
        text=overview_headers + overview_table_flat,
        text_align="center",
    )
    md_file.new_line()

    # Detailed sections for each dataset
    for dataset_dir in output_dir.iterdir():
        if dataset_dir.is_dir():
            for json_file in dataset_dir.glob("*_profile.json"):
                with open(json_file, "r") as f:
                    try:
                        data = json.load(f)
                    except Exception as e:
                        logger.error(f"Failed to load {json_file}: {e}")
                        continue

                table = data.get("table", {})
                alerts = data.get("alerts", [])
                analysis = data.get("analysis", {})
                title = analysis.get("title", json_file.stem)
                html_report = dataset_dir / f"{json_file.stem}.html"

                md_file.new_header(level=2, title=f"Summary for Table: {title}")
                md_file.new_line(
                    f"[🔗 View full HTML report]({html_report})  "
                    if html_report.exists()
                    else ""
                )
                md_file.new_line(
                    "This section summarizes the main statistics and alerts for this dataset."
                )
                # Table for stats using new_table
                stats_headers = ["Stat", "Value"]
                stats_rows = [
                    ["Rows", table.get("n", "N/A")],
                    ["Columns", table.get("n_var", "N/A")],
                    ["Missing cells", table.get("n_cells_missing", "N/A")],
                    ["Columns with missing", table.get("n_vars_with_missing", "N/A")],
                    ["Duplicate rows", table.get("n_duplicates", "N/A")],
                    ["Types", table.get("types", {})],
                    ["Analysis started", analysis.get("date_start", "N/A")],
                    ["Analysis ended", analysis.get("date_end", "N/A")],
                ]
                stats_table_flat = [
                    cell for row in ([stats_headers] + stats_rows) for cell in row
                ]
                md_file.new_table(
                    columns=2,
                    rows=len(stats_rows) + 1,
                    text=stats_table_flat,
                    text_align="left",
                )
                md_file.new_line()
                md_file.new_header(level=3, title="Alerts ⚠️")
                md_file.new_line(
                    "Alerts highlight potential issues or noteworthy findings in the data."
                )
                if alerts:
                    md_file.new_list([f"⚠️ {alert}" for alert in alerts])
                else:
                    md_file.new_list(["✅ No alerts."])
                md_file.new_line("\n---\n")

    # HTML Reports section
    md_file.new_header(level=2, title="HTML Reports")
    md_file.new_line(
        "Below are links to the full HTML profiling reports for each dataset, which provide detailed visualizations and statistics."
    )
    html_links = []
    for dataset_dir in output_dir.iterdir():
        if dataset_dir.is_dir():
            for html_file in dataset_dir.glob("*_profile.html"):
                html_links.append(f"[{html_file.stem}]({html_file})")
    if html_links:
        md_file.new_list(html_links)
    md_file.create_md_file()
    logger.info("Markdown summary report saved to: %s", summary_path)


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

    # After all reports are generated, create the Markdown summary
    generate_markdown_summary_report(output_dir)


if __name__ == "__main__":
    main()
