"""
A script to sample a SQLite database so the entire database
does not need to be uploaded to Git for profiling purposes.
"""

import argparse
import logging
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, cast

from rich.console import Console

from logging_setup import configure_logging as _configure_logging

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    """Configure logging for the CLI using shared setup."""

    _configure_logging(verbose=verbose, console=Console())


@dataclass
class ForeignKeyGroup:
    """Represents a single foreign key constraint group.

    For multi-column foreign keys, all column pairs that belong to the same
    constraint are grouped together and share the same parent table.
    """

    parent_table: str
    column_pairs: List[Tuple[str, str]]  # (fk_col_in_child, pk_or_unique_col_in_parent)


def get_user_tables(cursor: sqlite3.Cursor, schema_prefix: str) -> List[str]:
    """Return names of user tables from the given schema.

    System tables (prefixed with ``sqlite_``) are excluded.

    Args:
        cursor: Cursor bound to a connection where the schema is attached.
        schema_prefix: Schema name (e.g., ``main`` or ``src``).

    Returns:
        Sorted list of table names.
    """

    cursor.execute(
        f"""
        SELECT name
        FROM {schema_prefix}.sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )

    return [r[0] for r in cursor.fetchall()]


def get_create_statements(
    cursor: sqlite3.Cursor, obj_types: Tuple[str, ...], schema_prefix: str
) -> List[str]:
    """Fetch CREATE statements for objects of the given types in a schema.

    Args:
        cursor: Cursor bound to the connection.
        obj_types: Tuple of object types (e.g., ("table",), ("index",)).
        schema_prefix: Schema name (e.g., ``src``).

    Returns:
        Ordered list of SQL CREATE statements.
    """

    placeholders = ",".join(["?"] * len(obj_types))

    cursor.execute(
        f"""
        SELECT sql
        FROM {schema_prefix}.sqlite_master
        WHERE type IN ({placeholders})
          AND name NOT LIKE 'sqlite_%'
          AND sql IS NOT NULL
        ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1 ELSE 2 END, name
        """,
        obj_types,
    )

    return [r[0] for r in cursor.fetchall()]


def get_table_columns(
    cursor: sqlite3.Cursor, schema_prefix: str, table: str
) -> List[str]:
    """Return column names for a table.

    Args:
        cursor: Cursor bound to the connection.
        schema_prefix: Schema name (e.g., ``src``).
        table: Table name.

    Returns:
        List of column names as strings.
    """

    escaped_table = table.replace("'", "''")
    cursor.execute(f"PRAGMA {schema_prefix}.table_info('{escaped_table}')")
    return [row[1] for row in cursor.fetchall()]  # name column


def get_row_count(cursor: sqlite3.Cursor, schema_prefix: str, table: str) -> int:
    """Return total number of rows in the specified table."""

    cursor.execute(f'SELECT COUNT(*) FROM {schema_prefix}."{table}"')
    return int(cursor.fetchone()[0])


def get_foreign_keys(
    cursor: sqlite3.Cursor, schema_prefix: str, table: str
) -> List[ForeignKeyGroup]:
    """Return foreign key constraint groups defined on a table.

    Args:
        cursor: Cursor bound to the connection.
        schema_prefix: Schema name (e.g., ``src``).
        table: Table name.

    Returns:
        List of grouped foreign key constraints (each group may contain
        multiple column pairs for composite keys).
    """

    # Group multi-column FKs by id
    escaped_table = table.replace("'", "''")
    cursor.execute(f"PRAGMA {schema_prefix}.foreign_key_list('{escaped_table}')")
    rows = cursor.fetchall()

    if not rows:
        return []

    groups: Dict[int, Tuple[str, List[Tuple[str, str]]]] = {}

    for row in rows:
        # PRAGMA foreign_key_list returns: (id, seq, table, from, to, on_update, on_delete, match)
        fk_id = row[0]
        parent_table = row[2]
        from_col = row[3]
        to_col = row[4]

        if fk_id not in groups:
            groups[fk_id] = (parent_table, [])

        groups[fk_id][1].append((from_col, to_col))

    result: List[ForeignKeyGroup] = []

    for parent_table, pairs in groups.values():
        # Keep order stable by seq (already grouped by fk_id, but ensure src order)
        result.append(ForeignKeyGroup(parent_table=parent_table, column_pairs=pairs))

    return result


def topological_order(
    tables: List[str], fk_map: Dict[str, List[ForeignKeyGroup]]
) -> List[str]:
    """Return an insertion order that prefers parents before children.

    Attempts a topological sort using the foreign key relationships. In the
    presence of cycles, remaining tables are appended in arbitrary order.
    """

    # Build graph: edge child -> parent
    parents_by_child: Dict[str, Set[str]] = {t: set() for t in tables}
    children_by_parent: Dict[str, Set[str]] = {t: set() for t in tables}

    for child, fks in fk_map.items():
        for g in fks:
            if g.parent_table in parents_by_child:  # only consider in-scope parents
                parents_by_child[child].add(g.parent_table)
                children_by_parent[g.parent_table].add(child)

    # Kahn's algorithm
    no_parent = [t for t, ps in parents_by_child.items() if not ps]
    ordered: List[str] = []

    while no_parent:
        n = no_parent.pop()
        ordered.append(n)
        for m in list(children_by_parent.get(n, [])):
            parents_by_child[m].discard(n)
            if not parents_by_child[m]:
                no_parent.append(m)
        children_by_parent[n].clear()

    # Add remaining (cycles) in any order
    remaining = [t for t in tables if t not in ordered]
    return ordered + remaining


def quote_ident(name: str) -> str:
    """Quote an identifier for safe inclusion in SQL statements."""

    return '"' + name.replace('"', '""') + '"'


def build_fk_condition(alias: str, fk_groups: List[ForeignKeyGroup]) -> Optional[str]:
    """Build a WHERE predicate that preserves FK consistency during sampling.

    For each foreign key group on the child table alias, allows rows where:
    - Any FK column is NULL, or
    - The referenced parent row exists in ``main`` schema.

    Args:
        alias: SQL alias used for the child table in the SELECT.
        fk_groups: Foreign key groups defined on the child table.

    Returns:
        A SQL string for a WHERE clause (without the ``WHERE`` keyword) or
        None if there are no foreign keys.
    """

    if not fk_groups:
        return None

    # For each fk group: (any fk col is NULL) OR EXISTS( join to main parent )
    group_conditions: List[str] = []

    for idx, group in enumerate(fk_groups):
        null_checks = [
            f"{alias}.{quote_ident(fk_col)} IS NULL" for fk_col, _ in group.column_pairs
        ]

        parent_alias = f"p{idx}"

        join_conditions = [
            f"{parent_alias}.{quote_ident(parent_col)} = {alias}.{quote_ident(fk_col)}"
            for fk_col, parent_col in group.column_pairs
        ]

        exists_sql = (
            f"EXISTS (SELECT 1 FROM main.{quote_ident(group.parent_table)} {parent_alias} "
            f"WHERE {' AND '.join(join_conditions)})"
        )

        group_conditions.append(f"(({' OR '.join(null_checks)}) OR {exists_sql})")

    return " AND ".join(group_conditions)


def copy_schema(
    dest: sqlite3.Connection, src_cursor: sqlite3.Cursor, include_indexes: bool
) -> None:
    """Copy schema objects from ``src`` schema into ``main``.

    Copies user tables and, optionally, their indexes. Views and triggers are
    intentionally skipped to keep the sampled database portable.

    Args:
        dest: Destination connection (where ``src`` is attached).
        src_cursor: Cursor bound to ``dest``.
        include_indexes: When True, also creates indexes after tables.
    """

    # Copy tables
    for sql in get_create_statements(src_cursor, ("table",), "src"):
        dest.execute(sql)

    # Optionally copy indexes (not views/triggers to keep it simple and portable)
    if include_indexes:
        for sql in get_create_statements(src_cursor, ("index",), "src"):
            # Skip auto-indexes which sometimes return NULL or malformed SQL
            if sql and sql.strip().upper().startswith("CREATE INDEX"):
                dest.execute(sql)


def vacuum(conn: sqlite3.Connection) -> None:
    """Run VACUUM safely outside transactions."""

    # VACUUM cannot run inside a transaction; temporarily enable autocommit
    original_isolation = conn.isolation_level
    try:
        conn.isolation_level = None  # autocommit mode
        conn.execute("VACUUM")
    finally:
        conn.isolation_level = original_isolation


def set_fast_pragmas(conn: sqlite3.Connection) -> None:
    """Apply PRAGMA settings to speed up bulk inserts for sampling."""

    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-100000")  # ~100MB memory cache
    conn.execute("PRAGMA foreign_keys=OFF")


def insert_sample_for_table(
    dest: sqlite3.Connection,
    table: str,
    take_rows: int,
    fk_groups: List[ForeignKeyGroup],
    *,
    verbose_sql: bool = False,
) -> int:
    """Insert a random sample of rows for a single table.

    Sampling is uniform via ``ORDER BY RANDOM() LIMIT N`` and constrained by
    foreign key predicates to prefer rows whose parents are already present in
    ``main``.

    Args:
        dest: Destination connection (with ``src`` attached).
        table: Table name to sample from.
        take_rows: Number of rows to insert.
        fk_groups: Foreign key groups for the table.
        verbose_sql: When True, logs the generated INSERT..SELECT SQL.

    Returns:
        Number of rows reported inserted by SQLite.
    """

    if take_rows <= 0:
        return 0

    columns = get_table_columns(dest.cursor(), "src", table)

    if not columns:
        return 0

    alias = "s"
    select_cols = ", ".join([f"{alias}.{quote_ident(c)}" for c in columns])
    base_from = f"FROM src.{quote_ident(table)} {alias}"
    where_clause = build_fk_condition(alias, fk_groups)
    where_sql = f"WHERE {where_clause} " if where_clause else ""

    sql = (
        f"INSERT INTO main.{quote_ident(table)} ({', '.join([quote_ident(c) for c in columns])})\n"
        f"SELECT {select_cols} {base_from} {where_sql}ORDER BY RANDOM() LIMIT {take_rows}"
    )

    if verbose_sql:
        logger.debug("SQL for %s:\n%s", table, sql)

    cur = dest.cursor()
    cur.execute(sql)

    return cur.rowcount if cur.rowcount is not None else 0


def sample_sqlite(
    source_db: Path,
    output_db: Path,
    *,
    target_size_mb: Optional[float],
    ratio: Optional[float],
    include_indexes: bool,
    min_rows_per_table: int,
    max_rows_per_table: Optional[int],
    max_iterations: int,
    verbose_sql: bool,
) -> None:
    """Create a sampled copy of a SQLite database.

    Either samples by a fixed ratio or iteratively adjusts the ratio to
    approximate a target output size. Ensures that all tables exist and tries
    to preserve foreign-key consistency between sampled rows.

    Args:
        source_db: Path to source SQLite database file.
        output_db: Path to write the sampled database file.
        target_size_mb: Desired maximum size in megabytes (approximate).
        ratio: Fixed sampling ratio in (0, 1]; ignored when ``target_size_mb``
            is provided.
        include_indexes: Whether to recreate indexes in the sampled database.
        min_rows_per_table: Minimum number of rows to include for non-empty tables.
        max_rows_per_table: Optional approximate cap of rows per table. When set,
            each table's planned rows are further limited to this many. This cap
            may override the minimum when both are provided.
        max_iterations: Max attempts to adjust ratio when targeting size.
        verbose_sql: When True, logs generated SQL statements.
    """

    if not source_db.exists():
        raise FileNotFoundError(f"Source database not found: {source_db}")
    if target_size_mb is None and ratio is None:
        raise ValueError("Either --target-size-mb or --ratio must be provided")
    if ratio is not None and not (0.0 < ratio <= 1.0):
        raise ValueError("--ratio must be in (0, 1]")
    if max_rows_per_table is not None and max_rows_per_table < 1:
        raise ValueError("--max-rows-per-table must be >= 1 when provided")

    src_size_bytes = source_db.stat().st_size

    desired_bytes = (
        None if target_size_mb is None else int(target_size_mb * 1024 * 1024)
    )

    # Initial ratio guess
    if ratio is None:
        ratio_est = min(1.0, max(0.0001, cast(int, desired_bytes) / src_size_bytes))
    else:
        ratio_est = ratio

    attempt = 0
    while True:
        attempt += 1
        if output_db.exists():
            output_db.unlink()

        logger.info(
            "Creating sample (attempt %d) with ratio ~ %.4f", attempt, ratio_est
        )

        dest = sqlite3.connect(output_db)
        try:
            set_fast_pragmas(dest)
            dest.execute(f"ATTACH DATABASE {quote_string(str(source_db))} AS src")

            src_cur = dest.cursor()  # Using dest connection with attached src
            tables = get_user_tables(src_cur, "src")

            # Build FK map for all tables
            fk_map: Dict[str, List[ForeignKeyGroup]] = {
                t: get_foreign_keys(src_cur, "src", t) for t in tables
            }

            # Copy schema first
            copy_schema(
                dest, src_cur, include_indexes=False
            )  # Indexes later, after inserts

            # Determine sampling order
            order = topological_order(tables, fk_map)

            # Compute planned row counts
            planned_counts: Dict[str, Tuple[int, int]] = (
                {}
            )  # table -> (source_count, take)
            for t in tables:
                total = get_row_count(src_cur, "src", t)
                take = min(
                    total,
                    max(
                        min_rows_per_table if total > 0 else 0,
                        math.ceil(total * ratio_est),
                    ),
                )
                if max_rows_per_table is not None:
                    take = min(take, max_rows_per_table)
                planned_counts[t] = (total, take)

            # Insert per order
            for t in order:
                total, take = planned_counts[t]
                if total == 0 or take == 0:
                    logger.debug("%s: no rows to copy", t)
                    continue
                inserted = insert_sample_for_table(
                    dest,
                    t,
                    take,
                    fk_map.get(t, []),
                    verbose_sql=verbose_sql,
                )
                logger.info(
                    "%s: inserted %d / planned %d (source %d)", t, inserted, take, total
                )

            # Optionally create indexes after data load
            if include_indexes:
                for sql in get_create_statements(src_cur, ("index",), "src"):
                    if sql and sql.strip().upper().startswith("CREATE INDEX"):
                        dest.execute(sql)

            # Shrink and flush
            vacuum(dest)
            dest.commit()
        finally:
            dest.close()

        if desired_bytes is None:
            break

        out_size = output_db.stat().st_size
        logger.info(
            "Output DB size: %.2f MB (target %.2f MB)",
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

        # Adjust ratio downward
        shrink_factor = desired_bytes / out_size
        # Be a bit conservative to avoid oscillation
        ratio_est = max(0.00005, min(1.0, ratio_est * shrink_factor * 0.95))

    # Final overview of the resulting sample
    _log_output_overview(output_db)


def _safe_fetchone_int(cursor: sqlite3.Cursor) -> Optional[int]:
    """Fetch a single integer from a cursor."""

    row = cursor.fetchone()

    if not row:
        return None

    try:
        return int(row[0]) if row[0] is not None else None
    except Exception:
        return None


def _log_output_overview(db_path: Path) -> None:
    """Log an overview of the resulting SQLite sample.

    Shows number of tables, total size, and per-table row counts and sizes
    (when the `dbstat` virtual table is available).
    """

    try:
        conn = sqlite3.connect(db_path)
    except Exception as exc:
        logger.warning("Could not open output DB for overview: %s", exc)
        return

    try:
        cur = conn.cursor()

        # Total file size via PRAGMA for portability
        cur.execute("PRAGMA page_size")
        page_size = _safe_fetchone_int(cur) or 0
        cur.execute("PRAGMA page_count")
        page_count = _safe_fetchone_int(cur) or 0
        total_bytes = page_size * page_count

        # Tables list from main schema
        tables = get_user_tables(cur, "main")
        num_tables = len(tables)

        # Attempt per-table sizes via dbstat; gracefully degrade if unavailable
        dbstat_available = True
        try:
            cur.execute("SELECT 1 FROM dbstat LIMIT 1")
            _ = cur.fetchone()
        except sqlite3.OperationalError:
            dbstat_available = False

        table_summaries: List[Tuple[str, int, Optional[int]]] = []
        for t in tables:
            # Row count
            rows = get_row_count(cur, "main", t)

            # Size in bytes (if dbstat is available)
            size_bytes: Optional[int] = None
            if dbstat_available:
                try:
                    cur.execute("SELECT sum(pgsize) FROM dbstat WHERE name = ?", (t,))
                    size_bytes = _safe_fetchone_int(cur)
                except sqlite3.OperationalError:
                    # Some SQLite builds may not expose pgsize; ignore sizing gracefully
                    size_bytes = None

            table_summaries.append((t, rows, size_bytes))

        # Sort tables by size desc when available, otherwise by rows desc
        def _sort_key(item: Tuple[str, int, Optional[int]]):
            name, rows, size_opt = item
            return (-(size_opt or -1), -rows, name)

        table_summaries.sort(key=_sort_key)

        # Header
        total_mb = total_bytes / (1024 * 1024) if total_bytes else 0.0
        logger.info(
            "Overview of output sample: %d tables, total size %.2f MB",
            num_tables,
            total_mb,
        )

        # Per-table lines
        for name, rows, size_opt in table_summaries:
            if size_opt is not None and total_bytes:
                size_mb = size_opt / (1024 * 1024)
                pct = (size_opt / total_bytes) * 100.0
                logger.info(
                    "  - %s: %d rows, ~%.2f MB (%.1f%%)", name, rows, size_mb, pct
                )
            else:
                logger.info("  - %s: %d rows", name, rows)

        if not dbstat_available:
            logger.info(
                "Per-table size not available (SQLite build without dbstat). Showing row counts only."
            )
    except Exception as exc:
        logger.warning("Failed to build overview: %s", exc)
    finally:
        conn.close()


def quote_string(value: str) -> str:
    """Return a safely quoted SQL string literal for ATTACH statements."""

    return "'" + value.replace("'", "''") + "'"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the sampler CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Create a randomly-sampled copy of a SQLite database while preserving all tables and, best-effort, foreign-key consistency."
        )
    )

    parser.add_argument("source", type=Path, help="Path to source .sqlite file")
    parser.add_argument("output", type=Path, help="Path to output sampled .sqlite file")
    size_group = parser.add_mutually_exclusive_group(required=False)

    size_group.add_argument(
        "--target-size-mb",
        type=float,
        help="Approximate max size of output database in MB",
    )

    size_group.add_argument("--ratio", type=float, help="Sampling ratio in (0,1]")

    parser.add_argument(
        "--min-rows-per-table",
        type=int,
        default=1,
        help="Minimum rows to include per non-empty table",
    )

    parser.add_argument(
        "--max-rows-per-table",
        type=int,
        default=int(1e5),
        help=(
            "Approximate cap of rows to include per table (overrides minimum when smaller)"
        ),
    )

    parser.add_argument(
        "--include-indexes",
        action="store_true",
        help="Copy indexes after sampling (slower, larger output)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max attempts to adjust ratio to reach target size",
    )

    parser.add_argument(
        "--verbose-sql",
        action="store_true",
        help="Log generated SQL statements for debugging",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for creating a sampled SQLite database."""

    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        sample_sqlite(
            source_db=args.source,
            output_db=args.output,
            target_size_mb=args.target_size_mb,
            ratio=args.ratio,
            include_indexes=args.include_indexes,
            min_rows_per_table=args.min_rows_per_table,
            max_rows_per_table=args.max_rows_per_table,
            max_iterations=args.max_iterations,
            verbose_sql=args.verbose_sql,
        )
    except Exception as exc:
        logger.exception("Failed to sample database: %s", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
