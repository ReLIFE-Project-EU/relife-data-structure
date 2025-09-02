#!/usr/bin/env python3
"""
EDA Report Consolidator - Main Entry Point

This script provides a simple, configuration-free entry point for consolidating
YData Profiling JSON reports into a comprehensive analysis report.

Usage:
    python consolidate_reports.py [options]

The script automatically discovers reports in the 'reports' directory and generates
a timestamped consolidated report with data quality insights and recommendations.
"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console

from consolidator.models import ConsolidatorConfig
from consolidator.orchestrator import ConsolidationOrchestrator
from consolidator.error_handler import setup_error_logging, log_system_info
from logging_setup import configure_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Consolidate YData Profiling reports into a comprehensive analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python consolidate_reports.py
    python consolidate_reports.py --reports-dir /path/to/reports
    python consolidate_reports.py --output consolidated_analysis.md --verbose
    python consolidate_reports.py --no-timestamp --no-progress
        """
    )
    
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory containing YData Profiling JSON reports (default: reports)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consolidated_reports") / "consolidated_report.md",
        help="Output file path for the consolidated report (default: consolidated_report.md)"
    )
    
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Disable timestamp in output filename"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress reporting and visual output"
    )
    
    parser.add_argument(
        "--no-appendix",
        action="store_true",
        help="Exclude detailed appendix from the report"
    )
    
    parser.add_argument(
        "--max-issues",
        type=int,
        default=10,
        help="Maximum number of priority issues to include in main report (default: 10)"
    )
    
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=20.0,
        help="Threshold for flagging high missing data percentage (default: 20.0)"
    )
    
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=1.0,
        help="Threshold for flagging high duplicate percentage (default: 1.0)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="EDA Report Consolidator 1.0.0"
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> ConsolidatorConfig:
    """Create ConsolidatorConfig from command line arguments."""
    from consolidator.models import QualityThresholds
    
    # Create quality thresholds from arguments
    quality_thresholds = QualityThresholds(
        high_missing_data_pct=args.missing_threshold,
        high_duplicate_pct=args.duplicate_threshold
    )
    
    # Create main configuration
    config = ConsolidatorConfig(
        reports_directory=args.reports_dir,
        output_file=args.output,
        quality_thresholds=quality_thresholds,
        timestamp_reports=not args.no_timestamp,
        enable_progress_reporting=not args.no_progress,
        include_detailed_appendix=not args.no_appendix,
        max_priority_issues_in_main_report=args.max_issues
    )
    
    return config


def validate_environment(config: ConsolidatorConfig) -> bool:
    """
    Validate the environment and configuration.
    
    Returns:
        True if environment is valid, False otherwise
    """
    # Check if reports directory exists
    if not config.reports_directory.exists():
        print(f"❌ Error: Reports directory not found: {config.reports_directory}")
        print(f"   Please ensure the directory exists and contains YData Profiling JSON reports.")
        return False
    
    if not config.reports_directory.is_dir():
        print(f"❌ Error: Reports path is not a directory: {config.reports_directory}")
        return False
    
    # Check if reports directory has any files
    json_files = list(config.reports_directory.rglob("*.json"))
    if not json_files:
        print(f"⚠️  Warning: No JSON files found in {config.reports_directory}")
        print(f"   The consolidator will scan for YData Profiling reports, but none may be found.")
    
    # Check if output directory is writable
    try:
        config.output_file.parent.mkdir(parents=True, exist_ok=True)
        test_file = config.output_file.parent / ".write_test"
        test_file.touch()
        test_file.unlink()
    except (PermissionError, OSError) as e:
        print(f"❌ Error: Cannot write to output directory: {config.output_file.parent}")
        print(f"   {str(e)}")
        return False
    
    return True


def display_startup_info(config: ConsolidatorConfig, console: Console) -> None:
    """Display startup information and configuration."""
    if not config.enable_progress_reporting:
        return
    
    console.print()
    console.print("🔍 EDA Report Consolidator", style="bold blue")
    console.print("=" * 50, style="dim")
    console.print(f"Reports Directory: {config.reports_directory}")
    console.print(f"Output File: {config.output_file}")
    console.print(f"Timestamp Reports: {'Yes' if config.timestamp_reports else 'No'}")
    console.print(f"Include Appendix: {'Yes' if config.include_detailed_appendix else 'No'}")
    console.print(f"Max Priority Issues: {config.max_priority_issues_in_main_report}")
    console.print()


def main() -> int:
    """
    Main entry point for the EDA Report Consolidator.
    
    Returns:
        Exit code: 0 for success, 1 for failure
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Configure logging with error handling integration
        configure_logging(verbose=args.verbose)
        logger = logging.getLogger(__name__)
        
        # Log system information for debugging
        if args.verbose:
            log_system_info()
        
        # Create configuration
        config = create_config_from_args(args)
        
        # Create console for output
        console = Console() if config.enable_progress_reporting else Console(quiet=True)
        
        # Display startup information
        display_startup_info(config, console)
        
        # Validate environment
        if not validate_environment(config):
            return 1
        
        # Create and run orchestrator
        orchestrator = ConsolidationOrchestrator(config, console)
        result = orchestrator.run_consolidation()
        
        # Handle results
        if result.success:
            logger.info(f"Consolidation completed successfully. Report saved to: {result.report_path}")
            
            if not config.enable_progress_reporting:
                # Print minimal success message when progress reporting is disabled
                print(f"✅ Consolidation completed. Report saved to: {result.report_path}")
            
            # Log any warnings or errors that occurred
            if result.warnings:
                logger.warning(f"Consolidation completed with {len(result.warnings)} warnings")
                for warning in result.warnings:
                    logger.warning(f"  - {warning}")
            
            if result.errors_encountered:
                logger.warning(f"Consolidation completed with {len(result.errors_encountered)} errors")
                for error in result.errors_encountered:
                    logger.warning(f"  - {error}")
            
            return 0
        else:
            logger.error("Consolidation failed")
            
            if not config.enable_progress_reporting:
                print("❌ Consolidation failed. Check logs for details.")
            
            # Log all errors
            for error in result.errors_encountered:
                logger.error(f"  - {error}")
            
            return 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Consolidation interrupted by user")
        return 1
    
    except Exception as e:
        logger.exception("Unexpected error in main")
        print(f"❌ Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())