"""
Orchestration module for the EDA Report Consolidator.

This module contains the ConsolidationOrchestrator class that coordinates
all components of the consolidation process and provides progress reporting.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text

from .models import (
    ConsolidatorConfig,
    ConsolidationResult,
    ProcessingError,
    ProfileData,
    QualityIssue,
)
from .scanner import ReportScanner
from .parser import ProfileParser
from .analyzer import DataAnalyzer
from .quality import QualityAssessor
from .generator import ReportGenerator
from .error_handler import ConsolidatorErrorHandler, setup_error_logging


logger = logging.getLogger(__name__)


class ConsolidationOrchestrator:
    """
    Orchestrates the entire EDA report consolidation process.
    
    This class serves as the main coordinator for the consolidation workflow,
    managing all components and providing a unified interface for report generation.
    It implements comprehensive error handling, progress reporting, and workflow
    management to ensure reliable operation across diverse environments.
    
    Workflow Orchestration:
    1. Configuration validation and environment setup
    2. Report discovery and scanning with progress tracking
    3. JSON parsing and data extraction with error recovery
    4. Dataset analysis and pattern identification
    5. Quality assessment and issue prioritization
    6. Report generation with length management
    7. Output writing with error handling
    
    Key Features:
    - End-to-end workflow coordination with comprehensive error handling
    - Rich console progress reporting with visual feedback
    - Memory-efficient processing for large dataset collections
    - Graceful error recovery and partial processing capabilities
    - Detailed execution metrics and performance tracking
    - Configurable processing parameters and thresholds
    
    Error Handling Strategy:
    - Centralized error collection and reporting
    - Graceful degradation when individual files fail
    - Comprehensive logging with context information
    - User-friendly error summaries and recommendations
    - Continuation of processing despite non-critical failures
    
    Progress Reporting:
    - Visual progress bars for each processing stage
    - Real-time status updates and file processing feedback
    - Summary statistics and performance metrics
    - Error and warning counts with categorization
    
    Implements Requirements:
    - 5.1: Execute without manual configuration
    - 5.3: Generate timestamped reports
    - 5.4: Provide clear output indicating success or issues
    - 5.5: Adapt to changes in reports directory structure
    
    Attributes:
        config (ConsolidatorConfig): Configuration settings for the consolidation process
        console (Console): Rich console for progress reporting and user feedback
        error_handler (ConsolidatorErrorHandler): Centralized error handling system
        scanner (ReportScanner): Component for discovering and validating report files
        parser (ProfileParser): Component for parsing YData Profiling JSON files
        analyzer (DataAnalyzer): Component for analyzing dataset collections
        quality_assessor (QualityAssessor): Component for assessing data quality
        report_generator (ReportGenerator): Component for generating consolidated reports
        errors (List[ProcessingError]): Collection of errors encountered during processing
        warnings (List[str]): Collection of warnings generated during processing
        start_time (Optional[float]): Timestamp when processing began
    
    Example:
        >>> config = ConsolidatorConfig(reports_directory=Path("reports"))
        >>> orchestrator = ConsolidationOrchestrator(config, Console())
        >>> result = orchestrator.run_consolidation()
        >>> if result.success:
        ...     print(f"Successfully processed {result.datasets_processed} datasets")
        ... else:
        ...     print(f"Processing failed with {len(result.errors_encountered)} errors")
    """
    
    def __init__(self, config: ConsolidatorConfig, console: Optional[Console] = None):
        """
        Initialize the orchestrator with configuration and components.
        
        Args:
            config: Configuration for the consolidation process
            console: Optional Rich console for progress reporting
        """
        self.config = config
        self.console = console or Console()
        
        # Initialize centralized error handler
        self.error_handler = setup_error_logging(verbose=False)
        
        # Initialize components with shared error handler
        self.scanner = ReportScanner(self.error_handler)
        self.parser = ProfileParser(self.error_handler)
        self.analyzer = DataAnalyzer()
        self.quality_assessor = QualityAssessor(config.quality_thresholds)
        self.report_generator = ReportGenerator(config)
        
        # Track processing state
        self.errors: List[ProcessingError] = []
        self.warnings: List[str] = []
        self.start_time: Optional[float] = None
        
    def run_consolidation(self) -> ConsolidationResult:
        """
        Execute the complete consolidation workflow with comprehensive error handling.
        
        Returns:
            ConsolidationResult with success status, output path, and metrics
        """
        self.start_time = time.time()
        timestamp = datetime.now()
        
        try:
            # Display startup banner
            self._display_startup_banner()
            
            # Validate configuration with error handling
            try:
                self._validate_configuration()
            except Exception as e:
                self.error_handler.handle_unexpected_error(e, "configuration validation")
                return self._create_failure_result(f"Configuration validation failed: {str(e)}", timestamp)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
                transient=False
            ) as progress:
                
                # Step 1: Scan for reports with error handling
                scan_task = progress.add_task("Scanning reports directory...", total=100)
                try:
                    scan_result = self.scanner.scan_reports_directory(self.config.reports_directory)
                    progress.update(scan_task, completed=100)
                    
                    if not scan_result.valid_reports:
                        error_msg = "No valid YData Profiling reports found"
                        if self.error_handler.has_critical_errors():
                            error_msg += f". Critical errors encountered: {len(self.error_handler.errors)}"
                        
                        # For empty directories, still try to generate a report
                        if scan_result.total_files_scanned == 0:
                            # Generate empty report
                            try:
                                from .analyzer import DataAnalyzer
                                from .models import DatasetAnalysis, SizeDistribution, QualitySummary, SchemaAnalysis, ColumnNamingAnalysis
                                
                                empty_analysis = DatasetAnalysis(
                                    total_datasets=0,
                                    analysis_timestamp=datetime.now(),
                                    size_distribution=SizeDistribution(0, 0, 0, 0, 0.0, 0.0),
                                    quality_summary=QualitySummary(0, 0, 0, {}, 100.0),
                                    schema_analysis=SchemaAnalysis(
                                        common_data_types={},
                                        column_naming_analysis=ColumnNamingAnalysis({}, [], {}),
                                        schema_inconsistencies=[],
                                        schema_recommendations=[],
                                        data_type_distribution_by_category={},
                                        unique_column_names=set(),
                                        most_common_columns=[]
                                    ),
                                    schema_patterns=[],
                                    category_breakdown={},
                                    category_insights={},
                                    sample_vs_complete={},
                                    standardization_opportunities=[]
                                )
                                
                                consolidated_report = self.report_generator.generate_consolidated_report(empty_analysis, [])
                                output_path = self._write_report_to_file(consolidated_report, timestamp)
                                
                                execution_time = time.time() - self.start_time
                                self._display_success_summary(output_path, 0, execution_time)
                                
                                return ConsolidationResult(
                                    success=True,
                                    report_path=output_path,
                                    datasets_processed=0,
                                    errors_encountered=[error_msg],
                                    warnings=self.error_handler.warnings + self.warnings,
                                    execution_time=execution_time,
                                    timestamp=timestamp
                                )
                            except Exception as e:
                                pass  # Fall through to failure case
                        
                        return self._create_failure_result(error_msg, timestamp)
                    
                    self._report_scan_results(scan_result)
                    
                except Exception as e:
                    self.error_handler.handle_unexpected_error(e, "directory scanning")
                    return self._create_failure_result(f"Directory scanning failed: {str(e)}", timestamp)
                
                # Step 2: Parse JSON files with comprehensive error handling
                parse_task = progress.add_task("Parsing JSON reports...", total=len(scan_result.valid_reports))
                parsed_datasets = []
                parsing_errors = 0
                
                for i, report_file in enumerate(scan_result.valid_reports):
                    try:
                        parse_result = self.parser.parse_profile_json(report_file.path)
                        
                        if parse_result.success and parse_result.profile_data:
                            parsed_datasets.append(parse_result.profile_data)
                        else:
                            parsing_errors += 1
                            self._handle_parsing_error(report_file.path, parse_result.errors)
                            
                            # If we have partial data, log it for potential future use
                            if parse_result.partial_data:
                                logger.debug(f"Partial data available for {report_file.path}")
                        
                    except MemoryError as e:
                        self.error_handler.handle_memory_error(report_file.path, e, "JSON parsing")
                        logger.critical("Memory error during parsing - stopping processing")
                        break
                        
                    except Exception as e:
                        self.error_handler.handle_unexpected_error(e, "JSON parsing", report_file.path)
                        parsing_errors += 1
                    
                    progress.update(parse_task, completed=i + 1)
                
                if not parsed_datasets:
                    error_msg = f"No datasets could be successfully parsed. Parsing errors: {parsing_errors}"
                    
                    # Try to generate an empty report for graceful handling
                    try:
                        from .analyzer import DataAnalyzer
                        from .models import DatasetAnalysis, SizeDistribution, QualitySummary, SchemaAnalysis, ColumnNamingAnalysis
                        
                        empty_analysis = DatasetAnalysis(
                            total_datasets=0,
                            analysis_timestamp=datetime.now(),
                            size_distribution=SizeDistribution(0, 0, 0, 0, 0.0, 0.0),
                            quality_summary=QualitySummary(0, 0, 0, {}, 100.0),
                            schema_analysis=SchemaAnalysis(
                                common_data_types={},
                                column_naming_analysis=ColumnNamingAnalysis({}, [], {}),
                                schema_inconsistencies=[],
                                schema_recommendations=[],
                                data_type_distribution_by_category={},
                                unique_column_names=set(),
                                most_common_columns=[]
                            ),
                            schema_patterns=[],
                            category_breakdown={},
                            category_insights={},
                            sample_vs_complete={},
                            standardization_opportunities=[]
                        )
                        
                        consolidated_report = self.report_generator.generate_consolidated_report(empty_analysis, [])
                        output_path = self._write_report_to_file(consolidated_report, timestamp)
                        
                        execution_time = time.time() - self.start_time
                        
                        return ConsolidationResult(
                            success=False,
                            report_path=output_path,
                            datasets_processed=0,
                            errors_encountered=[error_msg],
                            warnings=self.error_handler.warnings + self.warnings,
                            execution_time=execution_time,
                            timestamp=timestamp
                        )
                    except Exception:
                        return self._create_failure_result(error_msg, timestamp)
                
                logger.info(f"Successfully parsed {len(parsed_datasets)} datasets, "
                           f"{parsing_errors} parsing errors")
                
                # Step 3: Analyze datasets with error handling
                analysis_task = progress.add_task("Analyzing dataset collection...", total=100)
                try:
                    dataset_analysis = self.analyzer.analyze_dataset_collection(parsed_datasets)
                    progress.update(analysis_task, completed=100)
                except Exception as e:
                    self.error_handler.handle_unexpected_error(e, "dataset analysis")
                    return self._create_failure_result(f"Dataset analysis failed: {str(e)}", timestamp)
                
                # Step 4: Assess quality with error handling
                quality_task = progress.add_task("Assessing data quality...", total=100)
                try:
                    quality_issues = self.quality_assessor.flag_quality_issues(parsed_datasets)
                    progress.update(quality_task, completed=100)
                except Exception as e:
                    self.error_handler.handle_unexpected_error(e, "quality assessment")
                    return self._create_failure_result(f"Quality assessment failed: {str(e)}", timestamp)
                
                # Step 5: Generate report with error handling
                report_task = progress.add_task("Generating consolidated report...", total=100)
                try:
                    consolidated_report = self.report_generator.generate_consolidated_report(
                        dataset_analysis, quality_issues
                    )
                    progress.update(report_task, completed=100)
                except Exception as e:
                    self.error_handler.handle_unexpected_error(e, "report generation")
                    return self._create_failure_result(f"Report generation failed: {str(e)}", timestamp)
                
                # Step 6: Write output with error handling
                output_task = progress.add_task("Writing output file...", total=100)
                try:
                    output_path = self._write_report_to_file(consolidated_report, timestamp)
                    progress.update(output_task, completed=100)
                except Exception as e:
                    self.error_handler.handle_file_system_error(self.config.output_file, e, "report writing")
                    return self._create_failure_result(f"Failed to write report: {str(e)}", timestamp)
            
            # Display success summary with error information
            execution_time = time.time() - self.start_time
            self._display_success_summary(output_path, len(parsed_datasets), execution_time)
            
            # Collect all errors from error handler
            all_errors = [error.message for error in self.error_handler.errors] + [error.message for error in self.errors]
            all_warnings = self.error_handler.warnings + self.warnings
            
            return ConsolidationResult(
                success=True,
                report_path=output_path,
                datasets_processed=len(parsed_datasets),
                errors_encountered=all_errors,
                warnings=all_warnings,
                execution_time=execution_time,
                timestamp=timestamp
            )
            
        except MemoryError as e:
            self.error_handler.handle_memory_error(None, e, "consolidation process")
            self._display_error_message(f"Memory error: {str(e)}")
            return self._create_failure_result(f"Memory error: {str(e)}", timestamp)
            
        except Exception as e:
            self.error_handler.handle_unexpected_error(e, "consolidation process")
            logger.exception("Unexpected error during consolidation")
            self._display_error_message(f"Unexpected error: {str(e)}")
            return self._create_failure_result(str(e), timestamp)
    
    def _validate_configuration(self) -> None:
        """Validate the configuration and reports directory with comprehensive error handling."""
        # Validate reports directory
        try:
            if not self.config.reports_directory.exists():
                # For tests, we want to handle this gracefully
                self.error_handler.handle_directory_error(self.config.reports_directory, 
                    FileNotFoundError(f"Directory not found: {self.config.reports_directory}"))
                raise FileNotFoundError(f"Reports directory not found: {self.config.reports_directory}")
            
            if not self.config.reports_directory.is_dir():
                raise NotADirectoryError(f"Reports path is not a directory: {self.config.reports_directory}")
                
        except (PermissionError, OSError) as e:
            self.error_handler.handle_directory_error(self.config.reports_directory, e)
            raise
        
        # Ensure output directory exists with error handling
        try:
            self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            self.error_handler.handle_file_system_error(self.config.output_file.parent, e, "output directory creation")
            raise
        
        # Validate output file path is writable
        try:
            # Test write access by creating a temporary file
            test_file = self.config.output_file.parent / ".write_test"
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            self.error_handler.handle_file_system_error(self.config.output_file.parent, e, "output directory write test")
            raise PermissionError(f"Cannot write to output directory: {self.config.output_file.parent}")
    
    def _display_startup_banner(self) -> None:
        """Display startup banner with configuration info."""
        if not self.config.enable_progress_reporting:
            return
            
        banner_text = Text()
        banner_text.append("EDA Report Consolidator\n", style="bold blue")
        banner_text.append(f"Reports Directory: {self.config.reports_directory}\n", style="dim")
        banner_text.append(f"Output File: {self.config.output_file}\n", style="dim")
        banner_text.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
        
        panel = Panel(
            banner_text,
            title="Starting Consolidation",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()
    
    def _report_scan_results(self, scan_result) -> None:
        """Report the results of the directory scan."""
        if not self.config.enable_progress_reporting:
            return
            
        self.console.print(f"✅ Found {len(scan_result.valid_reports)} valid reports")
        
        if scan_result.invalid_files:
            self.console.print(f"⚠️  Skipped {len(scan_result.invalid_files)} invalid files", style="yellow")
        
        if scan_result.categories_found:
            self.console.print("📁 Categories discovered:")
            for category, count in scan_result.categories_found.items():
                self.console.print(f"   • {category}: {count} datasets")
        
        self.console.print()
    
    def _handle_parsing_error(self, file_path: Path, errors: List[str]) -> None:
        """Handle and log parsing errors."""
        error_msg = f"Failed to parse {file_path.name}: {'; '.join(errors)}"
        logger.warning(error_msg)
        
        self.errors.append(ProcessingError(
            error_type="parsing_error",
            file_path=file_path,
            message=error_msg,
            exception=None
        ))
    
    def _write_report_to_file(self, consolidated_report, timestamp: datetime) -> Path:
        """Write the consolidated report to the output file with error handling."""
        output_path = self.config.output_file
        
        # Add timestamp to filename if configured
        if self.config.timestamp_reports:
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            stem = output_path.stem
            suffix = output_path.suffix
            output_path = output_path.parent / f"{stem}_{timestamp_str}{suffix}"
        
        # Write the report with comprehensive error handling
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(consolidated_report.main_report)
                
                # Add error summary to the report (Requirement 1.4, 2.5)
                error_summary = self.error_handler.create_error_summary()
                if error_summary and error_summary != "No errors or warnings encountered during processing.":
                    f.write("\n\n---\n\n")
                    f.write(error_summary)
                
                if self.config.include_detailed_appendix and consolidated_report.detailed_appendix:
                    f.write("\n\n---\n\n")
                    f.write(consolidated_report.detailed_appendix)
        
        except (PermissionError, OSError, IOError) as e:
            self.error_handler.handle_file_system_error(output_path, e, "report file writing")
            raise
        except Exception as e:
            self.error_handler.handle_unexpected_error(e, "report file writing", output_path)
            raise
        
        return output_path
    
    def _display_success_summary(self, output_path: Path, datasets_count: int, execution_time: float) -> None:
        """Display success summary with key metrics and error information."""
        if not self.config.enable_progress_reporting:
            return
            
        summary_text = Text()
        summary_text.append("✅ Consolidation completed successfully!\n\n", style="bold green")
        summary_text.append(f"📊 Datasets processed: {datasets_count}\n")
        summary_text.append(f"📄 Report saved to: {output_path}\n")
        summary_text.append(f"⏱️  Execution time: {execution_time:.2f} seconds\n")
        
        # Include error handler statistics
        total_errors = len(self.error_handler.errors) + len(self.errors)
        total_warnings = len(self.error_handler.warnings) + len(self.warnings)
        
        if total_errors > 0:
            summary_text.append(f"⚠️  Errors encountered: {total_errors}\n", style="yellow")
            
            # Show error breakdown
            error_stats = self.error_handler.get_error_statistics()
            if error_stats:
                summary_text.append("   Error breakdown:\n", style="dim")
                for error_type, count in error_stats.items():
                    summary_text.append(f"   • {error_type.replace('_', ' ').title()}: {count}\n", style="dim")
        
        if total_warnings > 0:
            summary_text.append(f"⚠️  Warnings: {total_warnings}\n", style="yellow")
        
        if total_errors == 0 and total_warnings == 0:
            summary_text.append("✨ No errors or warnings encountered!\n", style="green")
        
        panel = Panel(
            summary_text,
            title="Consolidation Complete",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def _display_error_message(self, error_msg: str) -> None:
        """Display error message in a formatted panel."""
        if not self.config.enable_progress_reporting:
            return
            
        error_text = Text()
        error_text.append("❌ Consolidation failed\n\n", style="bold red")
        error_text.append(f"Error: {error_msg}\n")
        
        if self.errors:
            error_text.append(f"\nAdditional errors encountered: {len(self.errors)}")
        
        panel = Panel(
            error_text,
            title="Error",
            border_style="red",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def _create_failure_result(self, error_msg: str, timestamp: datetime) -> ConsolidationResult:
        """Create a ConsolidationResult for failure cases with comprehensive error information."""
        execution_time = time.time() - self.start_time if self.start_time else 0.0
        
        # Collect all errors from both sources
        all_errors = [error_msg] + [error.message for error in self.error_handler.errors] + [error.message for error in self.errors]
        all_warnings = self.error_handler.warnings + self.warnings
        
        return ConsolidationResult(
            success=False,
            report_path=Path(""),
            datasets_processed=0,
            errors_encountered=all_errors,
            warnings=all_warnings,
            execution_time=execution_time,
            timestamp=timestamp
        )
    
    def report_progress(self, stage: str, progress: float) -> None:
        """
        Report progress for a specific stage (for external use).
        
        Args:
            stage: Name of the current processing stage
            progress: Progress value between 0.0 and 1.0
        """
        if self.config.enable_progress_reporting:
            percentage = int(progress * 100)
            self.console.print(f"[{percentage:3d}%] {stage}")
        
        # This method can be overridden by tests for progress tracking
    
    def handle_workflow_errors(self, errors: List[Exception]) -> None:
        """
        Handle workflow errors (for external use).
        
        Args:
            errors: List of exceptions encountered during processing
        """
        for error in errors:
            logger.error(f"Workflow error: {str(error)}")
            self.errors.append(ProcessingError(
                error_type="workflow_error",
                file_path=None,
                message=str(error),
                exception=error
            ))