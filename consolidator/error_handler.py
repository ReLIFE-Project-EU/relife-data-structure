"""
Comprehensive error handling and logging for the EDA Report Consolidator.

This module provides centralized error handling, logging integration, and error
summary reporting functionality to ensure graceful handling of all error conditions.
"""

import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from .models import ProcessingError


class ConsolidatorErrorHandler:
    """
    Centralized error handler for the EDA Report Consolidator.
    
    Provides comprehensive error handling, logging integration with existing
    logging_setup.py, and error summary reporting capabilities.
    """
    
    def __init__(self, logger_name: str = __name__):
        """
        Initialize the error handler.
        
        Args:
            logger_name: Name for the logger instance
        """
        self.logger = logging.getLogger(logger_name)
        self.errors: List[ProcessingError] = []
        self.warnings: List[str] = []
        
        # Error type counters for summary reporting
        self.error_counts: Dict[str, int] = {}
        
    def handle_file_system_error(self, file_path: Path, error: Exception, 
                                operation: str = "file operation") -> bool:
        """
        Handle file system errors with graceful recovery (Requirement 1.4, 5.5).
        
        Args:
            file_path: Path to the file that caused the error
            error: The exception that occurred
            operation: Description of the operation that failed
            
        Returns:
            True if the error was handled gracefully, False if it's critical
        """
        error_msg = f"{operation} failed for {file_path}: {str(error)}"
        
        if isinstance(error, FileNotFoundError):
            self.logger.warning(f"File not found: {file_path}")
            self._add_error("file_not_found", file_path, error_msg, error)
            return True  # Can continue processing other files
            
        elif isinstance(error, PermissionError):
            self.logger.error(f"Permission denied accessing {file_path}: {str(error)}")
            self._add_error("permission_error", file_path, error_msg, error)
            return True  # Can continue processing other files
            
        elif isinstance(error, OSError):
            self.logger.error(f"OS error accessing {file_path}: {str(error)}")
            self._add_error("os_error", file_path, error_msg, error)
            return True  # Can continue processing other files
            
        elif isinstance(error, IOError):
            self.logger.error(f"IO error with {file_path}: {str(error)}")
            self._add_error("io_error", file_path, error_msg, error)
            return True  # Can continue processing other files
            
        else:
            self.logger.error(f"Unexpected file system error with {file_path}: {str(error)}")
            self._add_error("unexpected_fs_error", file_path, error_msg, error)
            return False  # Might be critical
    
    def handle_json_parsing_error(self, file_path: Path, error: Exception, 
                                 context: Optional[str] = None) -> bool:
        """
        Handle JSON parsing errors with detailed logging (Requirement 2.5).
        
        Args:
            file_path: Path to the JSON file that failed to parse
            error: The parsing exception
            context: Additional context about what was being parsed
            
        Returns:
            True if the error was handled gracefully
        """
        context_str = f" ({context})" if context else ""
        error_msg = f"JSON parsing failed for {file_path}{context_str}: {str(error)}"
        
        if hasattr(error, 'lineno') and hasattr(error, 'colno'):
            # JSONDecodeError with position information
            self.logger.error(f"JSON syntax error in {file_path} at line {error.lineno}, "
                            f"column {error.colno}: {str(error)}")
        else:
            self.logger.error(error_msg)
        
        # Log the full traceback at debug level for detailed diagnostics
        self.logger.debug(f"Full traceback for {file_path}:\n{traceback.format_exc()}")
        
        self._add_error("json_parsing_error", file_path, error_msg, error)
        return True  # Can continue with other files
    
    def handle_validation_error(self, file_path: Path, error: Exception, 
                               field_name: Optional[str] = None) -> bool:
        """
        Handle data validation errors during parsing.
        
        Args:
            file_path: Path to the file with validation issues
            error: The validation exception
            field_name: Name of the field that failed validation
            
        Returns:
            True if the error was handled gracefully
        """
        field_str = f" (field: {field_name})" if field_name else ""
        error_msg = f"Data validation failed for {file_path}{field_str}: {str(error)}"
        
        self.logger.warning(error_msg)
        self._add_error("validation_error", file_path, error_msg, error)
        return True
    
    def handle_memory_error(self, file_path: Optional[Path], error: MemoryError, 
                           operation: str = "processing") -> bool:
        """
        Handle memory errors during processing.
        
        Args:
            file_path: Path to the file being processed (if applicable)
            error: The memory error
            operation: Description of the operation that failed
            
        Returns:
            False as memory errors are typically critical
        """
        error_msg = f"Memory error during {operation}"
        if file_path:
            error_msg += f" of {file_path}"
        error_msg += f": {str(error)}"
        
        self.logger.critical(error_msg)
        self.logger.info("Consider processing files in smaller batches or increasing available memory")
        
        self._add_error("memory_error", file_path, error_msg, error)
        return False  # Memory errors are typically critical
    
    def handle_directory_error(self, directory_path: Path, error: Exception) -> bool:
        """
        Handle directory-related errors (Requirement 5.5).
        
        Args:
            directory_path: Path to the directory that caused the error
            error: The exception that occurred
            
        Returns:
            True if the error was handled gracefully
        """
        if isinstance(error, FileNotFoundError):
            self.logger.error(f"Reports directory not found: {directory_path}")
            self.logger.info("Please ensure the reports directory exists and contains YData Profiling JSON files")
            self._add_error("directory_not_found", directory_path, 
                          f"Directory not found: {directory_path}", error)
            return False  # Critical error - can't proceed without reports directory
            
        elif isinstance(error, NotADirectoryError):
            self.logger.error(f"Path is not a directory: {directory_path}")
            self._add_error("not_a_directory", directory_path, 
                          f"Path is not a directory: {directory_path}", error)
            return False  # Critical error
            
        elif isinstance(error, PermissionError):
            self.logger.error(f"Permission denied accessing directory: {directory_path}")
            self.logger.info("Please check directory permissions and try again")
            self._add_error("directory_permission_error", directory_path, 
                          f"Permission denied: {directory_path}", error)
            return False  # Critical error
            
        else:
            error_msg = f"Directory error with {directory_path}: {str(error)}"
            self.logger.error(error_msg)
            self._add_error("directory_error", directory_path, error_msg, error)
            return False
    
    def handle_empty_directory(self, directory_path: Path) -> None:
        """
        Handle case where reports directory is empty (Requirement 5.5).
        
        Args:
            directory_path: Path to the empty directory
        """
        warning_msg = f"No JSON files found in reports directory: {directory_path}"
        self.logger.warning(warning_msg)
        self.logger.info("Please ensure the directory contains YData Profiling JSON reports")
        self.warnings.append(warning_msg)
    
    def log_parsing_warning(self, file_path: Path, field: str, 
                           default_value: any, reason: str = "") -> None:
        """
        Log warnings for parsing issues with fallback values.
        
        Args:
            file_path: Path to the file with parsing issues
            field: Name of the field that couldn't be parsed
            default_value: Default value being used
            reason: Reason for the parsing issue
        """
        reason_str = f" ({reason})" if reason else ""
        warning_msg = (f"Using default value for field '{field}' in {file_path.name}: "
                      f"{default_value}{reason_str}")
        
        self.logger.debug(warning_msg)
        self.warnings.append(warning_msg)
    
    def handle_unexpected_error(self, error: Exception, context: str = "", 
                               file_path: Optional[Path] = None) -> bool:
        """
        Handle unexpected errors with full logging.
        
        Args:
            error: The unexpected exception
            context: Context description of what was happening
            file_path: File being processed when error occurred (if applicable)
            
        Returns:
            False as unexpected errors are typically critical
        """
        context_str = f" during {context}" if context else ""
        file_str = f" (file: {file_path})" if file_path else ""
        error_msg = f"Unexpected error{context_str}{file_str}: {str(error)}"
        
        self.logger.exception(error_msg)  # This logs the full traceback
        self._add_error("unexpected_error", file_path, error_msg, error)
        return False
    
    def create_error_summary(self) -> str:
        """
        Create a comprehensive error summary for final output (Requirement 1.4, 2.5).
        
        Returns:
            Formatted error summary string
        """
        if not self.errors and not self.warnings:
            return "No errors or warnings encountered during processing."
        
        summary_lines = ["## Processing Summary"]
        
        if self.errors:
            summary_lines.extend([
                f"",
                f"### Errors Encountered ({len(self.errors)} total)",
                ""
            ])
            
            # Group errors by type
            error_by_type = {}
            for error in self.errors:
                error_type = error.error_type
                if error_type not in error_by_type:
                    error_by_type[error_type] = []
                error_by_type[error_type].append(error)
            
            for error_type, error_list in error_by_type.items():
                summary_lines.append(f"**{error_type.replace('_', ' ').title()}** ({len(error_list)} errors):")
                
                for error in error_list[:5]:  # Show first 5 errors of each type
                    file_info = f" - {error.file_path.name}" if error.file_path else ""
                    summary_lines.append(f"  - {error.message}{file_info}")
                
                if len(error_list) > 5:
                    summary_lines.append(f"  - ... and {len(error_list) - 5} more")
                
                summary_lines.append("")
        
        if self.warnings:
            summary_lines.extend([
                f"### Warnings ({len(self.warnings)} total)",
                ""
            ])
            
            # Show first 10 warnings
            for warning in self.warnings[:10]:
                summary_lines.append(f"  - {warning}")
            
            if len(self.warnings) > 10:
                summary_lines.append(f"  - ... and {len(self.warnings) - 10} more warnings")
        
        # Add recommendations
        summary_lines.extend([
            "",
            "### Recommendations",
            ""
        ])
        
        if any(error.error_type == "permission_error" for error in self.errors):
            summary_lines.append("- Check file and directory permissions for the reports directory")
        
        if any(error.error_type == "json_parsing_error" for error in self.errors):
            summary_lines.append("- Verify that JSON files are valid YData Profiling reports")
            summary_lines.append("- Check for corrupted or incomplete JSON files")
        
        if any(error.error_type == "memory_error" for error in self.errors):
            summary_lines.append("- Consider processing files in smaller batches")
            summary_lines.append("- Increase available system memory if possible")
        
        if not any(error.error_type in ["permission_error", "json_parsing_error", "memory_error"] 
                  for error in self.errors):
            summary_lines.append("- Review the detailed error messages above for specific guidance")
        
        return "\n".join(summary_lines)
    
    def get_error_statistics(self) -> Dict[str, int]:
        """
        Get statistics about errors encountered.
        
        Returns:
            Dictionary with error type counts
        """
        stats = {}
        for error in self.errors:
            error_type = error.error_type
            stats[error_type] = stats.get(error_type, 0) + 1
        return stats
    
    def has_critical_errors(self) -> bool:
        """
        Check if any critical errors were encountered.
        
        Returns:
            True if critical errors exist
        """
        critical_error_types = {
            "directory_not_found", "not_a_directory", "directory_permission_error",
            "memory_error", "unexpected_error"
        }
        
        return any(error.error_type in critical_error_types for error in self.errors)
    
    def clear_errors(self) -> None:
        """Clear all recorded errors and warnings."""
        self.errors.clear()
        self.warnings.clear()
        self.error_counts.clear()
    
    def _add_error(self, error_type: str, file_path: Optional[Path], 
                   message: str, exception: Optional[Exception]) -> None:
        """
        Add an error to the internal tracking.
        
        Args:
            error_type: Type/category of the error
            file_path: File associated with the error (if any)
            message: Error message
            exception: The original exception (if any)
        """
        error = ProcessingError(
            error_type=error_type,
            file_path=file_path,
            message=message,
            exception=exception,
            timestamp=datetime.now()
        )
        
        self.errors.append(error)
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1


def setup_error_logging(verbose: bool = False) -> ConsolidatorErrorHandler:
    """
    Set up error logging integration with existing logging_setup.py.
    
    Args:
        verbose: Enable verbose logging
        
    Returns:
        Configured ConsolidatorErrorHandler instance
    """
    # Import and configure logging using existing setup
    try:
        from logging_setup import configure_logging
        configure_logging(verbose=verbose)
    except ImportError:
        # Fallback if logging_setup is not available
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Create and return error handler
    return ConsolidatorErrorHandler()


def log_system_info() -> None:
    """Log system information for debugging purposes."""
    logger = logging.getLogger(__name__)
    
    try:
        import platform
        import sys
        
        logger.debug(f"Python version: {sys.version}")
        logger.debug(f"Platform: {platform.platform()}")
        logger.debug(f"Working directory: {os.getcwd()}")
        
    except Exception as e:
        logger.debug(f"Could not log system info: {e}")