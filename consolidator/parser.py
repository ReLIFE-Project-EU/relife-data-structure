"""
JSON parsing and data extraction system for YData Profiling reports.

This module implements the ProfileParser class that extracts structured data
from YData Profiling JSON files with robust error handling and partial extraction
capabilities.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import (
    AnalysisInfo,
    ParseResult,
    PartialProfileData,
    ProfileData,
    TableStats,
    VariableStats,
)
from .error_handler import ConsolidatorErrorHandler

logger = logging.getLogger(__name__)


class ProfileParser:
    """
    Parser for YData Profiling JSON files with robust error handling.
    
    This class provides comprehensive parsing capabilities for YData Profiling
    JSON reports, extracting structured data while maintaining resilience to
    malformed or incomplete files. It implements intelligent fallback mechanisms
    to maximize data recovery from problematic files.
    
    Key Features:
    - Robust JSON parsing with comprehensive error handling
    - Partial data extraction when full parsing fails
    - Automatic data type inference and validation
    - Quality issue detection during parsing
    - Memory-efficient processing for large files
    - Detailed logging and error reporting
    
    Extraction Capabilities:
    - Analysis metadata (title, dates, duration)
    - Table-level statistics (rows, columns, memory usage, missing data)
    - Variable-level information (data types, missing percentages, unique values)
    - Data type distributions across all columns
    - Quality flags and issue identification
    
    Error Handling Strategy:
    - Graceful degradation: attempts partial extraction when full parsing fails
    - Default value substitution for missing required fields
    - Comprehensive error logging with file path context
    - Continuation of processing despite individual file failures
    
    Implements Requirements:
    - 2.1: Extract table-level statistics (row count, column count, memory usage)
    - 2.2: Extract data type distribution information
    - 2.3: Identify datasets with data quality issues
    - 2.4: Extract column-level information for key variables
    - 2.5: Handle errors gracefully and report what could be extracted
    
    Attributes:
        required_table_fields (List[str]): Essential fields expected in table section
        required_analysis_fields (List[str]): Essential fields expected in analysis section
        error_handler (ConsolidatorErrorHandler): Centralized error handling system
    
    Example:
        >>> parser = ProfileParser()
        >>> result = parser.parse_profile_json(Path("dataset_profile.json"))
        >>> if result.success:
        ...     print(f"Parsed {result.profile_data.dataset_name}: {result.profile_data.table_stats.n_rows} rows")
        ... else:
        ...     print(f"Parsing failed: {result.errors}")
    """
    
    def __init__(self, error_handler: Optional[ConsolidatorErrorHandler] = None):
        """
        Initialize the ProfileParser.
        
        Args:
            error_handler: Optional error handler for comprehensive error management
        """
        self.required_table_fields = ['n', 'n_var', 'memory_size', 'types']
        self.required_analysis_fields = ['title', 'date_start', 'date_end']
        self.error_handler = error_handler or ConsolidatorErrorHandler(__name__)
    
    def parse_profile_json(self, file_path: Path) -> ParseResult:
        """
        Parse a YData Profiling JSON file and extract structured data.
        
        Args:
            file_path: Path to the JSON file to parse
            
        Returns:
            ParseResult containing success status, extracted data, and any errors
            
        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        errors = []
        warnings = []
        
        try:
            # Load and validate JSON with comprehensive error handling
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except json.JSONDecodeError as e:
                self.error_handler.handle_json_parsing_error(file_path, e, "JSON loading")
                return ParseResult(
                    success=False,
                    profile_data=None,
                    partial_data=None,
                    errors=[f"Invalid JSON format: {str(e)}"],
                    warnings=warnings
                )
            except (FileNotFoundError, PermissionError, OSError, IOError) as e:
                if self.error_handler.handle_file_system_error(file_path, e, "JSON file reading"):
                    return ParseResult(
                        success=False,
                        profile_data=None,
                        partial_data=None,
                        errors=[f"File not found: {str(e)}"],
                        warnings=warnings
                    )
                else:
                    # Critical error
                    return ParseResult(
                        success=False,
                        profile_data=None,
                        partial_data=None,
                        errors=[f"Critical file system error: {str(e)}"],
                        warnings=warnings
                    )
            except MemoryError as e:
                self.error_handler.handle_memory_error(file_path, e, "JSON loading")
                return ParseResult(
                    success=False,
                    profile_data=None,
                    partial_data=None,
                    errors=[f"Memory error loading JSON: {str(e)}"],
                    warnings=warnings
                )
            
            logger.debug(f"Successfully loaded JSON from {file_path}")
            
            # Extract dataset name from file path
            dataset_name = self._extract_dataset_name(file_path)
            
            # Try full extraction first
            try:
                profile_data = self._extract_full_profile(json_data, file_path, dataset_name)
                return ParseResult(
                    success=True,
                    profile_data=profile_data,
                    partial_data=None,
                    errors=errors,
                    warnings=warnings
                )
            
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                # Handle validation/extraction errors
                self.error_handler.handle_validation_error(file_path, e, "full profile extraction")
                logger.warning(f"Full extraction failed for {file_path}: {e}")
                warnings.append(f"Full extraction failed: {str(e)}")
                
                # Attempt partial extraction (Requirement 2.5)
                try:
                    partial_data = self._extract_partial_profile(json_data, file_path, dataset_name, str(e))
                    
                    # Always return partial data when full extraction fails
                    return ParseResult(
                        success=False,
                        profile_data=None,
                        partial_data=partial_data,
                        errors=[str(e)],
                        warnings=warnings
                    )
                except Exception as partial_e:
                    self.error_handler.handle_unexpected_error(partial_e, "partial profile extraction", file_path)
                    return ParseResult(
                        success=False,
                        profile_data=None,
                        partial_data=None,
                        errors=[str(e), f"Partial extraction also failed: {str(partial_e)}"],
                        warnings=warnings
                    )
            
            except Exception as e:
                # Handle unexpected errors during extraction
                self.error_handler.handle_unexpected_error(e, "profile extraction", file_path)
                
                # Still attempt partial extraction as a last resort
                try:
                    partial_data = self._extract_partial_profile(json_data, file_path, dataset_name, str(e))
                    return ParseResult(
                        success=False,
                        profile_data=None,
                        partial_data=partial_data,
                        errors=[f"Unexpected error: {str(e)}"],
                        warnings=warnings
                    )
                except Exception as partial_e:
                    return ParseResult(
                        success=False,
                        profile_data=None,
                        partial_data=None,
                        errors=[f"Unexpected error: {str(e)}", f"Partial extraction failed: {str(partial_e)}"],
                        warnings=warnings
                    )
            
        except Exception as e:
            # Handle any other unexpected errors
            self.error_handler.handle_unexpected_error(e, "JSON parsing", file_path)
            return ParseResult(
                success=False,
                profile_data=None,
                partial_data=None,
                errors=[f"Unexpected parsing error: {str(e)}"],
                warnings=warnings
            )
    
    def _extract_full_profile(self, json_data: dict, file_path: Path, dataset_name: str) -> ProfileData:
        """
        Extract complete profile data from JSON.
        
        Args:
            json_data: Parsed JSON data
            file_path: Path to the source file
            dataset_name: Name of the dataset
            
        Returns:
            Complete ProfileData object
            
        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        # Extract analysis info (Requirement 2.1)
        analysis_info = self._extract_analysis_info(json_data)
        
        # Extract table-level statistics (Requirement 2.1)
        table_stats = self._extract_table_stats(json_data)
        
        # Extract variable-level information (Requirement 2.4)
        variables = self._extract_variable_stats(json_data)
        
        # Extract data type distribution (Requirement 2.2)
        data_type_distribution = self._extract_data_type_distribution(json_data)
        
        # Identify quality flags (Requirement 2.3)
        quality_flags = self._identify_quality_issues(json_data)
        
        return ProfileData(
            file_path=file_path,
            dataset_name=dataset_name,
            analysis_info=analysis_info,
            table_stats=table_stats,
            variables=variables,
            data_type_distribution=data_type_distribution,
            quality_flags=quality_flags
        )
    
    def _extract_partial_profile(self, json_data: dict, file_path: Path, 
                                dataset_name: str, error_msg: str) -> PartialProfileData:
        """
        Extract partial data when full extraction fails (Requirement 2.5).
        
        Args:
            json_data: Parsed JSON data
            file_path: Path to the source file
            dataset_name: Name of the dataset
            error_msg: Error message from failed full extraction
            
        Returns:
            PartialProfileData with whatever could be extracted
        """
        extracted_fields = {}
        missing_fields = []
        extraction_errors = [error_msg]
        
        # Try to extract individual components with fallbacks
        try:
            if 'table' in json_data:
                extracted_fields['table'] = self._extract_with_fallbacks(json_data, 'table')
        except Exception as e:
            missing_fields.append('table')
            extraction_errors.append(f"Table extraction failed: {str(e)}")
        
        try:
            if 'analysis' in json_data:
                analysis_data = json_data['analysis']
                if isinstance(analysis_data, dict):
                    extracted_fields['analysis'] = analysis_data
                    # Also extract title separately for easier access
                    if 'title' in analysis_data:
                        extracted_fields['title'] = analysis_data['title']
        except Exception as e:
            missing_fields.append('analysis')
            extraction_errors.append(f"Analysis extraction failed: {str(e)}")
        
        try:
            if 'variables' in json_data:
                # Extract basic variable info without full processing
                variables_basic = {}
                for var_name, var_data in json_data['variables'].items():
                    variables_basic[var_name] = {
                        'type': var_data.get('type', 'Unknown'),
                        'n_missing': var_data.get('n_missing', 0),
                        'memory_size': var_data.get('memory_size', 0)
                    }
                extracted_fields['variables_basic'] = variables_basic
        except Exception as e:
            missing_fields.append('variables')
            extraction_errors.append(f"Variables extraction failed: {str(e)}")
        
        return PartialProfileData(
            file_path=file_path,
            dataset_name=dataset_name,
            extracted_fields=extracted_fields,
            missing_fields=missing_fields,
            extraction_errors=extraction_errors
        )
    
    def _create_profile_from_partial(self, partial_data: PartialProfileData, file_path: Path, dataset_name: str) -> ProfileData:
        """Create a ProfileData object from partial data with defaults."""
        # Extract analysis info with defaults
        analysis_data = partial_data.extracted_fields.get('analysis', {})
        analysis_info = AnalysisInfo(
            title=analysis_data.get('title', 'Unknown Analysis'),
            date_start=datetime.now(),
            date_end=datetime.now()
        )
        
        # Extract table stats with defaults
        table_data = partial_data.extracted_fields.get('table', {})
        table_stats = TableStats(
            n_rows=table_data.get('n', 0),
            n_columns=table_data.get('n_var', 0),
            memory_size=table_data.get('memory_size', 0),
            missing_cells_pct=table_data.get('p_cells_missing', 0.0) * 100,
            duplicate_rows_pct=table_data.get('p_duplicates', 0.0) * 100,
            data_types=table_data.get('types', {}),
            is_sample_dataset=False
        )
        
        # Extract variables with defaults
        variables = []
        variables_data = partial_data.extracted_fields.get('variables_basic', {})
        for var_name, var_info in variables_data.items():
            variables.append(VariableStats(
                name=var_name,
                data_type=var_info.get('type', 'Unknown'),
                missing_pct=0.0,
                unique_values=0,
                memory_size=var_info.get('memory_size', 0)
            ))
        
        return ProfileData(
            file_path=file_path,
            dataset_name=dataset_name,
            analysis_info=analysis_info,
            table_stats=table_stats,
            variables=variables,
            data_type_distribution=table_data.get('types', {}),
            quality_flags=[]
        )
    
    def _extract_analysis_info(self, json_data: dict) -> AnalysisInfo:
        """Extract analysis metadata from JSON with error handling."""
        analysis = json_data['analysis']
        
        # Check if analysis has required fields
        required_fields = ['title', 'date_start', 'date_end']
        missing_fields = [field for field in required_fields if field not in analysis]
        if missing_fields:
            raise KeyError(f"Missing required analysis fields: {missing_fields}")
        
        # Extract title with fallback
        title = analysis.get('title', 'Unknown Analysis')
        if not title or not isinstance(title, str):
            title = 'Unknown Analysis'
        
        # Extract dates with error handling
        try:
            date_start = datetime.fromisoformat(analysis['date_start'])
        except (ValueError, KeyError, TypeError) as e:
            logger.debug(f"Error parsing date_start: {e}, using current time")
            date_start = datetime.now()
        
        try:
            date_end = datetime.fromisoformat(analysis['date_end'])
        except (ValueError, KeyError, TypeError) as e:
            logger.debug(f"Error parsing date_end: {e}, using current time")
            date_end = datetime.now()
        
        return AnalysisInfo(
            title=title,
            date_start=date_start,
            date_end=date_end
        )
    
    def _extract_table_stats(self, json_data: dict) -> TableStats:
        """
        Extract table-level statistics (Requirement 2.1) with error handling.
        
        Args:
            json_data: Parsed JSON data
            
        Returns:
            TableStats object with extracted statistics
        """
        table = json_data['table']
        
        # Check if table has required fields
        required_fields = ['n', 'n_var', 'memory_size']
        missing_fields = [field for field in required_fields if field not in table]
        if missing_fields:
            raise KeyError(f"Missing required table fields: {missing_fields}")
        
        # Extract basic table statistics with fallbacks
        try:
            n_rows = int(table['n'])
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Error extracting n_rows: {e}, using default 0")
            n_rows = 0
        
        try:
            n_columns = int(table['n_var'])
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Error extracting n_columns: {e}, using default 0")
            n_columns = 0
        
        try:
            memory_size = int(table['memory_size'])
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Error extracting memory_size: {e}, using default 0")
            memory_size = 0
        
        # Calculate missing data percentage with error handling
        try:
            missing_cells_pct = float(table.get('p_cells_missing', 0.0)) * 100
        except (ValueError, TypeError) as e:
            logger.debug(f"Error calculating missing_cells_pct: {e}, using default 0.0")
            missing_cells_pct = 0.0
        
        # Calculate duplicate percentage with error handling
        try:
            duplicate_rows_pct = float(table.get('p_duplicates', 0.0)) * 100
        except (ValueError, TypeError) as e:
            logger.debug(f"Error calculating duplicate_rows_pct: {e}, using default 0.0")
            duplicate_rows_pct = 0.0
        
        # Extract data types distribution with error handling
        data_types = table.get('types', {})
        if not isinstance(data_types, dict):
            logger.debug(f"Invalid data_types format: {type(data_types)}, using empty dict")
            data_types = {}
        
        # Determine if this appears to be a sample dataset
        is_sample_dataset = n_rows < 10000 and n_rows > 0  # Simple heuristic
        
        return TableStats(
            n_rows=n_rows,
            n_columns=n_columns,
            memory_size=memory_size,
            missing_cells_pct=missing_cells_pct,
            duplicate_rows_pct=duplicate_rows_pct,
            data_types=data_types,
            is_sample_dataset=is_sample_dataset
        )
    
    def _extract_variable_stats(self, json_data: dict) -> List[VariableStats]:
        """
        Extract variable-level information (Requirement 2.4) with error handling.
        
        Args:
            json_data: Parsed JSON data
            
        Returns:
            List of VariableStats objects
        """
        variables = []
        variables_data = json_data.get('variables', {})
        
        if not isinstance(variables_data, dict):
            logger.debug(f"Invalid variables format: {type(variables_data)}, using empty dict")
            return variables
        
        for var_name, var_data in variables_data.items():
            try:
                if not isinstance(var_data, dict):
                    logger.debug(f"Invalid variable data for {var_name}: {type(var_data)}, skipping")
                    continue
                
                # Calculate missing percentage with error handling
                try:
                    missing_pct = float(var_data.get('p_missing', 0.0)) * 100
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error calculating missing_pct for {var_name}: {e}, using 0.0")
                    missing_pct = 0.0
                
                # Get unique values count with error handling
                try:
                    unique_values = int(var_data.get('n_distinct', 0))
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error extracting unique_values for {var_name}: {e}, using 0")
                    unique_values = 0
                
                # Get memory size with error handling
                try:
                    memory_size = int(var_data.get('memory_size', 0))
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error extracting memory_size for {var_name}: {e}, using 0")
                    memory_size = 0
                
                # Get data type with fallback
                data_type = var_data.get('type', 'Unknown')
                if not isinstance(data_type, str):
                    data_type = str(data_type) if data_type is not None else 'Unknown'
                
                variables.append(VariableStats(
                    name=var_name,
                    data_type=data_type,
                    missing_pct=missing_pct,
                    unique_values=unique_values,
                    memory_size=memory_size
                ))
                
            except Exception as e:
                logger.debug(f"Error processing variable {var_name}: {e}, skipping")
                continue
        
        return variables
    
    def _extract_data_type_distribution(self, json_data: dict) -> Dict[str, int]:
        """
        Extract data type distribution from table.types field (Requirement 2.2).
        
        Args:
            json_data: Parsed JSON data
            
        Returns:
            Dictionary mapping data types to their counts
        """
        table = json_data.get('table', {})
        return table.get('types', {})
    
    def _identify_quality_issues(self, json_data: dict) -> List[str]:
        """
        Identify data quality issues (Requirement 2.3).
        
        Args:
            json_data: Parsed JSON data
            
        Returns:
            List of quality issue descriptions
        """
        quality_flags = []
        table = json_data.get('table', {})
        
        # Check for high missing data
        missing_pct = table.get('p_cells_missing', 0.0) * 100
        if missing_pct > 20.0:
            quality_flags.append(f"High missing data: {missing_pct:.1f}%")
        
        # Check for duplicates
        duplicate_pct = table.get('p_duplicates', 0.0) * 100
        if duplicate_pct > 1.0:
            quality_flags.append(f"High duplicate rate: {duplicate_pct:.1f}%")
        
        # Check for memory concerns (>100MB)
        memory_mb = table.get('memory_size', 0) / (1024 * 1024)
        if memory_mb > 100:
            quality_flags.append(f"Large memory usage: {memory_mb:.1f}MB")
        
        # Check for unusual data type distributions
        types = table.get('types', {})
        total_vars = sum(types.values())
        if total_vars > 0:
            for data_type, count in types.items():
                proportion = count / total_vars
                if proportion > 0.8 and total_vars > 5:
                    quality_flags.append(f"Unusual type distribution: {proportion:.1%} {data_type}")
        
        return quality_flags
    
    def _extract_dataset_name(self, file_path: Path) -> str:
        """
        Extract dataset name from file path.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dataset name derived from the path
        """
        # Use the parent directory name as dataset name
        return file_path.parent.name
    
    def _extract_with_fallbacks(self, json_data: dict, field_path: str) -> Any:
        """
        Extract field with fallback handling for partial extraction.
        
        Args:
            json_data: JSON data to extract from
            field_path: Field path to extract
            
        Returns:
            Extracted field value or None if not found
        """
        try:
            if field_path in json_data:
                return json_data[field_path]
        except (KeyError, TypeError):
            pass
        
        return None
    
    def handle_parsing_errors(self, file_path: Path, error: Exception) -> PartialProfileData:
        """
        Handle parsing errors and create partial data (Requirement 2.5).
        
        Args:
            file_path: Path to the file that failed to parse
            error: The exception that occurred
            
        Returns:
            PartialProfileData with error information
        """
        dataset_name = self._extract_dataset_name(file_path)
        
        return PartialProfileData(
            file_path=file_path,
            dataset_name=dataset_name,
            extracted_fields={},
            missing_fields=['all'],
            extraction_errors=[str(error)]
        )