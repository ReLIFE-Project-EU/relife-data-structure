"""
Core data models for the EDA Report Consolidator.

This module contains all the data structures used throughout the consolidation
process, from file scanning to report generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ReportFile:
    """Represents a discovered report file with metadata."""
    path: Path
    dataset_name: str
    category: str
    file_size: int
    last_modified: datetime
    is_valid_ydata_profile: bool


@dataclass
class AnalysisInfo:
    """Analysis metadata from YData Profiling reports."""
    title: str
    date_start: datetime
    date_end: datetime


@dataclass
class VariableStats:
    """Statistics for individual variables/columns."""
    name: str
    data_type: str
    missing_pct: float
    unique_values: int
    memory_size: int


@dataclass
class TableStats:
    """Table-level statistics from YData Profiling reports."""
    n_rows: int
    n_columns: int
    memory_size: int
    missing_cells_pct: float
    duplicate_rows_pct: float
    data_types: Dict[str, int]
    is_sample_dataset: bool = False  # Inferred from size patterns


@dataclass
class QualityIssue:
    """Represents a data quality issue identified in a dataset."""
    dataset_name: str
    issue_type: str  # 'missing_data', 'duplicates', 'memory_concern', 'unusual_types'
    severity: str    # 'low', 'medium', 'high', 'critical'
    description: str
    recommendation: str
    affected_columns: List[str]
    metrics: Dict[str, float]


@dataclass
class ProfileData:
    """Complete profile data extracted from a YData Profiling JSON report."""
    file_path: Path
    dataset_name: str
    analysis_info: AnalysisInfo
    table_stats: TableStats
    variables: List[VariableStats]
    data_type_distribution: Dict[str, int]
    quality_flags: List[str]


@dataclass
class ScanResult:
    """Results from scanning the reports directory."""
    valid_reports: List[ReportFile]
    invalid_files: List[Path]
    total_files_scanned: int
    scan_duration: float
    categories_found: Dict[str, int]


@dataclass
class ParseResult:
    """Results from parsing a YData Profiling JSON file."""
    success: bool
    profile_data: Optional[ProfileData]
    partial_data: Optional['PartialProfileData']
    errors: List[str]
    warnings: List[str]


@dataclass
class PartialProfileData:
    """Partial data extracted when full parsing fails."""
    file_path: Path
    dataset_name: str
    extracted_fields: Dict[str, Any]
    missing_fields: List[str]
    extraction_errors: List[str]


@dataclass
class QualityThresholds:
    """Configurable thresholds for data quality assessment."""
    high_missing_data_pct: float = 20.0  # Requirement 4.1
    high_duplicate_pct: float = 1.0      # Requirement 4.2
    large_dataset_rows: int = 1_000_000
    memory_concern_mb: int = 100
    unusual_type_distribution_threshold: float = 0.8
    sample_dataset_row_threshold: int = 10_000


@dataclass
class SchemaPattern:
    """Represents a common schema pattern across datasets."""
    pattern_name: str
    datasets: List[str]
    common_columns: List[str]
    data_types: Dict[str, str]


@dataclass
class SizeDistribution:
    """Distribution of dataset sizes."""
    small_datasets: int  # < 1MB
    medium_datasets: int  # 1MB - 100MB
    large_datasets: int  # > 100MB
    total_memory_usage: int
    average_rows: float
    average_columns: float


@dataclass
class QualitySummary:
    """Summary of data quality across all datasets."""
    datasets_with_issues: int
    total_quality_issues: int
    high_priority_issues: int
    common_issue_types: Dict[str, int]
    overall_quality_score: float


@dataclass
class ColumnNamingAnalysis:
    """Analysis of column naming patterns and inconsistencies."""
    naming_conventions: Dict[str, List[str]]  # Convention type -> column examples
    inconsistent_naming: List[Tuple[str, List[str]]]  # Similar columns with different names
    standardization_suggestions: Dict[str, str]  # Current name -> suggested standard name


@dataclass
class SchemaInconsistency:
    """Represents an inconsistency in schema across datasets."""
    inconsistency_type: str  # 'data_type_mismatch', 'naming_inconsistency', 'missing_columns'
    affected_datasets: List[str]
    description: str
    severity: str
    column_details: Dict[str, Any]


@dataclass
class SchemaRecommendation:
    """Recommendation for schema standardization."""
    recommendation_type: str  # 'standardize_naming', 'unify_data_types', 'add_missing_columns'
    priority: str  # 'high', 'medium', 'low'
    description: str
    affected_datasets: List[str]
    implementation_notes: str


@dataclass
class SchemaAnalysis:
    """Comprehensive schema analysis across all datasets."""
    common_data_types: Dict[str, int]  # Data type -> count across all datasets
    column_naming_analysis: ColumnNamingAnalysis
    schema_inconsistencies: List[SchemaInconsistency]
    schema_recommendations: List[SchemaRecommendation]
    data_type_distribution_by_category: Dict[str, Dict[str, int]]
    unique_column_names: Set[str]
    most_common_columns: List[Tuple[str, int]]


@dataclass
class CategoryInsight:
    """Insights for a specific dataset category."""
    category_name: str
    dataset_count: int
    common_patterns: List[str]
    quality_trends: str
    recommendations: List[str]


@dataclass
class StandardizationOpportunity:
    """Opportunity for schema or data standardization."""
    opportunity_type: str
    affected_datasets: List[str]
    description: str
    potential_benefit: str


@dataclass
class DatasetAnalysis:
    """Complete analysis of the dataset collection."""
    total_datasets: int
    analysis_timestamp: datetime
    size_distribution: SizeDistribution
    quality_summary: QualitySummary
    schema_analysis: SchemaAnalysis
    schema_patterns: List[SchemaPattern]
    category_breakdown: Dict[str, int]
    category_insights: Dict[str, CategoryInsight]
    sample_vs_complete: Dict[str, str]
    standardization_opportunities: List[StandardizationOpportunity]


@dataclass
class QualityAssessment:
    """Quality assessment for a single dataset."""
    dataset_name: str
    overall_score: float  # 0-100
    issues: List[QualityIssue]
    strengths: List[str]
    investigation_priority: str  # 'low', 'medium', 'high'


@dataclass
class DatasetPriority:
    """Priority ranking for dataset attention."""
    dataset_name: str
    priority_score: int
    primary_issues: List[str]
    recommended_actions: List[str]


@dataclass
class ReportSection:
    """A section of the consolidated report."""
    title: str
    content: str
    priority: int
    estimated_reading_time: int


@dataclass
class ConsolidatedReport:
    """The final consolidated report."""
    executive_summary: str
    main_report: str
    detailed_appendix: str
    generation_timestamp: datetime
    estimated_reading_time_minutes: int
    datasets_analyzed: int
    critical_issues_count: int


@dataclass
class ConsolidationResult:
    """Result of the entire consolidation process."""
    success: bool
    report_path: Path
    datasets_processed: int
    errors_encountered: List[str]
    warnings: List[str]
    execution_time: float
    timestamp: datetime


@dataclass
class ConsolidatorConfig:
    """Configuration for the EDA Report Consolidator."""
    reports_directory: Path = Path("reports")  # Requirement 5.1 - no manual configuration needed
    output_file: Path = Path("consolidated_report.md")
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    max_report_length: int = 10000  # characters - supports Requirement 7.5 (10-minute review)
    include_detailed_appendix: bool = True  # Requirement 7.4
    parallel_processing: bool = True
    max_workers: int = 4
    max_priority_issues_in_main_report: int = 10  # Requirement 7.3
    enable_progress_reporting: bool = True  # Requirement 5.4
    timestamp_reports: bool = True  # Requirement 5.3
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.max_report_length <= 0:
            raise ValueError("max_report_length must be positive")
        
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        
        if self.max_priority_issues_in_main_report <= 0:
            raise ValueError("max_priority_issues_in_main_report must be positive")
        
        # Ensure paths are Path objects
        if not isinstance(self.reports_directory, Path):
            self.reports_directory = Path(self.reports_directory)
        
        if not isinstance(self.output_file, Path):
            self.output_file = Path(self.output_file)


# Processing error types for error handling
@dataclass
class ProcessingError:
    """Represents an error that occurred during processing."""
    error_type: str
    file_path: Optional[Path]
    message: str
    exception: Optional[Exception]
    timestamp: datetime = field(default_factory=datetime.now)