"""
EDA Report Consolidator Package

A system for consolidating and analyzing YData Profiling JSON reports
from multiple datasets to generate comprehensive consolidated reports.
"""

__version__ = "1.0.0"
__author__ = "EDA Report Consolidator"

from .models import (
    ReportFile,
    ProfileData,
    TableStats,
    VariableStats,
    QualityIssue,
    ConsolidatorConfig,
    QualityThresholds,
    ConsolidationResult,
)
from .analyzer import DataAnalyzer
from .orchestrator import ConsolidationOrchestrator
from .error_handler import ConsolidatorErrorHandler, setup_error_logging

__all__ = [
    "ReportFile",
    "ProfileData", 
    "TableStats",
    "VariableStats",
    "QualityIssue",
    "ConsolidatorConfig",
    "QualityThresholds",
    "ConsolidationResult",
    "DataAnalyzer",
    "ConsolidationOrchestrator",
    "ConsolidatorErrorHandler",
    "setup_error_logging",
]