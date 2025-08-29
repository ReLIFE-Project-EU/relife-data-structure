"""
Data quality assessment system for EDA Report Consolidator.

This module implements comprehensive data quality assessment including issue detection,
dataset prioritization, and recommendation generation for database administrators.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    DatasetPriority,
    ProfileData,
    QualityAssessment,
    QualityIssue,
    QualityThresholds,
    VariableStats,
)

logger = logging.getLogger(__name__)


class QualityAssessor:
    """
    Assesses data quality across datasets and generates actionable recommendations.
    
    This class provides comprehensive data quality assessment capabilities designed
    specifically for database administrators. It evaluates datasets against configurable
    thresholds and generates prioritized recommendations for data management.
    
    Key Features:
    - Configurable quality thresholds for different issue types
    - Multi-dimensional quality scoring (0-100 scale)
    - Severity-based issue classification (low/medium/high/critical)
    - Actionable recommendations for each identified issue
    - Dataset prioritization for attention and remediation
    
    Quality Assessment Dimensions:
    - Missing Data: Identifies datasets with high percentages of missing values
    - Duplicate Records: Flags datasets with significant duplicate content
    - Memory Usage: Highlights datasets with concerning memory footprints
    - Data Type Distribution: Detects unusual or problematic type patterns
    
    Implements Requirements:
    - 4.1: Flag datasets with high percentages of missing data (>20%)
    - 4.2: Flag datasets with significant duplicate records (>1%)
    - 4.3: Flag datasets with unusual data type distributions
    - 4.4: Flag datasets with memory usage concerns
    - 6.1: Provide recommendations for flagged datasets
    - 6.2: Suggest which datasets might benefit from further investigation
    
    Attributes:
        thresholds (QualityThresholds): Configurable thresholds for quality assessment
    
    Example:
        >>> thresholds = QualityThresholds(high_missing_data_pct=15.0, high_duplicate_pct=0.5)
        >>> assessor = QualityAssessor(thresholds)
        >>> assessment = assessor.assess_data_quality(dataset)
        >>> if assessment.investigation_priority == "high":
        ...     print(f"Dataset {dataset.dataset_name} requires immediate attention")
    """
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        """
        Initialize the QualityAssessor with configurable thresholds.
        
        Args:
            thresholds: Quality thresholds configuration. Uses defaults if None.
        """
        self.thresholds = thresholds or QualityThresholds()
        logger.info(f"QualityAssessor initialized with thresholds: "
                   f"missing_data={self.thresholds.high_missing_data_pct}%, "
                   f"duplicates={self.thresholds.high_duplicate_pct}%")
    
    def assess_data_quality(self, dataset: ProfileData) -> QualityAssessment:
        """
        Assess the overall data quality of a single dataset.
        
        Performs comprehensive quality evaluation across multiple dimensions:
        - Missing data analysis (cell-level and column-level)
        - Duplicate record detection
        - Memory usage assessment
        - Data type distribution analysis
        - Overall quality scoring and prioritization
        
        Args:
            dataset (ProfileData): The dataset profile to assess. Must contain:
                - table_stats: Table-level statistics including row count, missing data percentages
                - variables: List of variable statistics for column-level analysis
                - data_type_distribution: Distribution of data types across columns
                - dataset_name: Identifier for the dataset
        
        Returns:
            QualityAssessment: Comprehensive assessment containing:
                - dataset_name: Name of the assessed dataset
                - overall_score: Quality score from 0-100 (higher is better)
                - issues: List of identified quality issues with severity and recommendations
                - strengths: List of positive quality aspects
                - investigation_priority: Priority level (low/medium/high) for further investigation
        
        Raises:
            ValueError: If dataset is None or missing required attributes
            TypeError: If dataset is not a ProfileData instance
        
        Example:
            >>> assessment = assessor.assess_data_quality(dataset)
            >>> print(f"Quality Score: {assessment.overall_score:.1f}/100")
            >>> for issue in assessment.issues:
            ...     print(f"- {issue.severity.upper()}: {issue.description}")
        """
        logger.debug(f"Assessing data quality for dataset: {dataset.dataset_name}")
        
        issues = []
        strengths = []
        
        # Check for missing data issues (Requirement 4.1)
        missing_data_issue = self._check_missing_data(dataset)
        if missing_data_issue:
            issues.append(missing_data_issue)
        else:
            strengths.append("Low missing data percentage")
        
        # Check for duplicate records (Requirement 4.2)
        duplicate_issue = self._check_duplicates(dataset)
        if duplicate_issue:
            issues.append(duplicate_issue)
        else:
            strengths.append("Low duplicate record percentage")
        
        # Check for memory concerns (Requirement 4.4)
        memory_issue = self.assess_memory_concerns(dataset)
        if memory_issue:
            issues.append(memory_issue)
        
        # Check for unusual data type distributions (Requirement 4.3)
        type_issue = self.detect_unusual_data_type_distributions(dataset)
        if type_issue:
            issues.append(type_issue)
        
        # Calculate overall quality score
        overall_score = self._calculate_quality_score(dataset, issues)
        
        # Determine investigation priority
        investigation_priority = self._determine_investigation_priority(issues, overall_score)
        
        return QualityAssessment(
            dataset_name=dataset.dataset_name,
            overall_score=overall_score,
            issues=issues,
            strengths=strengths,
            investigation_priority=investigation_priority
        )
    
    def flag_quality_issues(self, datasets: List[ProfileData]) -> List[QualityIssue]:
        """
        Identify and flag quality issues across all datasets.
        
        Args:
            datasets: List of dataset profiles to analyze
            
        Returns:
            List of quality issues found across all datasets
        """
        logger.info(f"Flagging quality issues across {len(datasets)} datasets")
        
        all_issues = []
        
        for dataset in datasets:
            assessment = self.assess_data_quality(dataset)
            all_issues.extend(assessment.issues)
        
        logger.info(f"Found {len(all_issues)} quality issues across all datasets")
        return all_issues
    
    def prioritize_datasets(self, issues: List[QualityIssue]) -> List[DatasetPriority]:
        """
        Prioritize datasets based on severity of quality issues.
        
        Args:
            issues: List of quality issues to prioritize
            
        Returns:
            List of dataset priorities sorted by priority score (highest first)
        """
        logger.info(f"Prioritizing datasets based on {len(issues)} quality issues")
        
        # Group issues by dataset
        dataset_issues: Dict[str, List[QualityIssue]] = {}
        for issue in issues:
            if issue.dataset_name not in dataset_issues:
                dataset_issues[issue.dataset_name] = []
            dataset_issues[issue.dataset_name].append(issue)
        
        priorities = []
        
        for dataset_name, dataset_issues_list in dataset_issues.items():
            priority_score = self._calculate_priority_score(dataset_issues_list)
            primary_issues = [issue.issue_type for issue in dataset_issues_list[:3]]  # Top 3 issues
            recommended_actions = self._generate_priority_actions(dataset_issues_list)
            
            priorities.append(DatasetPriority(
                dataset_name=dataset_name,
                priority_score=priority_score,
                primary_issues=primary_issues,
                recommended_actions=recommended_actions
            ))
        
        # Sort by priority score (highest first)
        priorities.sort(key=lambda x: x.priority_score, reverse=True)
        
        logger.info(f"Generated priorities for {len(priorities)} datasets")
        return priorities
    
    def generate_recommendations(self, issue: QualityIssue) -> List[str]:
        """
        Generate actionable recommendations for a specific quality issue.
        
        Args:
            issue: The quality issue to generate recommendations for
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        if issue.issue_type == "missing_data":
            recommendations.extend(self._get_missing_data_recommendations(issue))
        elif issue.issue_type == "duplicates":
            recommendations.extend(self._get_duplicate_recommendations(issue))
        elif issue.issue_type == "memory_concern":
            recommendations.extend(self._get_memory_recommendations(issue))
        elif issue.issue_type == "unusual_types":
            recommendations.extend(self._get_type_distribution_recommendations(issue))
        
        return recommendations
    
    def assess_memory_concerns(self, dataset: ProfileData) -> Optional[QualityIssue]:
        """
        Assess if a dataset has memory usage concerns.
        
        Args:
            dataset: The dataset to assess
            
        Returns:
            QualityIssue if memory concerns are found, None otherwise
        """
        memory_mb = dataset.table_stats.memory_size / (1024 * 1024)  # Convert to MB
        
        if memory_mb > self.thresholds.memory_concern_mb:
            severity = "high" if memory_mb > self.thresholds.memory_concern_mb * 5 else "medium"
            
            return QualityIssue(
                dataset_name=dataset.dataset_name,
                issue_type="memory_concern",
                severity=severity,
                description=f"Dataset uses {memory_mb:.1f}MB of memory, which exceeds the {self.thresholds.memory_concern_mb}MB threshold",
                recommendation="Consider data sampling, compression, or chunked processing for large datasets",
                affected_columns=[],
                metrics={"memory_mb": memory_mb, "threshold_mb": self.thresholds.memory_concern_mb}
            )
        
        return None
    
    def detect_unusual_data_type_distributions(self, dataset: ProfileData) -> Optional[QualityIssue]:
        """
        Detect unusual data type distributions in a dataset.
        
        Args:
            dataset: The dataset to analyze
            
        Returns:
            QualityIssue if unusual distributions are found, None otherwise
        """
        if not dataset.data_type_distribution:
            return None
        
        total_columns = sum(dataset.data_type_distribution.values())
        if total_columns == 0:
            return None
        
        # Check if any single data type dominates (>80% of columns)
        for data_type, count in dataset.data_type_distribution.items():
            proportion = count / total_columns
            
            if proportion > self.thresholds.unusual_type_distribution_threshold:
                return QualityIssue(
                    dataset_name=dataset.dataset_name,
                    issue_type="unusual_types",
                    severity="medium",
                    description=f"Dataset has unusual data type distribution: {data_type} represents {proportion:.1%} of all columns",
                    recommendation="Review data types for potential standardization or schema improvements",
                    affected_columns=[],
                    metrics={"dominant_type": data_type, "proportion": proportion}
                )
        
        return None
    
    def suggest_investigation_candidates(self, datasets: List[ProfileData]) -> List[str]:
        """
        Suggest datasets that would benefit from further investigation.
        
        Args:
            datasets: List of datasets to analyze
            
        Returns:
            List of dataset names recommended for investigation
        """
        candidates = []
        
        for dataset in datasets:
            assessment = self.assess_data_quality(dataset)
            
            # Suggest datasets with high priority issues or low quality scores
            if (assessment.investigation_priority == "high" or 
                assessment.overall_score < 60 or
                len(assessment.issues) >= 3):
                candidates.append(dataset.dataset_name)
        
        logger.info(f"Identified {len(candidates)} datasets as investigation candidates")
        return candidates
    
    def get_quality_thresholds(self) -> QualityThresholds:
        """
        Get the current quality thresholds configuration.
        
        Returns:
            Current QualityThresholds configuration
        """
        return self.thresholds
    
    # Private helper methods
    
    def _check_missing_data(self, dataset: ProfileData) -> Optional[QualityIssue]:
        """Check for high missing data percentage (Requirement 4.1)."""
        missing_pct = dataset.table_stats.missing_cells_pct
        
        if missing_pct > self.thresholds.high_missing_data_pct:
            # Find columns with high missing data
            affected_columns = []
            for var in dataset.variables:
                if var.missing_pct > self.thresholds.high_missing_data_pct:
                    affected_columns.append(var.name)
            
            severity = self._determine_severity(missing_pct, self.thresholds.high_missing_data_pct)
            
            return QualityIssue(
                dataset_name=dataset.dataset_name,
                issue_type="missing_data",
                severity=severity,
                description=f"Dataset has {missing_pct:.1f}% missing data, exceeding the {self.thresholds.high_missing_data_pct}% threshold",
                recommendation="Investigate data collection processes and consider imputation strategies",
                affected_columns=affected_columns,
                metrics={"missing_pct": missing_pct, "threshold": self.thresholds.high_missing_data_pct}
            )
        
        return None
    
    def _check_duplicates(self, dataset: ProfileData) -> Optional[QualityIssue]:
        """Check for high duplicate percentage (Requirement 4.2)."""
        duplicate_pct = dataset.table_stats.duplicate_rows_pct
        
        if duplicate_pct > self.thresholds.high_duplicate_pct:
            severity = self._determine_severity(duplicate_pct, self.thresholds.high_duplicate_pct)
            
            return QualityIssue(
                dataset_name=dataset.dataset_name,
                issue_type="duplicates",
                severity=severity,
                description=f"Dataset has {duplicate_pct:.1f}% duplicate records, exceeding the {self.thresholds.high_duplicate_pct}% threshold",
                recommendation="Review data ingestion processes and implement deduplication procedures",
                affected_columns=[],
                metrics={"duplicate_pct": duplicate_pct, "threshold": self.thresholds.high_duplicate_pct}
            )
        
        return None
    
    def _calculate_quality_score(self, dataset: ProfileData, issues: List[QualityIssue]) -> float:
        """Calculate overall quality score (0-100)."""
        base_score = 100.0
        
        # Deduct points for each issue based on severity
        severity_penalties = {"low": 5, "medium": 15, "high": 25, "critical": 40}
        
        for issue in issues:
            penalty = severity_penalties.get(issue.severity, 10)
            base_score -= penalty
        
        # Additional factors
        missing_pct = dataset.table_stats.missing_cells_pct
        duplicate_pct = dataset.table_stats.duplicate_rows_pct
        
        # Gradual penalty for missing data even below threshold
        if missing_pct > 0:
            base_score -= min(missing_pct * 0.5, 20)  # Max 20 points for missing data
        
        # Gradual penalty for duplicates even below threshold
        if duplicate_pct > 0:
            base_score -= min(duplicate_pct * 2, 15)  # Max 15 points for duplicates
        
        return max(0.0, min(100.0, base_score))
    
    def _determine_investigation_priority(self, issues: List[QualityIssue], quality_score: float) -> str:
        """Determine investigation priority based on issues and quality score."""
        critical_issues = [i for i in issues if i.severity == "critical"]
        high_issues = [i for i in issues if i.severity == "high"]
        
        if critical_issues or quality_score < 40:
            return "high"
        elif high_issues or quality_score < 70:
            return "medium"
        else:
            return "low"
    
    def _determine_severity(self, value: float, threshold: float) -> str:
        """Determine severity based on how much a value exceeds the threshold."""
        if value > threshold * 10:
            return "critical"
        elif value > threshold * 3:
            return "high"
        elif value > threshold * 1.05:
            return "medium"
        else:
            return "low"
    
    def _calculate_priority_score(self, issues: List[QualityIssue]) -> int:
        """Calculate priority score for a dataset based on its issues."""
        severity_scores = {"low": 1, "medium": 3, "high": 7, "critical": 15}
        
        total_score = 0
        for issue in issues:
            total_score += severity_scores.get(issue.severity, 1)
        
        # Bonus for multiple issues (indicates systemic problems)
        if len(issues) > 2:
            total_score += len(issues) * 2
        
        return total_score
    
    def _generate_priority_actions(self, issues: List[QualityIssue]) -> List[str]:
        """Generate recommended actions for a dataset based on its issues."""
        actions = set()  # Use set to avoid duplicates
        
        for issue in issues:
            if issue.issue_type == "missing_data":
                actions.add("Investigate data collection and implement missing data handling")
            elif issue.issue_type == "duplicates":
                actions.add("Implement deduplication procedures")
            elif issue.issue_type == "memory_concern":
                actions.add("Consider data optimization or sampling strategies")
            elif issue.issue_type == "unusual_types":
                actions.add("Review and standardize data types")
        
        # Add general actions based on severity
        critical_issues = [i for i in issues if i.severity == "critical"]
        if critical_issues:
            actions.add("Immediate attention required - critical data quality issues")
        
        return list(actions)
    
    def _get_missing_data_recommendations(self, issue: QualityIssue) -> List[str]:
        """Get specific recommendations for missing data issues."""
        recommendations = [
            "Analyze missing data patterns to understand if data is missing at random",
            "Review data collection processes to identify and fix gaps",
            "Consider appropriate imputation strategies for missing values",
            "Document missing data handling procedures for future reference"
        ]
        
        if issue.severity in ["high", "critical"]:
            recommendations.insert(0, "Urgent: High levels of missing data may indicate systematic collection issues")
        
        return recommendations
    
    def _get_duplicate_recommendations(self, issue: QualityIssue) -> List[str]:
        """Get specific recommendations for duplicate data issues."""
        recommendations = [
            "Implement automated deduplication procedures in data ingestion pipeline",
            "Identify root causes of duplicate record creation",
            "Establish unique identifiers and constraints to prevent duplicates",
            "Regular monitoring and cleanup of duplicate records"
        ]
        
        if issue.severity in ["high", "critical"]:
            recommendations.insert(0, "Urgent: High duplicate rates suggest data pipeline issues")
        
        return recommendations
    
    def _get_memory_recommendations(self, issue: QualityIssue) -> List[str]:
        """Get specific recommendations for memory concern issues."""
        return [
            "Consider data sampling for analysis and development environments",
            "Implement data compression techniques to reduce memory footprint",
            "Use chunked processing for large datasets",
            "Evaluate if all columns are necessary for analysis",
            "Consider data archiving strategies for historical data"
        ]
    
    def _get_type_distribution_recommendations(self, issue: QualityIssue) -> List[str]:
        """Get specific recommendations for unusual type distribution issues."""
        return [
            "Review data schema design and type assignments",
            "Consider data type standardization across similar datasets",
            "Evaluate if current data types are optimal for intended use",
            "Document rationale for unusual type distributions if intentional"
        ]