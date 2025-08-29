"""
Unit tests for the QualityAssessor component.
Tests Requirements: 4.1, 4.2, 4.3, 4.4
"""

from datetime import datetime
from pathlib import Path

import pytest

from consolidator.models import (
    AnalysisInfo,
    ProfileData,
    QualityAssessment,
    QualityIssue,
    QualityThresholds,
    TableStats,
    VariableStats,
)
from consolidator.quality import QualityAssessor


class TestQualityAssessor:
    """Test suite for QualityAssessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.thresholds = QualityThresholds(
            high_missing_data_pct=20.0,
            high_duplicate_pct=1.0,
            large_dataset_rows=1_000_000,
            memory_concern_mb=100,
            unusual_type_distribution_threshold=0.8,
            sample_dataset_row_threshold=10_000,
        )
        self.assessor = QualityAssessor(self.thresholds)

    def create_test_profile(
        self,
        missing_pct=0.0,
        duplicate_pct=0.0,
        memory_mb=10,
        rows=1000,
        unusual_types=False,
    ):
        """Helper to create test ProfileData objects."""
        data_types = (
            {"Numeric": 5, "Categorical": 3} if not unusual_types else {"Unknown": 8}
        )

        return ProfileData(
            file_path=Path("test/dataset/data_profile.json"),
            dataset_name="test_dataset",
            analysis_info=AnalysisInfo(
                title="Test Dataset",
                date_start=datetime(2025, 8, 29, 10, 0, 0),
                date_end=datetime(2025, 8, 29, 10, 0, 5),
            ),
            table_stats=TableStats(
                n_rows=rows,
                n_columns=8,
                memory_size=memory_mb * 1024 * 1024,  # Convert MB to bytes
                missing_cells_pct=missing_pct,
                duplicate_rows_pct=duplicate_pct,
                data_types=data_types,
                is_sample_dataset=rows < 10000,
            ),
            variables=[
                VariableStats(
                    name="test_var",
                    data_type="Numeric",
                    missing_pct=missing_pct,
                    unique_values=rows,
                    memory_size=memory_mb * 1024 * 1024 // 8,
                )
            ],
            data_type_distribution=data_types,
            quality_flags=[],
        )

    def test_assess_high_quality_dataset(self):
        """Test assessment of high quality dataset with no issues."""
        profile = self.create_test_profile(
            missing_pct=0.0, duplicate_pct=0.0, memory_mb=10, rows=5000
        )

        assessment = self.assessor.assess_data_quality(profile)

        assert isinstance(assessment, QualityAssessment)
        assert assessment.dataset_name == "test_dataset"
        assert assessment.overall_score > 80  # High quality score
        assert len(assessment.issues) == 0
        assert len(assessment.strengths) > 0
        assert assessment.investigation_priority == "low"

    def test_assess_high_duplicates(self):
        """Test flagging datasets with high duplicate records (Requirement 4.2)."""
        profile = self.create_test_profile(duplicate_pct=5.0)  # Above 1% threshold

        assessment = self.assessor.assess_data_quality(profile)

        assert len(assessment.issues) > 0
        duplicate_issue = next(
            (issue for issue in assessment.issues if issue.issue_type == "duplicates"),
            None,
        )
        assert duplicate_issue is not None
        assert duplicate_issue.severity in ["medium", "high"]

    def test_assess_memory_concerns(self):
        """Test flagging datasets with memory usage concerns (Requirement 4.4)."""
        profile = self.create_test_profile(memory_mb=150)  # Above 100MB threshold

        assessment = self.assessor.assess_data_quality(profile)

        memory_issue = next(
            (
                issue
                for issue in assessment.issues
                if issue.issue_type == "memory_concern"
            ),
            None,
        )
        assert memory_issue is not None
        assert "memory" in memory_issue.description.lower()

    def test_assess_unusual_data_types(self):
        """Test flagging datasets with unusual data type distributions (Requirement 4.3)."""
        profile = self.create_test_profile(unusual_types=True)

        assessment = self.assessor.assess_data_quality(profile)

        type_issue = next(
            (
                issue
                for issue in assessment.issues
                if issue.issue_type == "unusual_types"
            ),
            None,
        )
        assert type_issue is not None
        assert "data type" in type_issue.description.lower()

    def test_flag_quality_issues_multiple_datasets(self, sample_profile_data):
        """Test flagging quality issues across multiple datasets."""
        issues = self.assessor.flag_quality_issues(sample_profile_data)

        assert len(issues) > 0

        # Should find issues in the poor quality dataset
        poor_quality_issues = [
            issue for issue in issues if issue.dataset_name == "dataset2"
        ]
        assert len(poor_quality_issues) > 0

        # Check for expected issue types
        issue_types = [issue.issue_type for issue in poor_quality_issues]
        assert "missing_data" in issue_types
        assert "duplicates" in issue_types

    def test_prioritize_datasets(self, sample_profile_data):
        """Test dataset prioritization based on quality issues (Requirement 4.4)."""
        issues = self.assessor.flag_quality_issues(sample_profile_data)
        priorities = self.assessor.prioritize_datasets(issues)

        assert len(priorities) > 0

        # Priorities should be sorted by severity
        for i in range(len(priorities) - 1):
            current_score = priorities[i].priority_score
            next_score = priorities[i + 1].priority_score
            assert current_score >= next_score

        # High priority dataset should have recommendations
        high_priority = priorities[0]
        assert len(high_priority.recommended_actions) > 0

    def test_generate_recommendations_missing_data(self):
        """Test recommendation generation for missing data issues."""
        issue = QualityIssue(
            dataset_name="test_dataset",
            issue_type="missing_data",
            severity="high",
            description="High percentage of missing data",
            recommendation="",
            affected_columns=["column1", "column2"],
            metrics={"missing_pct": 30.0},
        )

        recommendations = self.assessor.generate_recommendations(issue)

        assert len(recommendations) > 0
        assert any("missing" in rec.lower() for rec in recommendations)
        assert any(
            "imputation" in rec.lower() or "removal" in rec.lower()
            for rec in recommendations
        )

    def test_generate_recommendations_duplicates(self):
        """Test recommendation generation for duplicate records."""
        issue = QualityIssue(
            dataset_name="test_dataset",
            issue_type="duplicates",
            severity="medium",
            description="High percentage of duplicate records",
            recommendation="",
            affected_columns=[],
            metrics={"duplicate_pct": 3.0},
        )

        recommendations = self.assessor.generate_recommendations(issue)

        assert len(recommendations) > 0
        assert any("duplicate" in rec.lower() for rec in recommendations)
        assert any(
            "deduplication" in rec.lower() or "remove" in rec.lower()
            for rec in recommendations
        )

    def test_generate_recommendations_memory_concern(self):
        """Test recommendation generation for memory concerns."""
        issue = QualityIssue(
            dataset_name="test_dataset",
            issue_type="memory_concern",
            severity="medium",
            description="Large memory usage",
            recommendation="",
            affected_columns=[],
            metrics={"memory_mb": 500},
        )

        recommendations = self.assessor.generate_recommendations(issue)

        assert len(recommendations) > 0
        assert any(
            "memory" in rec.lower() or "optimization" in rec.lower()
            for rec in recommendations
        )

    def test_assess_memory_concerns_large_dataset(self):
        """Test memory concern assessment for large datasets."""
        profile = self.create_test_profile(memory_mb=200, rows=2_000_000)

        memory_issue = self.assessor.assess_memory_concerns(profile)

        assert memory_issue is not None
        assert memory_issue.issue_type == "memory_concern"
        assert memory_issue.metrics["memory_mb"] == 200

    def test_assess_memory_concerns_small_dataset(self):
        """Test memory concern assessment for small datasets."""
        profile = self.create_test_profile(memory_mb=50, rows=1000)

        memory_issue = self.assessor.assess_memory_concerns(profile)

        assert memory_issue is None  # No memory concerns for small datasets

    def test_detect_unusual_data_type_distributions(self):
        """Test detection of unusual data type distributions."""
        profile = self.create_test_profile(unusual_types=True)

        type_issue = self.assessor.detect_unusual_data_type_distributions(profile)

        assert type_issue is not None
        assert type_issue.issue_type == "unusual_types"
        assert "Unknown" in type_issue.description

    def test_detect_normal_data_type_distributions(self):
        """Test that normal data type distributions don't trigger issues."""
        profile = self.create_test_profile(unusual_types=False)

        type_issue = self.assessor.detect_unusual_data_type_distributions(profile)

        assert type_issue is None  # Normal distribution should not trigger issue

    def test_suggest_investigation_candidates(self, sample_profile_data):
        """Test suggestion of datasets for further investigation."""
        candidates = self.assessor.suggest_investigation_candidates(sample_profile_data)

        assert len(candidates) > 0
        # Should suggest the poor quality dataset
        assert "dataset2" in candidates

    def test_quality_thresholds_customization(self):
        """Test that quality thresholds can be customized."""
        custom_thresholds = QualityThresholds(
            high_missing_data_pct=10.0,  # Lower threshold
            high_duplicate_pct=0.5,  # Lower threshold
            memory_concern_mb=50,  # Lower threshold
        )

        custom_assessor = QualityAssessor(custom_thresholds)

        # Dataset that would be OK with default thresholds
        profile = self.create_test_profile(
            missing_pct=15.0,  # Between 10% and 20%
            duplicate_pct=0.8,  # Between 0.5% and 1%
            memory_mb=75,  # Between 50MB and 100MB
        )

        assessment = custom_assessor.assess_data_quality(profile)

        # Should flag issues with custom thresholds
        assert len(assessment.issues) > 0

    def test_overall_quality_score_calculation(self):
        """Test overall quality score calculation."""
        # Perfect dataset
        perfect_profile = self.create_test_profile(
            missing_pct=0.0, duplicate_pct=0.0, memory_mb=10
        )
        perfect_assessment = self.assessor.assess_data_quality(perfect_profile)

        # Poor dataset
        poor_profile = self.create_test_profile(
            missing_pct=50.0, duplicate_pct=10.0, memory_mb=200
        )
        poor_assessment = self.assessor.assess_data_quality(poor_profile)

        # Perfect dataset should have higher score
        assert perfect_assessment.overall_score > poor_assessment.overall_score
        assert perfect_assessment.overall_score >= 90
        assert poor_assessment.overall_score <= 50

    def test_investigation_priority_assignment(self):
        """Test investigation priority assignment based on issues."""
        # Low priority (no issues)
        good_profile = self.create_test_profile()
        good_assessment = self.assessor.assess_data_quality(good_profile)
        assert good_assessment.investigation_priority == "low"

        # Medium priority (some issues)
        medium_profile = self.create_test_profile(missing_pct=15.0)
        medium_assessment = self.assessor.assess_data_quality(medium_profile)
        assert medium_assessment.investigation_priority in ["low", "medium"]

        # High priority (multiple severe issues)
        bad_profile = self.create_test_profile(
            missing_pct=40.0, duplicate_pct=8.0, memory_mb=300
        )
        bad_assessment = self.assessor.assess_data_quality(bad_profile)
        assert bad_assessment.investigation_priority in ["medium", "high"]
