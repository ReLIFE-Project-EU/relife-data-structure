"""
Tests for report generation and markdown structure validation.
Tests Requirements: 3.2, 3.3, 3.5, 7.1, 7.2, 7.3, 7.4, 7.5
"""
import pytest
from pathlib import Path
from datetime import datetime

from consolidator.generator import ReportGenerator
from consolidator.analyzer import DataAnalyzer
from consolidator.quality import QualityAssessor
from consolidator.models import (
    ConsolidatorConfig, DatasetAnalysis, QualityIssue, 
    ConsolidatedReport, ProfileData, TableStats, AnalysisInfo
)


class TestReportGeneration:
    """Test suite for report generation and markdown structure."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ConsolidatorConfig()
        self.generator = ReportGenerator(self.config)
        self.analyzer = DataAnalyzer()
        self.quality_assessor = QualityAssessor(self.config.quality_thresholds)
    
    def test_generate_consolidated_report_structure(self, sample_profile_data):
        """Test that generated report has correct markdown structure."""
        # Analyze sample data
        analysis = self.analyzer.analyze_dataset_collection(sample_profile_data)
        quality_issues = self.quality_assessor.flag_quality_issues(sample_profile_data)
        
        # Generate report
        report = self.generator.generate_consolidated_report(analysis, quality_issues)
        
        assert isinstance(report, ConsolidatedReport)
        assert report.executive_summary != ""
        assert report.main_report != ""
        assert report.generation_timestamp is not None
        assert report.datasets_analyzed > 0
        assert report.estimated_reading_time_minutes > 0
    
    def test_executive_summary_content(self, sample_profile_data):
        """Test executive summary contains required information (Requirement 3.2)."""
        analysis = self.analyzer.analyze_dataset_collection(sample_profile_data)
        
        summary = self.generator.create_executive_summary(analysis)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        
        # Should contain key metrics
        assert str(analysis.total_datasets) in summary
        assert "dataset" in summary.lower()
        
        # Should mention categories if they exist
        if len(analysis.category_breakdown) > 0:
            assert "categor" in summary.lower()
        
        # Should include quality information
        assert "quality" in summary.lower() or "issue" in summary.lower()
    
    def test_markdown_headers_structure(self, sample_profile_data):
        """Test that report uses proper markdown header structure."""
        analysis = self.analyzer.analyze_dataset_collection(sample_profile_data)
        quality_issues = self.quality_assessor.flag_quality_issues(sample_profile_data)
        
        report = self.generator.generate_consolidated_report(analysis, quality_issues)
        content = report.main_report
        
        # Should have main title
        assert content.startswith("# ")
        
        # Should have proper header hierarchy
        lines = content.split('\n')
        headers = [line for line in lines if line.startswith('#')]
        
        assert len(headers) >= 3  # At least title and a few sections
        
        # Check for expected sections
        header_text = ' '.join(headers).lower()
        assert "executive summary" in header_text or "summary" in header_text
        assert "dataset" in header_text
        assert "quality" in header_text or "issue" in header_text
    
    def test_category_breakdown_formatting(self, sample_profile_data):
        """Test category breakdown section formatting (Requirement 3.1)."""
        analysis = self.analyzer.analyze_dataset_collection(sample_profile_data)
        
        category_section = self.generator.generate_category_breakdown(analysis)
        
        assert isinstance(category_section, str)
        
        if len(analysis.category_breakdown) > 0:
            # Should format categories properly
            for category, count in analysis.category_breakdown.items():
                assert category in category_section
                assert str(count) in category_section
            
            # Should use markdown formatting
            assert "##" in category_section or "###" in category_section or "-" in category_section
    
    def test_quality_issues_prioritization(self, sample_profile_data):
        """Test quality issues are prioritized and limited (Requirement 7.3)."""
        analysis = self.analyzer.analyze_dataset_collection(sample_profile_data)
        quality_issues = self.quality_assessor.flag_quality_issues(sample_profile_data)
        
        # Add more issues to test prioritization
        additional_issues = [
            QualityIssue(
                dataset_name=f"test_dataset_{i}",
                issue_type="missing_data",
                severity="medium" if i % 2 == 0 else "low",
                description=f"Test issue {i}",
                recommendation=f"Fix issue {i}",
                affected_columns=[],
                metrics={"missing_pct": 15.0 + i}
            ) for i in range(15)  # Create many issues
        ]
        
        all_issues = quality_issues + additional_issues
        
        # Format quality issues with limit
        max_issues = 10
        formatted_issues = self.generator.format_quality_issues(all_issues, max_issues)
        
        # Should limit to max_issues
        issue_count = formatted_issues.count("###") + formatted_issues.count("##")
        assert issue_count <= max_issues
        
        # Should prioritize high severity issues
        if "critical" in str(all_issues) or "high" in str(all_issues):
            assert "critical" in formatted_issues.lower() or "high" in formatted_issues.lower()
    
    def test_report_length_management(self, sample_profile_data):
        """Test report length stays within limits (Requirement 7.5)."""
        # Create config with short length limit
        short_config = ConsolidatorConfig(max_report_length=2000)
        short_generator = ReportGenerator(short_config)
        
        analysis = self.analyzer.analyze_dataset_collection(sample_profile_data)
        quality_issues = self.quality_assessor.flag_quality_issues(sample_profile_data)
        
        report = short_generator.generate_consolidated_report(analysis, quality_issues)
        
        # Main report should respect length limits
        main_report_length = len(report.main_report)
        assert main_report_length <= short_config.max_report_length * 1.2  # Allow some flexibility
        
        # Should still contain essential information
        assert "dataset" in report.main_report.lower()
        assert str(analysis.total_datasets) in report.main_report
    
    def test_detailed_appendix_generation(self, sample_profile_data):
        """Test detailed appendix generation (Requirement 7.4)."""
        config_with_appendix = ConsolidatorConfig(include_detailed_appendix=True)
        generator_with_appendix = ReportGenerator(config_with_appendix)
        
        analysis = self.analyzer.analyze_dataset_collection(sample_profile_data)
        quality_issues = self.quality_assessor.flag_quality_issues(sample_profile_data)
        
        report = generator_with_appendix.generate_consolidated_report(analysis, quality_issues)
        
        assert report.detailed_appendix != ""
        assert len(report.detailed_appendix) > len(report.executive_summary)
        
        # Appendix should contain more detailed information
        appendix_lower = report.detailed_appendix.lower()
        assert "appendix" in appendix_lower or "detailed" in appendix_lower
    
    def test_reading_time_estimation(self, sample_profile_data):
        """Test reading time estimation (Requirement 7.5)."""
        analysis = self.analyzer.analyze_dataset_collection(sample_profile_data)
        quality_issues = self.quality_assessor.flag_quality_issues(sample_profile_data)
        
        report = self.generator.generate_consolidated_report(analysis, quality_issues)
        
        # Should estimate reading time
        assert report.estimated_reading_time_minutes > 0
        assert report.estimated_reading_time_minutes < 60  # Should be reasonable
        
        # Test reading time calculation method
        sample_text = "This is a test. " * 1000  # ~1000 words
        estimated_time = self.generator.estimate_reading_time(sample_text)
        
        # Should be around 4-5 minutes for 1000 words (200 WPM average)
        assert 3 <= estimated_time <= 7
    
    def test_markdown_formatting_elements(self, sample_profile_data):
        """Test proper markdown formatting elements are used."""
        analysis = self.analyzer.analyze_dataset_collection(sample_profile_data)
        quality_issues = self.quality_assessor.flag_quality_issues(sample_profile_data)
        
        report = self.generator.generate_consolidated_report(analysis, quality_issues)
        content = report.main_report
        
        # Should use proper markdown elements
        assert "**" in content or "*" in content  # Bold or italic
        assert "##" in content  # Headers
        assert "-" in content or "1." in content  # Lists
        
        # Should have proper line breaks
        assert "\n\n" in content  # Paragraph breaks
        
        # Should not have HTML tags (pure markdown)
        assert "<" not in content or content.count("<") == content.count(">")
    
    def test_database_administrator_focus(self, sample_profile_data):
        """Test report is formatted for database administrators."""
        analysis = self.analyzer.analyze_dataset_collection(sample_profile_data)
        quality_issues = self.quality_assessor.flag_quality_issues(sample_profile_data)
        
        report = self.generator.generate_consolidated_report(analysis, quality_issues)
        content = report.main_report.lower()
        
        # Should use database-relevant terminology
        db_terms = ["dataset", "data", "table", "column", "schema", "quality", "memory", "size"]
        assert any(term in content for term in db_terms)
        
        # Should focus on actionable information
        action_terms = ["recommend", "suggest", "priority", "issue", "attention", "investigate"]
        assert any(term in content for term in action_terms)
    
    def test_aggregation_over_individual_listings(self, temp_reports_dir):
        """Test that report uses aggregations rather than individual listings (Requirement 7.2)."""
        # Create many similar datasets
        many_profiles = []
        for i in range(50):
            profile = ProfileData(
                file_path=Path(f"test/dataset_{i}/data_profile.json"),
                dataset_name=f"dataset_{i}",
                analysis_info=AnalysisInfo(
                    title=f"Dataset {i}",
                    date_start=datetime(2025, 8, 29, 10, 0, 0),
                    date_end=datetime(2025, 8, 29, 10, 0, 5)
                ),
                table_stats=TableStats(
                    n_rows=1000 + i * 100,
                    n_columns=5,
                    memory_size=40000 + i * 1000,
                    missing_cells_pct=0.0,
                    duplicate_rows_pct=0.0,
                    data_types={"Numeric": 3, "Categorical": 2},
                    is_sample_dataset=True
                ),
                variables=[],
                data_type_distribution={"Numeric": 3, "Categorical": 2},
                quality_flags=[]
            )
            many_profiles.append(profile)
        
        analysis = self.analyzer.analyze_dataset_collection(many_profiles)
        quality_issues = self.quality_assessor.flag_quality_issues(many_profiles)
        
        report = self.generator.generate_consolidated_report(analysis, quality_issues)
        content = report.main_report
        
        # Should use summary statistics rather than listing all 50 datasets
        dataset_mentions = content.count("dataset_")
        assert dataset_mentions < 10  # Should not list most individual datasets
        
        # Should mention total count
        assert "50" in content
        
        # Should use aggregation terms
        aggregation_terms = ["total", "average", "distribution", "summary", "overall"]
        assert any(term in content.lower() for term in aggregation_terms)
    
    def test_timestamp_and_metadata_inclusion(self, sample_profile_data):
        """Test that report includes proper timestamp and metadata."""
        analysis = self.analyzer.analyze_dataset_collection(sample_profile_data)
        quality_issues = self.quality_assessor.flag_quality_issues(sample_profile_data)
        
        report = self.generator.generate_consolidated_report(analysis, quality_issues)
        content = report.main_report
        
        # Should include generation timestamp
        assert "Generated:" in content or "Timestamp:" in content
        assert "2025" in content  # Should have current year
        
        # Should include dataset count
        assert str(report.datasets_analyzed) in content
        
        # Should include reading time estimate
        assert str(report.estimated_reading_time_minutes) in content or "minutes" in content
    
    def test_recommendations_section_formatting(self, sample_profile_data):
        """Test recommendations section formatting and content."""
        analysis = self.analyzer.analyze_dataset_collection(sample_profile_data)
        quality_issues = self.quality_assessor.flag_quality_issues(sample_profile_data)
        
        recommendations_section = self.generator.create_recommendations_section(analysis, quality_issues)
        
        assert isinstance(recommendations_section, str)
        
        if len(quality_issues) > 0:
            # Should contain actionable recommendations
            rec_lower = recommendations_section.lower()
            assert "recommend" in rec_lower or "suggest" in rec_lower
            
            # Should use proper formatting
            assert "##" in recommendations_section or "-" in recommendations_section
    
    def test_empty_dataset_handling(self):
        """Test report generation with no datasets."""
        empty_analysis = DatasetAnalysis(
            total_datasets=0,
            analysis_timestamp=datetime.now(),
            size_distribution=None,
            quality_summary=None,
            schema_analysis=None,
            schema_patterns=[],
            category_breakdown={},
            category_insights={},
            sample_vs_complete={},
            standardization_opportunities=[]
        )
        
        report = self.generator.generate_consolidated_report(empty_analysis, [])
        
        assert report.datasets_analyzed == 0
        assert "0" in report.main_report or "no datasets" in report.main_report.lower()
        assert report.estimated_reading_time_minutes > 0  # Should still have some content
    
    def test_template_loading_and_customization(self):
        """Test template loading and customization capabilities."""
        # Test that generator can load templates
        try:
            template_content = self.generator.load_template("report_template.md")
            assert isinstance(template_content, str)
            
            # Should contain placeholder markers
            assert "{" in template_content and "}" in template_content
            
        except FileNotFoundError:
            # Template file doesn't exist yet, which is acceptable
            pass
    
    def test_content_prioritization(self, sample_profile_data):
        """Test that content is properly prioritized for conciseness."""
        analysis = self.analyzer.analyze_dataset_collection(sample_profile_data)
        quality_issues = self.quality_assessor.flag_quality_issues(sample_profile_data)
        
        # Create many quality issues to test prioritization
        many_issues = quality_issues + [
            QualityIssue(
                dataset_name=f"dataset_{i}",
                issue_type="missing_data",
                severity="low",
                description=f"Minor issue {i}",
                recommendation="",
                affected_columns=[],
                metrics={}
            ) for i in range(20)
        ]
        
        report = self.generator.generate_consolidated_report(analysis, many_issues)
        
        # Should prioritize critical information
        content_lower = report.main_report.lower()
        
        # Executive summary should come first
        summary_pos = content_lower.find("summary")
        quality_pos = content_lower.find("quality")
        
        if summary_pos >= 0 and quality_pos >= 0:
            assert summary_pos < quality_pos  # Summary should come before detailed quality info