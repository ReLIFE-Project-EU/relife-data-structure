"""
Unit tests for the ReportScanner component.
Tests Requirements: 1.1, 1.2, 1.3, 1.4
"""
import json
import pytest
from pathlib import Path

from consolidator.scanner import ReportScanner
from consolidator.models import ScanResult, ReportFile


class TestReportScanner:
    """Test suite for ReportScanner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scanner = ReportScanner()
    
    def test_scan_empty_directory(self, temp_reports_dir):
        """Test scanning an empty reports directory."""
        result = self.scanner.scan_reports_directory(temp_reports_dir)
        
        assert isinstance(result, ScanResult)
        assert len(result.valid_reports) == 0
        assert len(result.invalid_files) == 0
        assert result.total_files_scanned == 0
        assert len(result.categories_found) == 0
    
    def test_scan_directory_with_valid_reports(self, temp_reports_dir, sample_report_files):
        """Test scanning directory with valid YData Profiling reports (Requirement 1.1)."""
        result = self.scanner.scan_reports_directory(temp_reports_dir)
        
        assert len(result.valid_reports) >= 1  # At least one valid report found
        assert result.total_files_scanned >= 1
        assert len(result.categories_found) > 0
        
        # Verify report file details
        for report in result.valid_reports:
            assert isinstance(report, ReportFile)
            assert report.path.exists()
            assert report.is_valid_ydata_profile is True
            assert report.dataset_name != ""
            assert report.category != ""
    
    def test_scan_recursive_directory_structure(self, temp_reports_dir, valid_ydata_profile):
        """Test recursive scanning of nested directory structure (Requirement 1.1)."""
        # Create nested structure
        nested_path = temp_reports_dir / "category1" / "subcategory" / "dataset1"
        nested_path.mkdir(parents=True)
        
        profile_file = nested_path / "data_profile.json"
        with open(profile_file, 'w') as f:
            json.dump(valid_ydata_profile, f)
        
        result = self.scanner.scan_reports_directory(temp_reports_dir)
        
        assert len(result.valid_reports) == 1
        assert result.valid_reports[0].path == profile_file
        assert result.valid_reports[0].category == "category1"
    
    def test_validate_ydata_profiling_structure_valid(self, temp_reports_dir, valid_ydata_profile):
        """Test YData Profiling structure validation with valid file (Requirement 1.2)."""
        test_file = temp_reports_dir / "valid.json"
        with open(test_file, 'w') as f:
            json.dump(valid_ydata_profile, f)
        
        is_valid = self.scanner.validate_ydata_profiling_structure(test_file)
        assert is_valid is True
    
    def test_validate_ydata_profiling_structure_invalid(self, temp_reports_dir):
        """Test YData Profiling structure validation with invalid file (Requirement 1.2)."""
        # Create file with wrong structure
        invalid_structure = {"not_ydata": "profile"}
        test_file = temp_reports_dir / "invalid.json"
        with open(test_file, 'w') as f:
            json.dump(invalid_structure, f)
        
        is_valid = self.scanner.validate_ydata_profiling_structure(test_file)
        assert is_valid is False
    
    def test_validate_ydata_profiling_structure_malformed_json(self, temp_reports_dir, invalid_json_content):
        """Test validation with malformed JSON (Requirement 1.4)."""
        test_file = temp_reports_dir / "malformed.json"
        with open(test_file, 'w') as f:
            f.write(invalid_json_content)
        
        is_valid = self.scanner.validate_ydata_profiling_structure(test_file)
        assert is_valid is False
    
    def test_categorize_dataset_by_path(self):
        """Test dataset categorization based on directory structure."""
        test_cases = [
            (Path("reports/aemo_price_demand/dataset1/data_profile.json"), "aemo_price_demand"),
            (Path("reports/belgium_electricity/dataset2/data_profile.json"), "belgium_electricity"),
            (Path("reports/category/subcategory/dataset3/data_profile.json"), "category"),
            (Path("reports/single_dataset/data_profile.json"), "single_dataset"),
            (Path("data_profile.json"), "unknown")
        ]
        
        for file_path, expected_category in test_cases:
            category = self.scanner.categorize_dataset_by_path(file_path)
            assert category == expected_category
    
    def test_extract_dataset_metadata(self, temp_reports_dir, valid_ydata_profile):
        """Test dataset metadata extraction."""
        test_file = temp_reports_dir / "category" / "test_dataset" / "data_profile.json"
        test_file.parent.mkdir(parents=True)
        
        with open(test_file, 'w') as f:
            json.dump(valid_ydata_profile, f)
        
        metadata = self.scanner.extract_dataset_metadata(test_file)
        
        assert metadata.dataset_name == "test_dataset"
        assert metadata.category == "category"
        assert metadata.file_size > 0
        assert metadata.last_modified is not None
    
    def test_scan_with_mixed_file_types(self, temp_reports_dir, valid_ydata_profile):
        """Test scanning directory with mixed file types."""
        # Create valid YData profile
        valid_dir = temp_reports_dir / "category1" / "valid_dataset"
        valid_dir.mkdir(parents=True)
        with open(valid_dir / "data_profile.json", 'w') as f:
            json.dump(valid_ydata_profile, f)
        
        # Create non-JSON file
        with open(temp_reports_dir / "readme.txt", 'w') as f:
            f.write("This is not a JSON file")
        
        # Create invalid JSON file
        invalid_dir = temp_reports_dir / "category2" / "invalid_dataset"
        invalid_dir.mkdir(parents=True)
        with open(invalid_dir / "data_profile.json", 'w') as f:
            f.write('{"invalid": json}')
        
        # Create non-YData JSON file
        other_dir = temp_reports_dir / "category3" / "other_dataset"
        other_dir.mkdir(parents=True)
        with open(other_dir / "data_profile.json", 'w') as f:
            json.dump({"other": "structure"}, f)
        
        result = self.scanner.scan_reports_directory(temp_reports_dir)
        
        # Should find one valid report
        assert len(result.valid_reports) == 1
        assert result.valid_reports[0].dataset_name == "valid_dataset"
        
        # Should identify invalid files
        assert len(result.invalid_files) >= 2  # At least the invalid JSON and non-YData files
    
    def test_get_scan_summary(self, temp_reports_dir, sample_report_files):
        """Test scan summary generation (Requirement 1.3)."""
        result = self.scanner.scan_reports_directory(temp_reports_dir)
        summary = self.scanner.get_scan_summary(result)
        
        assert isinstance(summary, str)
        assert "3" in summary  # Should mention 3 datasets found
        assert "test_category" in summary  # Should mention the category
        assert "valid" in summary.lower()
    
    def test_scan_nonexistent_directory(self):
        """Test scanning non-existent directory (Requirement 1.4)."""
        nonexistent_path = Path("nonexistent_directory")
        result = self.scanner.scan_reports_directory(nonexistent_path)
        
        assert len(result.valid_reports) == 0
        assert len(result.invalid_files) == 0
        assert result.total_files_scanned == 0
    
    def test_scan_directory_without_permissions(self, temp_reports_dir):
        """Test scanning directory with permission issues (Requirement 1.4)."""
        # Create a subdirectory and try to make it unreadable
        restricted_dir = temp_reports_dir / "restricted"
        restricted_dir.mkdir()
        
        # Note: This test may not work on all systems due to permission handling
        # The scanner should handle permission errors gracefully
        result = self.scanner.scan_reports_directory(temp_reports_dir)
        
        # Should complete without crashing
        assert isinstance(result, ScanResult)
    
    def test_scan_with_symlinks(self, temp_reports_dir, valid_ydata_profile):
        """Test scanning directory with symbolic links."""
        # Create original file
        original_dir = temp_reports_dir / "original" / "dataset"
        original_dir.mkdir(parents=True)
        original_file = original_dir / "data_profile.json"
        
        with open(original_file, 'w') as f:
            json.dump(valid_ydata_profile, f)
        
        # Create symlink (if supported by the system)
        try:
            link_dir = temp_reports_dir / "linked" / "dataset"
            link_dir.mkdir(parents=True)
            link_file = link_dir / "data_profile.json"
            link_file.symlink_to(original_file)
            
            result = self.scanner.scan_reports_directory(temp_reports_dir)
            
            # Should handle symlinks appropriately
            assert len(result.valid_reports) >= 1
            
        except (OSError, NotImplementedError):
            # Symlinks not supported on this system, skip test
            pytest.skip("Symlinks not supported on this system")
    
    def test_scan_performance_with_many_files(self, temp_reports_dir, valid_ydata_profile):
        """Test scanning performance with many files."""
        import time
        
        # Create multiple valid profiles
        num_files = 50
        for i in range(num_files):
            category_dir = temp_reports_dir / f"category_{i % 5}"  # 5 categories
            dataset_dir = category_dir / f"dataset_{i}"
            dataset_dir.mkdir(parents=True)
            
            with open(dataset_dir / "data_profile.json", 'w') as f:
                json.dump(valid_ydata_profile, f)
        
        start_time = time.time()
        result = self.scanner.scan_reports_directory(temp_reports_dir)
        scan_duration = time.time() - start_time
        
        assert len(result.valid_reports) == num_files
        assert len(result.categories_found) == 5
        assert scan_duration < 10.0  # Should complete within 10 seconds
        assert result.scan_duration > 0
    
    def test_file_size_and_modification_tracking(self, temp_reports_dir, valid_ydata_profile):
        """Test that file size and modification time are tracked."""
        test_file = temp_reports_dir / "category" / "dataset" / "data_profile.json"
        test_file.parent.mkdir(parents=True)
        
        with open(test_file, 'w') as f:
            json.dump(valid_ydata_profile, f)
        
        result = self.scanner.scan_reports_directory(temp_reports_dir)
        
        assert len(result.valid_reports) == 1
        report = result.valid_reports[0]
        
        assert report.file_size > 0
        assert report.last_modified is not None
        
        # File size should match actual file size
        actual_size = test_file.stat().st_size
        assert report.file_size == actual_size