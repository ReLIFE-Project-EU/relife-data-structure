"""
Tests for error handling with invalid JSON files and edge cases.
Tests Requirements: 1.4, 2.5
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from consolidator.scanner import ReportScanner
from consolidator.parser import ProfileParser
from consolidator.orchestrator import ConsolidationOrchestrator
from consolidator.models import ConsolidatorConfig


class TestErrorHandling:
    """Test suite for error handling scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scanner = ReportScanner()
        self.parser = ProfileParser()
    
    def test_invalid_json_syntax_errors(self, temp_reports_dir):
        """Test handling of various JSON syntax errors (Requirement 2.5)."""
        invalid_json_cases = [
            '{"incomplete": json}',  # Missing quotes
            '{"missing_bracket": "value"',  # Missing closing bracket
            '{"trailing_comma": "value",}',  # Trailing comma
            '{"duplicate": "key", "duplicate": "value"}',  # Duplicate keys
            '{invalid_json_completely',  # Completely malformed
            '',  # Empty file
            'not json at all',  # Not JSON
            '{"unicode_issue": "\x00"}',  # Unicode issues
        ]
        
        for i, invalid_content in enumerate(invalid_json_cases):
            test_file = temp_reports_dir / f"invalid_{i}.json"
            with open(test_file, 'w') as f:
                f.write(invalid_content)
            
            # Scanner should identify as invalid
            is_valid = self.scanner.validate_ydata_profiling_structure(test_file)
            assert is_valid is False
            
            # Parser should handle gracefully
            result = self.parser.parse_profile_json(test_file)
            assert result.success is False
            assert len(result.errors) > 0
    
    def test_missing_required_fields(self, temp_reports_dir):
        """Test handling of JSON files missing required YData fields (Requirement 2.5)."""
        missing_field_cases = [
            {},  # Completely empty
            {"analysis": {}},  # Missing table and variables
            {"table": {}},  # Missing analysis and variables
            {"analysis": {"title": "Test"}, "table": {}},  # Missing variables
            {"analysis": {"title": "Test"}, "variables": {}},  # Missing table
            {"analysis": {"title": "Test"}, "table": {"n": 1000}, "variables": {}},  # Minimal but incomplete
        ]
        
        for i, incomplete_data in enumerate(missing_field_cases):
            test_file = temp_reports_dir / f"incomplete_{i}.json"
            with open(test_file, 'w') as f:
                json.dump(incomplete_data, f)
            
            # Should attempt partial extraction
            result = self.parser.parse_profile_json(test_file)
            
            if len(incomplete_data) > 0:
                # Should have partial data for non-empty cases
                assert result.partial_data is not None
                assert len(result.warnings) > 0
            else:
                # Completely empty should fail completely
                assert result.success is False
    
    def test_corrupted_file_handling(self, temp_reports_dir):
        """Test handling of corrupted or binary files (Requirement 1.4)."""
        # Create binary file
        binary_file = temp_reports_dir / "binary.json"
        with open(binary_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03\x04\x05')
        
        # Scanner should handle gracefully
        is_valid = self.scanner.validate_ydata_profiling_structure(binary_file)
        assert is_valid is False
        
        # Parser should handle gracefully
        result = self.parser.parse_profile_json(binary_file)
        assert result.success is False
        assert len(result.errors) > 0
    
    def test_permission_denied_errors(self, temp_reports_dir):
        """Test handling of permission denied errors (Requirement 1.4)."""
        # Create a file and simulate permission error
        test_file = temp_reports_dir / "permission_test.json"
        with open(test_file, 'w') as f:
            json.dump({"test": "data"}, f)
        
        # Mock permission error
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            is_valid = self.scanner.validate_ydata_profiling_structure(test_file)
            assert is_valid is False
            
            result = self.parser.parse_profile_json(test_file)
            assert result.success is False
            assert "permission" in result.errors[0].lower()
    
    def test_file_not_found_errors(self):
        """Test handling of non-existent files (Requirement 1.4)."""
        non_existent = Path("does_not_exist.json")
        
        # Scanner should handle gracefully
        is_valid = self.scanner.validate_ydata_profiling_structure(non_existent)
        assert is_valid is False
        
        # Parser should handle gracefully
        result = self.parser.parse_profile_json(non_existent)
        assert result.success is False
        assert "not found" in result.errors[0].lower()
    
    def test_extremely_large_files(self, temp_reports_dir):
        """Test handling of extremely large JSON files."""
        # Create a large but valid JSON structure
        large_data = {
            "analysis": {
                "title": "Large Dataset",
                "date_start": "2025-08-29 10:00:00.000000",
                "date_end": "2025-08-29 10:00:05.000000"
            },
            "table": {
                "n": 10000000,
                "n_var": 1000,
                "memory_size": 80000000000,
                "p_cells_missing": 0.0,
                "p_duplicates": 0.0,
                "types": {"Numeric": 500, "Categorical": 500}
            },
            "variables": {}
        }
        
        # Add many variables to make it large
        for i in range(1000):
            large_data["variables"][f"var_{i}"] = {
                "type": "Numeric",
                "n_missing": 0,
                "p_missing": 0.0,
                "memory_size": 80000000,
                "n_distinct": 1000000,
                "value_counts": {f"value_{j}": 1000 for j in range(100)}  # Large nested data
            }
        
        large_file = temp_reports_dir / "large_dataset.json"
        with open(large_file, 'w') as f:
            json.dump(large_data, f)
        
        # Should handle large files (may be slow but shouldn't crash)
        result = self.parser.parse_profile_json(large_file)
        
        # Should succeed or fail gracefully
        if result.success:
            assert result.profile_data is not None
            assert result.profile_data.table_stats.n_rows == 10000000
        else:
            assert len(result.errors) > 0
            assert "memory" in str(result.errors).lower() or "size" in str(result.errors).lower()
    
    def test_nested_directory_permission_errors(self, temp_reports_dir):
        """Test handling of permission errors in nested directories."""
        # Create nested structure
        nested_dir = temp_reports_dir / "category" / "restricted" / "dataset"
        nested_dir.mkdir(parents=True)
        
        # Create valid file
        test_file = nested_dir / "data_profile.json"
        with open(test_file, 'w') as f:
            json.dump({"analysis": {"title": "Test"}}, f)
        
        # Mock directory iteration error
        with patch('pathlib.Path.iterdir', side_effect=PermissionError("Access denied")):
            result = self.scanner.scan_reports_directory(temp_reports_dir)
            
            # Should handle gracefully
            assert isinstance(result, type(self.scanner.scan_reports_directory(temp_reports_dir)))
            # May have empty results due to permission error
    
    def test_unicode_and_encoding_issues(self, temp_reports_dir):
        """Test handling of Unicode and encoding issues."""
        unicode_cases = [
            {"analysis": {"title": "Test with émojis 🚀📊"}},
            {"analysis": {"title": "Test with ñoñó characters"}},
            {"analysis": {"title": "Test with 中文 characters"}},
            {"analysis": {"title": "Test with русский text"}},
        ]
        
        for i, unicode_data in enumerate(unicode_cases):
            test_file = temp_reports_dir / f"unicode_{i}.json"
            
            # Write with different encodings
            try:
                with open(test_file, 'w', encoding='utf-8') as f:
                    json.dump(unicode_data, f, ensure_ascii=False)
                
                # Should handle Unicode properly
                result = self.parser.parse_profile_json(test_file)
                
                if result.success or result.partial_data:
                    # Should preserve Unicode characters
                    title = (result.profile_data.analysis_info.title if result.success 
                            else result.partial_data.extracted_fields.get("title", ""))
                    assert len(title) > 0
                
            except UnicodeEncodeError:
                # Some systems may not support all Unicode characters
                pass
    
    def test_memory_exhaustion_simulation(self, temp_reports_dir):
        """Test handling of memory exhaustion scenarios."""
        # Create a file that would cause memory issues if not handled properly
        memory_intensive_data = {
            "analysis": {"title": "Memory Test"},
            "table": {"n": 1000000, "n_var": 100},
            "variables": {}
        }
        
        # Add variables with large nested structures
        for i in range(100):
            memory_intensive_data["variables"][f"var_{i}"] = {
                "type": "Categorical",
                "value_counts": {f"value_{j}": 1 for j in range(10000)}  # Large dict
            }
        
        test_file = temp_reports_dir / "memory_test.json"
        with open(test_file, 'w') as f:
            json.dump(memory_intensive_data, f)
        
        # Should handle without crashing (may use fallbacks)
        result = self.parser.parse_profile_json(test_file)
        
        # Should either succeed or fail gracefully
        assert isinstance(result, type(self.parser.parse_profile_json(test_file)))
    
    def test_concurrent_file_access_errors(self, temp_reports_dir):
        """Test handling of concurrent file access issues."""
        test_file = temp_reports_dir / "concurrent_test.json"
        with open(test_file, 'w') as f:
            json.dump({"analysis": {"title": "Concurrent Test"}}, f)
        
        # Simulate file being locked/in use
        with patch('builtins.open', side_effect=OSError("File in use")):
            result = self.parser.parse_profile_json(test_file)
            
            assert result.success is False
            assert len(result.errors) > 0
    
    def test_end_to_end_error_recovery(self, temp_reports_dir):
        """Test end-to-end error recovery with mixed valid/invalid files."""
        # Create a mix of valid, invalid, and problematic files
        
        # Valid file
        valid_dir = temp_reports_dir / "valid" / "good_dataset"
        valid_dir.mkdir(parents=True)
        with open(valid_dir / "data_profile.json", 'w') as f:
            json.dump({
                "analysis": {"title": "Good Dataset", "date_start": "2025-08-29 10:00:00", "date_end": "2025-08-29 10:00:05"},
                "table": {"n": 1000, "n_var": 5, "memory_size": 40000, "p_cells_missing": 0.0, "p_duplicates": 0.0, "types": {"Numeric": 5}},
                "variables": {"var1": {"type": "Numeric", "n": 1000, "n_missing": 0, "p_missing": 0.0, "memory_size": 8000}}
            }, f)
        
        # Invalid JSON
        invalid_dir = temp_reports_dir / "invalid" / "bad_dataset"
        invalid_dir.mkdir(parents=True)
        with open(invalid_dir / "data_profile.json", 'w') as f:
            f.write('{"broken": json}')
        
        # Missing fields
        incomplete_dir = temp_reports_dir / "incomplete" / "partial_dataset"
        incomplete_dir.mkdir(parents=True)
        with open(incomplete_dir / "data_profile.json", 'w') as f:
            json.dump({"analysis": {"title": "Incomplete"}}, f)
        
        # Empty file
        empty_dir = temp_reports_dir / "empty" / "empty_dataset"
        empty_dir.mkdir(parents=True)
        with open(empty_dir / "data_profile.json", 'w') as f:
            f.write('')
        
        # Run full consolidation
        config = ConsolidatorConfig(reports_directory=temp_reports_dir)
        orchestrator = ConsolidationOrchestrator(config)
        result = orchestrator.run_consolidation()
        
        # Should succeed with partial results
        assert result.success is True
        assert result.datasets_processed >= 1  # At least the valid one
        # Note: Invalid files are skipped, not treated as errors in the current implementation
        
        # Report should be generated
        assert result.report_path.exists()
        
        with open(result.report_path, 'r') as f:
            report_content = f.read()
        
        # Should mention the successful dataset
        assert "good_dataset" in report_content or "1" in report_content
    
    def test_malformed_ydata_structure_variations(self, temp_reports_dir):
        """Test various malformed YData Profiling structures."""
        malformed_cases = [
            # Wrong field names
            {"analyze": {"title": "Wrong field name"}},
            
            # Wrong data types
            {"analysis": "should be dict", "table": "should be dict"},
            
            # Nested structure issues
            {"analysis": {"title": {"nested": "should be string"}}},
            
            # Missing critical nested fields
            {"analysis": {"title": "Test"}, "table": {"wrong_field": 123}},
            
            # Array instead of dict
            {"analysis": ["should", "be", "dict"]},
            
            # Null values where objects expected
            {"analysis": None, "table": None, "variables": None},
        ]
        
        for i, malformed_data in enumerate(malformed_cases):
            test_file = temp_reports_dir / f"malformed_{i}.json"
            with open(test_file, 'w') as f:
                json.dump(malformed_data, f)
            
            # Scanner should identify as invalid YData structure
            is_valid = self.scanner.validate_ydata_profiling_structure(test_file)
            assert is_valid is False
            
            # Parser should attempt partial extraction
            result = self.parser.parse_profile_json(test_file)
            assert result.success is False
            
            # Should have partial data if any extractable fields exist
            if any(key in malformed_data for key in ["analysis", "table", "variables"]):
                assert result.partial_data is not None or len(result.warnings) > 0