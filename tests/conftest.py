"""
Test configuration and fixtures for EDA Report Consolidator tests.
"""
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pytest

from consolidator.models import (
    ReportFile, ProfileData, TableStats, VariableStats, 
    AnalysisInfo, QualityThresholds, ConsolidatorConfig
)


@pytest.fixture
def temp_reports_dir():
    """Create a temporary directory structure for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        reports_path = Path(temp_dir) / "reports"
        reports_path.mkdir()
        yield reports_path


@pytest.fixture
def quality_thresholds():
    """Standard quality thresholds for testing."""
    return QualityThresholds(
        high_missing_data_pct=20.0,
        high_duplicate_pct=1.0,
        large_dataset_rows=1_000_000,
        memory_concern_mb=100,
        unusual_type_distribution_threshold=0.8,
        sample_dataset_row_threshold=10_000
    )


@pytest.fixture
def consolidator_config(temp_reports_dir):
    """Standard configuration for testing."""
    return ConsolidatorConfig(
        reports_directory=temp_reports_dir,
        output_file=temp_reports_dir.parent / "test_report.md",
        max_report_length=5000,
        max_priority_issues_in_main_report=5
    )


@pytest.fixture
def valid_ydata_profile() -> Dict[str, Any]:
    """Valid YData Profiling JSON structure for testing."""
    return {
        "analysis": {
            "title": "Profile Report - test_data",
            "date_start": "2025-08-29 10:00:00.000000",
            "date_end": "2025-08-29 10:00:05.000000"
        },
        "table": {
            "n": 1000,
            "n_var": 5,
            "memory_size": 40000,
            "record_size": 40.0,
            "n_cells_missing": 50,
            "n_vars_with_missing": 2,
            "n_vars_all_missing": 0,
            "p_cells_missing": 1.0,
            "types": {
                "Categorical": 2,
                "DateTime": 1,
                "Numeric": 2
            },
            "n_duplicates": 5,
            "p_duplicates": 0.5
        },
        "variables": {
            "id": {
                "n_distinct": 1000,
                "p_distinct": 1.0,
                "is_unique": True,
                "n_unique": 1000,
                "p_unique": 1.0,
                "type": "Numeric",
                "n_missing": 0,
                "n": 1000,
                "p_missing": 0.0,
                "memory_size": 8000
            },
            "name": {
                "n_distinct": 950,
                "p_distinct": 0.95,
                "is_unique": False,
                "n_unique": 0,
                "p_unique": 0.0,
                "type": "Categorical",
                "n_missing": 50,
                "n": 1000,
                "p_missing": 5.0,
                "memory_size": 16000
            }
        }
    }


@pytest.fixture
def high_quality_profile() -> Dict[str, Any]:
    """High quality dataset profile for testing."""
    return {
        "analysis": {
            "title": "Profile Report - high_quality_data",
            "date_start": "2025-08-29 10:00:00.000000",
            "date_end": "2025-08-29 10:00:02.000000"
        },
        "table": {
            "n": 50000,
            "n_var": 10,
            "memory_size": 2000000,
            "record_size": 40.0,
            "n_cells_missing": 0,
            "n_vars_with_missing": 0,
            "n_vars_all_missing": 0,
            "p_cells_missing": 0.0,
            "types": {
                "Categorical": 3,
                "DateTime": 2,
                "Numeric": 5
            },
            "n_duplicates": 0,
            "p_duplicates": 0.0
        },
        "variables": {
            "customer_id": {
                "type": "Numeric",
                "n": 10000,
                "n_missing": 0,
                "p_missing": 0.0,
                "memory_size": 400000
            },
            "transaction_date": {
                "type": "DateTime",
                "n": 10000,
                "n_missing": 0,
                "p_missing": 0.0,
                "memory_size": 400000
            }
        }
    }


@pytest.fixture
def poor_quality_profile() -> Dict[str, Any]:
    """Poor quality dataset profile for testing."""
    return {
        "analysis": {
            "title": "Profile Report - poor_quality_data",
            "date_start": "2025-08-29 10:00:00.000000",
            "date_end": "2025-08-29 10:00:10.000000"
        },
        "table": {
            "n": 10000,
            "n_var": 8,
            "memory_size": 800000,
            "record_size": 80.0,
            "n_cells_missing": 20000,
            "n_vars_with_missing": 6,
            "n_vars_all_missing": 1,
            "p_cells_missing": 25.0,  # High missing data
            "types": {
                "Categorical": 4,
                "Numeric": 4
            },
            "n_duplicates": 500,
            "p_duplicates": 5.0  # High duplicates
        },
        "variables": {
            "broken_field": {
                "type": "Categorical",
                "n": 10000,
                "n_missing": 8000,
                "p_missing": 80.0,  # Very high missing data
                "memory_size": 160000
            }
        }
    }


@pytest.fixture
def invalid_json_content():
    """Invalid JSON content for error testing."""
    return '{"analysis": {"title": "Broken JSON", "incomplete": true'


@pytest.fixture
def malformed_profile():
    """Malformed YData profile missing required fields."""
    return {
        "analysis": {
            "title": "Incomplete Profile"
            # Missing date_start and date_end
        },
        "table": {
            "n": 1000
            # Missing other required fields
        }
        # Missing variables section
    }


def create_test_json_file(directory: Path, filename: str, content: Dict[str, Any]) -> Path:
    """Helper function to create test JSON files."""
    category_dir = directory / "test_category"
    category_dir.mkdir(exist_ok=True)
    
    dataset_dir = category_dir / filename.replace('.json', '')
    dataset_dir.mkdir(exist_ok=True)
    
    file_path = dataset_dir / "data_profile.json"
    with open(file_path, 'w') as f:
        json.dump(content, f, indent=2)
    
    return file_path


@pytest.fixture
def sample_report_files(temp_reports_dir, valid_ydata_profile, high_quality_profile, poor_quality_profile):
    """Create sample report files for testing."""
    files = []
    
    # Create valid profile
    files.append(create_test_json_file(
        temp_reports_dir, "valid_dataset.json", valid_ydata_profile
    ))
    
    # Create high quality profile
    files.append(create_test_json_file(
        temp_reports_dir, "high_quality_dataset.json", high_quality_profile
    ))
    
    # Create poor quality profile
    files.append(create_test_json_file(
        temp_reports_dir, "poor_quality_dataset.json", poor_quality_profile
    ))
    
    return files


@pytest.fixture
def sample_profile_data():
    """Sample ProfileData objects for testing."""
    return [
        ProfileData(
            file_path=Path("test/dataset1/data_profile.json"),
            dataset_name="dataset1",
            analysis_info=AnalysisInfo(
                title="Test Dataset 1",
                date_start=datetime(2025, 8, 29, 10, 0, 0),
                date_end=datetime(2025, 8, 29, 10, 0, 5)
            ),
            table_stats=TableStats(
                n_rows=1000,
                n_columns=5,
                memory_size=40000,
                missing_cells_pct=1.0,
                duplicate_rows_pct=0.5,
                data_types={"Numeric": 3, "Categorical": 2},
                is_sample_dataset=True
            ),
            variables=[
                VariableStats(
                    name="id",
                    data_type="Numeric",
                    missing_pct=0.0,
                    unique_values=1000,
                    memory_size=8000
                )
            ],
            data_type_distribution={"Numeric": 3, "Categorical": 2},
            quality_flags=[]
        ),
        ProfileData(
            file_path=Path("test/dataset2/data_profile.json"),
            dataset_name="dataset2",
            analysis_info=AnalysisInfo(
                title="Test Dataset 2",
                date_start=datetime(2025, 8, 29, 10, 0, 0),
                date_end=datetime(2025, 8, 29, 10, 0, 10)
            ),
            table_stats=TableStats(
                n_rows=10000,
                n_columns=8,
                memory_size=800000,
                missing_cells_pct=25.0,  # High missing data
                duplicate_rows_pct=5.0,  # High duplicates
                data_types={"Categorical": 4, "Numeric": 4},
                is_sample_dataset=False
            ),
            variables=[
                VariableStats(
                    name="broken_field",
                    data_type="Categorical",
                    missing_pct=80.0,
                    unique_values=100,
                    memory_size=160000
                )
            ],
            data_type_distribution={"Categorical": 4, "Numeric": 4},
            quality_flags=["high_missing_data", "high_duplicates"]
        )
    ]