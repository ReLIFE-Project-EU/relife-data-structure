"""
Data analysis and aggregation engine for the EDA Report Consolidator.

This module provides comprehensive analysis of parsed dataset collections,
including size distribution, schema pattern detection, and category-based insights.
"""

import re
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from .models import (
    CategoryInsight,
    ColumnNamingAnalysis,
    DatasetAnalysis,
    ProfileData,
    QualitySummary,
    SchemaAnalysis,
    SchemaInconsistency,
    SchemaPattern,
    SchemaRecommendation,
    SizeDistribution,
    StandardizationOpportunity,
)


class DataAnalyzer:
    """
    Analyzes dataset collections to identify patterns, compute aggregations,
    and derive actionable insights for database administrators.

    This class provides comprehensive analysis capabilities including:
    - Dataset size distribution analysis
    - Schema pattern detection across similar datasets
    - Category-based grouping and insights generation
    - Data type distribution analysis
    - Schema standardization opportunity identification

    The analyzer is designed to help database administrators understand:
    - Overall data landscape structure
    - Common patterns and schemas across datasets
    - Opportunities for standardization and optimization
    - Quality trends by category

    Implements Requirements:
    - 3.1: Group datasets by logical categories based on directory structure
    - 3.4: Include schema information showing common data types and patterns
    - 6.3: Identify patterns across similar dataset types

    Attributes:
        _size_thresholds (Dict[str, int]): Thresholds for categorizing dataset sizes
            - 'small': 1MB (1,048,576 bytes)
            - 'medium': 100MB (104,857,600 bytes)

    Example:
        >>> analyzer = DataAnalyzer()
        >>> analysis = analyzer.analyze_dataset_collection(parsed_datasets)
        >>> print(f"Found {analysis.total_datasets} datasets in {len(analysis.category_breakdown)} categories")
    """

    def __init__(self):
        """Initialize the DataAnalyzer."""
        self._size_thresholds = {
            "small": 1_048_576,  # 1MB in bytes
            "medium": 104_857_600,  # 100MB in bytes
        }

    def analyze_dataset_collection(
        self, datasets: List[ProfileData]
    ) -> DatasetAnalysis:
        """
        Perform comprehensive analysis of the dataset collection.

        This method orchestrates the complete analysis workflow including:
        - Size distribution computation
        - Schema analysis across all datasets
        - Pattern identification within categories
        - Quality trend analysis
        - Standardization opportunity detection

        Args:
            datasets (List[ProfileData]): List of parsed ProfileData objects from YData Profiling reports.
                Each ProfileData should contain complete table statistics, variable information,
                and data type distributions.

        Returns:
            DatasetAnalysis: Complete analysis results containing:
                - total_datasets: Number of datasets analyzed
                - analysis_timestamp: When the analysis was performed
                - size_distribution: Breakdown of dataset sizes (small/medium/large)
                - quality_summary: Overall quality metrics and trends
                - schema_analysis: Comprehensive schema information and patterns
                - schema_patterns: Common patterns identified across datasets
                - category_breakdown: Number of datasets per category
                - category_insights: Detailed insights for each category
                - sample_vs_complete: Classification of datasets as samples or complete
                - standardization_opportunities: Identified opportunities for schema alignment

        Raises:
            ValueError: If datasets list is None
            TypeError: If datasets contains non-ProfileData objects

        Example:
            >>> datasets = [profile1, profile2, profile3]
            >>> analysis = analyzer.analyze_dataset_collection(datasets)
            >>> print(f"Analysis found {len(analysis.schema_patterns)} common patterns")
        """
        if not datasets:
            return self._create_empty_analysis()

        # Group datasets by category for analysis
        category_groups = self.group_by_category(datasets)

        # Compute core analyses
        size_distribution = self.compute_size_distribution(datasets)
        schema_analysis = self.analyze_schema_information(datasets)
        schema_patterns = self.identify_schema_patterns(datasets)
        category_insights = self.generate_category_insights(category_groups)
        sample_vs_complete = self.detect_sample_vs_complete_datasets(datasets)
        standardization_opportunities = self.find_schema_standardization_opportunities(
            datasets
        )

        # Create quality summary (basic implementation for analyzer)
        quality_summary = self._create_basic_quality_summary(datasets)

        return DatasetAnalysis(
            total_datasets=len(datasets),
            analysis_timestamp=datetime.now(),
            size_distribution=size_distribution,
            quality_summary=quality_summary,
            schema_analysis=schema_analysis,
            schema_patterns=schema_patterns,
            category_breakdown={
                cat: len(group) for cat, group in category_groups.items()
            },
            category_insights=category_insights,
            sample_vs_complete=sample_vs_complete,
            standardization_opportunities=standardization_opportunities,
        )

    def compute_size_distribution(
        self, datasets: List[ProfileData]
    ) -> SizeDistribution:
        """
        Compute size distribution categorizing datasets as small/medium/large.

        Args:
            datasets: List of ProfileData objects

        Returns:
            SizeDistribution with categorized counts and statistics
        """
        small_count = 0
        medium_count = 0
        large_count = 0
        total_memory = 0
        total_rows = 0
        total_columns = 0

        for dataset in datasets:
            memory_size = dataset.table_stats.memory_size
            total_memory += memory_size
            total_rows += dataset.table_stats.n_rows
            total_columns += dataset.table_stats.n_columns

            if memory_size < self._size_thresholds["small"]:
                small_count += 1
            elif memory_size < self._size_thresholds["medium"]:
                medium_count += 1
            else:
                large_count += 1

        num_datasets = len(datasets)
        avg_rows = total_rows / num_datasets if num_datasets > 0 else 0
        avg_columns = total_columns / num_datasets if num_datasets > 0 else 0

        return SizeDistribution(
            small_datasets=small_count,
            medium_datasets=medium_count,
            large_datasets=large_count,
            total_memory_usage=total_memory,
            average_rows=avg_rows,
            average_columns=avg_columns,
        )

    def group_by_category(
        self, datasets: List[ProfileData]
    ) -> Dict[str, List[ProfileData]]:
        """
        Group datasets by their logical categories based on directory structure.

        Args:
            datasets: List of ProfileData objects

        Returns:
            Dictionary mapping category names to lists of datasets
        """
        category_groups = defaultdict(list)

        for dataset in datasets:
            # Extract category from file path (parent directory name)
            path_parts = dataset.file_path.parts

            # Find the category by looking for the reports directory and taking the next part
            category = "uncategorized"
            for i, part in enumerate(path_parts):
                if part == "reports" and i + 1 < len(path_parts):
                    category = path_parts[i + 1]
                    break

            category_groups[category].append(dataset)

        return dict(category_groups)

    def analyze_schema_information(self, datasets: List[ProfileData]) -> SchemaAnalysis:
        """
        Analyze schema information across all datasets.

        Args:
            datasets: List of ProfileData objects

        Returns:
            SchemaAnalysis with comprehensive schema insights
        """
        # Collect all data types across datasets
        common_data_types = Counter()
        all_column_names = []
        unique_column_names = set()
        data_type_by_category = defaultdict(lambda: Counter())

        # Group by category for category-specific analysis
        category_groups = self.group_by_category(datasets)

        for dataset in datasets:
            # Count data types
            for data_type, count in dataset.data_type_distribution.items():
                common_data_types[data_type] += count

            # Collect column names
            for variable in dataset.variables:
                column_name = variable.name
                all_column_names.append(column_name)
                unique_column_names.add(column_name)

                # Get category for this dataset
                category = self._get_dataset_category(dataset)
                data_type_by_category[category][variable.data_type] += 1

        # Analyze column naming patterns
        column_naming_analysis = self.analyze_column_naming_patterns(datasets)

        # Detect schema inconsistencies
        schema_inconsistencies = self.detect_schema_inconsistencies(datasets)

        # Generate schema recommendations
        schema_recommendations = self.generate_schema_recommendations(
            column_naming_analysis, schema_inconsistencies
        )

        # Find most common columns
        column_counter = Counter(all_column_names)
        most_common_columns = column_counter.most_common(20)  # Top 20 most common

        return SchemaAnalysis(
            common_data_types=dict(common_data_types),
            column_naming_analysis=column_naming_analysis,
            schema_inconsistencies=schema_inconsistencies,
            schema_recommendations=schema_recommendations,
            data_type_distribution_by_category=dict(data_type_by_category),
            unique_column_names=unique_column_names,
            most_common_columns=most_common_columns,
        )

    def identify_schema_patterns(
        self, datasets: List[ProfileData]
    ) -> List[SchemaPattern]:
        """
        Identify common schema patterns across similar dataset structures.

        Args:
            datasets: List of ProfileData objects

        Returns:
            List of SchemaPattern objects representing common structures
        """
        patterns = []

        # Group datasets by category first
        category_groups = self.group_by_category(datasets)

        for category, category_datasets in category_groups.items():
            if len(category_datasets) < 2:
                continue  # Need at least 2 datasets to identify patterns

            # Find common columns within this category
            column_sets = []
            for dataset in category_datasets:
                column_set = {var.name for var in dataset.variables}
                column_sets.append(column_set)

            # Find intersection of all column sets (columns present in all datasets)
            if column_sets:
                common_columns = set.intersection(*column_sets)

                if len(common_columns) >= 3:  # Only consider meaningful patterns
                    # Determine common data types for these columns
                    common_data_types = {}
                    for col_name in common_columns:
                        data_types = []
                        for dataset in category_datasets:
                            for var in dataset.variables:
                                if var.name == col_name:
                                    data_types.append(var.data_type)
                                    break

                        # Use most common data type for this column
                        if data_types:
                            most_common_type = Counter(data_types).most_common(1)[0][0]
                            common_data_types[col_name] = most_common_type

                    pattern = SchemaPattern(
                        pattern_name=f"{category}_common_schema",
                        datasets=[d.dataset_name for d in category_datasets],
                        common_columns=sorted(list(common_columns)),
                        data_types=common_data_types,
                    )
                    patterns.append(pattern)

        return patterns

    def analyze_column_naming_patterns(
        self, datasets: List[ProfileData]
    ) -> ColumnNamingAnalysis:
        """
        Analyze column naming patterns and identify inconsistencies.

        Args:
            datasets: List of ProfileData objects

        Returns:
            ColumnNamingAnalysis with naming patterns and suggestions
        """
        all_columns = []
        for dataset in datasets:
            for variable in dataset.variables:
                all_columns.append(variable.name)

        # Identify naming conventions
        naming_conventions = {
            "snake_case": [],
            "camelCase": [],
            "PascalCase": [],
            "kebab-case": [],
            "UPPER_CASE": [],
            "mixed": [],
        }

        snake_case_pattern = re.compile(r"^[a-z]+(_[a-z0-9]+)*$")
        camel_case_pattern = re.compile(r"^[a-z]+([A-Z][a-z0-9]*)*$")
        pascal_case_pattern = re.compile(r"^[A-Z][a-z0-9]*([A-Z][a-z0-9]*)*$")
        kebab_case_pattern = re.compile(r"^[a-z]+(-[a-z0-9]+)*$")
        upper_case_pattern = re.compile(r"^[A-Z]+(_[A-Z0-9]+)*$")

        for col in set(all_columns):  # Use set to avoid duplicates
            if snake_case_pattern.match(col):
                naming_conventions["snake_case"].append(col)
            elif camel_case_pattern.match(col):
                naming_conventions["camelCase"].append(col)
            elif pascal_case_pattern.match(col):
                naming_conventions["PascalCase"].append(col)
            elif kebab_case_pattern.match(col):
                naming_conventions["kebab-case"].append(col)
            elif upper_case_pattern.match(col):
                naming_conventions["UPPER_CASE"].append(col)
            else:
                naming_conventions["mixed"].append(col)

        # Find similar column names that might be inconsistent
        inconsistent_naming = self._find_similar_column_names(all_columns)

        # Generate standardization suggestions
        standardization_suggestions = self._generate_naming_suggestions(
            inconsistent_naming
        )

        return ColumnNamingAnalysis(
            naming_conventions=naming_conventions,
            inconsistent_naming=inconsistent_naming,
            standardization_suggestions=standardization_suggestions,
        )

    def detect_schema_inconsistencies(
        self, datasets: List[ProfileData]
    ) -> List[SchemaInconsistency]:
        """
        Detect inconsistencies in schema across datasets.

        Args:
            datasets: List of ProfileData objects

        Returns:
            List of SchemaInconsistency objects
        """
        inconsistencies = []

        # Group by category to find inconsistencies within similar datasets
        category_groups = self.group_by_category(datasets)

        for category, category_datasets in category_groups.items():
            if len(category_datasets) < 2:
                continue

            # Check for data type mismatches for same column names
            column_types = defaultdict(set)
            column_datasets = defaultdict(list)

            for dataset in category_datasets:
                for variable in dataset.variables:
                    col_name = variable.name
                    column_types[col_name].add(variable.data_type)
                    column_datasets[col_name].append(dataset.dataset_name)

            # Find columns with multiple data types
            for col_name, data_types in column_types.items():
                if len(data_types) > 1:
                    inconsistency = SchemaInconsistency(
                        inconsistency_type="data_type_mismatch",
                        affected_datasets=column_datasets[col_name],
                        description=f"Column '{col_name}' has different data types: {', '.join(data_types)}",
                        severity="medium",
                        column_details={
                            "column_name": col_name,
                            "data_types": list(data_types),
                        },
                    )
                    inconsistencies.append(inconsistency)

        return inconsistencies

    def generate_schema_recommendations(
        self,
        column_analysis: ColumnNamingAnalysis,
        inconsistencies: List[SchemaInconsistency],
    ) -> List[SchemaRecommendation]:
        """
        Generate recommendations for schema standardization.

        Args:
            column_analysis: Column naming analysis results
            inconsistencies: List of detected schema inconsistencies

        Returns:
            List of SchemaRecommendation objects
        """
        recommendations = []

        # Recommend naming standardization if multiple conventions are used
        convention_counts = {
            conv: len(cols)
            for conv, cols in column_analysis.naming_conventions.items()
            if cols  # Only count non-empty conventions
        }

        if len(convention_counts) > 1:
            most_common_convention = max(convention_counts, key=convention_counts.get)  # type: ignore
            recommendation = SchemaRecommendation(
                recommendation_type="standardize_naming",
                priority="medium",
                description=f"Standardize column naming to {most_common_convention} convention",
                affected_datasets=[],  # Would need to track which datasets use which conventions
                implementation_notes=f"Most columns use {most_common_convention}. Consider renaming columns in other conventions.",
            )
            recommendations.append(recommendation)

        # Recommend data type unification for inconsistencies
        for inconsistency in inconsistencies:
            if inconsistency.inconsistency_type == "data_type_mismatch":
                recommendation = SchemaRecommendation(
                    recommendation_type="unify_data_types",
                    priority="high",
                    description=f"Unify data type for column '{inconsistency.column_details['column_name']}'",
                    affected_datasets=inconsistency.affected_datasets,
                    implementation_notes=f"Choose consistent data type from: {', '.join(inconsistency.column_details['data_types'])}",
                )
                recommendations.append(recommendation)

        return recommendations

    def generate_category_insights(
        self, category_groups: Dict[str, List[ProfileData]]
    ) -> Dict[str, CategoryInsight]:
        """
        Generate insights for each dataset category.

        Args:
            category_groups: Dictionary mapping categories to dataset lists

        Returns:
            Dictionary mapping category names to CategoryInsight objects
        """
        insights = {}

        for category, datasets in category_groups.items():
            if not datasets:
                continue

            # Analyze common patterns in this category
            common_patterns = []

            # Check for common column patterns
            all_columns = set()
            for dataset in datasets:
                for variable in dataset.variables:
                    all_columns.add(variable.name)

            if len(all_columns) > 0:
                # Find columns that appear in most datasets
                column_frequency = Counter()
                for dataset in datasets:
                    dataset_columns = {var.name for var in dataset.variables}
                    for col in dataset_columns:
                        column_frequency[col] += 1

                # Columns that appear in >50% of datasets
                threshold = len(datasets) * 0.5
                common_cols = [
                    col for col, freq in column_frequency.items() if freq > threshold
                ]
                if common_cols:
                    common_patterns.append(
                        f"Common columns: {', '.join(common_cols[:5])}"
                    )

            # Analyze data quality trends
            quality_issues = 0
            total_missing_pct = 0
            for dataset in datasets:
                if dataset.table_stats.missing_cells_pct > 10:  # Arbitrary threshold
                    quality_issues += 1
                total_missing_pct += dataset.table_stats.missing_cells_pct

            avg_missing_pct = total_missing_pct / len(datasets)
            if avg_missing_pct > 15:
                quality_trends = f"High missing data (avg {avg_missing_pct:.1f}%)"
            elif avg_missing_pct > 5:
                quality_trends = f"Moderate missing data (avg {avg_missing_pct:.1f}%)"
            else:
                quality_trends = (
                    f"Good data completeness (avg {avg_missing_pct:.1f}% missing)"
                )

            # Generate recommendations
            recommendations = []
            if quality_issues > len(datasets) * 0.3:  # >30% have quality issues
                recommendations.append(
                    "Review data collection processes for this category"
                )

            if len(common_patterns) > 0:
                recommendations.append(
                    "Consider standardizing schema across datasets in this category"
                )

            insights[category] = CategoryInsight(
                category_name=category,
                dataset_count=len(datasets),
                common_patterns=common_patterns,
                quality_trends=quality_trends,
                recommendations=recommendations,
            )

        return insights

    def detect_sample_vs_complete_datasets(
        self, datasets: List[ProfileData]
    ) -> Dict[str, str]:
        """
        Detect which datasets appear to be samples vs complete datasets.

        Args:
            datasets: List of ProfileData objects

        Returns:
            Dictionary mapping dataset names to 'sample' or 'complete'
        """
        sample_vs_complete = {}

        # Simple heuristic: datasets with "sample" in name or small row counts
        for dataset in datasets:
            dataset_name = dataset.dataset_name.lower()
            row_count = dataset.table_stats.n_rows

            if "sample" in dataset_name or row_count < 10000:
                sample_vs_complete[dataset.dataset_name] = "sample"
            else:
                sample_vs_complete[dataset.dataset_name] = "complete"

        return sample_vs_complete

    def find_schema_standardization_opportunities(
        self, datasets: List[ProfileData]
    ) -> List[StandardizationOpportunity]:
        """
        Find opportunities for schema or data standardization.

        Args:
            datasets: List of ProfileData objects

        Returns:
            List of StandardizationOpportunity objects
        """
        opportunities = []

        # Group by category to find standardization opportunities
        category_groups = self.group_by_category(datasets)

        for category, category_datasets in category_groups.items():
            if len(category_datasets) < 2:
                continue

            # Find datasets with similar but not identical schemas
            column_sets = {}
            for dataset in category_datasets:
                columns = {var.name for var in dataset.variables}
                column_sets[dataset.dataset_name] = columns

            # Look for datasets with high column overlap but not identical
            dataset_names = list(column_sets.keys())
            for i in range(len(dataset_names)):
                for j in range(i + 1, len(dataset_names)):
                    name1, name2 = dataset_names[i], dataset_names[j]
                    cols1, cols2 = column_sets[name1], column_sets[name2]

                    # Calculate similarity
                    intersection = len(cols1 & cols2)
                    union = len(cols1 | cols2)
                    similarity = intersection / union if union > 0 else 0

                    # If 70-95% similar, it's a standardization opportunity
                    if 0.7 <= similarity < 0.95:
                        missing_in_1 = cols2 - cols1
                        missing_in_2 = cols1 - cols2

                        description = f"Datasets have {similarity:.1%} column overlap"
                        if missing_in_1:
                            description += f". {name1} missing: {', '.join(list(missing_in_1)[:3])}"
                        if missing_in_2:
                            description += f". {name2} missing: {', '.join(list(missing_in_2)[:3])}"

                        opportunity = StandardizationOpportunity(
                            opportunity_type="schema_alignment",
                            affected_datasets=[name1, name2],
                            description=description,
                            potential_benefit="Improved data integration and analysis consistency",
                        )
                        opportunities.append(opportunity)

        return opportunities

    def detect_unusual_patterns(self, datasets: List[ProfileData]) -> List[str]:
        """
        Detect unusual patterns in the dataset collection.

        Args:
            datasets: List of ProfileData objects

        Returns:
            List of unusual pattern descriptions
        """
        unusual_patterns = []

        # Check for datasets with unusually high number of columns
        column_counts = [d.table_stats.n_columns for d in datasets]
        if column_counts:
            avg_columns = sum(column_counts) / len(column_counts)
            for dataset in datasets:
                if dataset.table_stats.n_columns > avg_columns * 3:  # 3x average
                    unusual_patterns.append(
                        f"{dataset.dataset_name} has unusually high column count ({dataset.table_stats.n_columns})"
                    )

        # Check for datasets with very high missing data
        for dataset in datasets:
            if dataset.table_stats.missing_cells_pct > 50:
                unusual_patterns.append(
                    f"{dataset.dataset_name} has very high missing data ({dataset.table_stats.missing_cells_pct:.1f}%)"
                )

        return unusual_patterns

    def _create_empty_analysis(self) -> DatasetAnalysis:
        """Create an empty DatasetAnalysis for when no datasets are provided."""
        return DatasetAnalysis(
            total_datasets=0,
            analysis_timestamp=datetime.now(),
            size_distribution=SizeDistribution(0, 0, 0, 0, 0.0, 0.0),
            quality_summary=QualitySummary(0, 0, 0, {}, 0.0),
            schema_analysis=SchemaAnalysis(
                common_data_types={},
                column_naming_analysis=ColumnNamingAnalysis({}, [], {}),
                schema_inconsistencies=[],
                schema_recommendations=[],
                data_type_distribution_by_category={},
                unique_column_names=set(),
                most_common_columns=[],
            ),
            schema_patterns=[],
            category_breakdown={},
            category_insights={},
            sample_vs_complete={},
            standardization_opportunities=[],
        )

    def _create_basic_quality_summary(
        self, datasets: List[ProfileData]
    ) -> QualitySummary:
        """Create a basic quality summary for the analyzer (detailed assessment done elsewhere)."""
        datasets_with_issues = 0
        total_issues = 0

        for dataset in datasets:
            has_issues = False
            if dataset.table_stats.missing_cells_pct > 20:
                has_issues = True
                total_issues += 1
            if dataset.table_stats.duplicate_rows_pct > 1:
                has_issues = True
                total_issues += 1
            if has_issues:
                datasets_with_issues += 1

        return QualitySummary(
            datasets_with_issues=datasets_with_issues,
            total_quality_issues=total_issues,
            high_priority_issues=total_issues,  # Simplified for now
            common_issue_types={"missing_data": total_issues},
            overall_quality_score=(
                max(0, 100 - (total_issues / len(datasets) * 20)) if datasets else 100
            ),
        )

    def _get_dataset_category(self, dataset: ProfileData) -> str:
        """Extract category from dataset file path."""
        path_parts = dataset.file_path.parts
        for i, part in enumerate(path_parts):
            if part == "reports" and i + 1 < len(path_parts):
                return path_parts[i + 1]
        return "uncategorized"

    def _find_similar_column_names(
        self, all_columns: List[str]
    ) -> List[Tuple[str, List[str]]]:
        """Find groups of similar column names that might be inconsistent."""
        unique_columns = list(set(all_columns))
        similar_groups = []
        processed = set()

        for i, col1 in enumerate(unique_columns):
            if col1 in processed:
                continue

            similar = [col1]
            for j, col2 in enumerate(unique_columns[i + 1 :], i + 1):
                if col2 in processed:
                    continue

                # Check similarity using sequence matcher
                similarity = SequenceMatcher(None, col1.lower(), col2.lower()).ratio()
                if similarity > 0.7:  # 70% similar
                    similar.append(col2)
                    processed.add(col2)

            if len(similar) > 1:
                similar_groups.append((col1, similar))
                processed.add(col1)

        return similar_groups

    def _generate_naming_suggestions(
        self, inconsistent_naming: List[Tuple[str, List[str]]]
    ) -> Dict[str, str]:
        """Generate standardization suggestions for inconsistent column names."""
        suggestions = {}

        for base_name, similar_names in inconsistent_naming:
            # Use the shortest name as the standard (often the cleanest)
            standard_name = min(similar_names, key=len)

            for name in similar_names:
                if name != standard_name:
                    suggestions[name] = standard_name

        return suggestions
