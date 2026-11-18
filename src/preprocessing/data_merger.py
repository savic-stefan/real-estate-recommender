from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config.config import AppConfig
from src.preprocessing.data_loader import DatasetProfile, RealEstateDataLoader
from src.utils.helpers import serialize_to_json
from src.utils.logger import get_logger


@dataclass(slots=True)
class MergeSummary:
    """Result metadata after schema harmonization and dataset merge."""

    dataset_count: int
    merged_rows: int
    merged_columns: list[str]
    output_path: str
    schema_report_path: str
    mapping_report_path: str
    dataset_row_counts: dict[str, int]


class SchemaHarmonizer:
    """Maps heterogeneous raw schemas to a unified real-estate attribute model."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.loader = RealEstateDataLoader(config.datasets)
        self.alias_lookup = self._build_alias_lookup()

    def execute(self) -> MergeSummary:
        """Run discovery, schema inspection, validation, harmonization, and merge."""

        dataset_paths = self.loader.discover_dataset_paths(
            self.config.paths.project_root,
            self.config.paths.raw_data_dir,
        )
        if not dataset_paths:
            raise FileNotFoundError("No CSV datasets were discovered for stage 2 processing.")

        profiles = self.loader.inspect_datasets(dataset_paths)
        mapping_records, comparison_frame, output_columns = self._prepare_schema_reports(profiles)
        schema_report_path = self.config.paths.tables_dir / self.config.datasets.schema_report_filename
        mapping_report_path = self.config.paths.tables_dir / self.config.datasets.mapping_report_filename
        comparison_frame.to_csv(schema_report_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(mapping_records).to_csv(mapping_report_path, index=False, encoding="utf-8-sig")

        merged_output_path = self.config.paths.processed_data_dir / self.config.datasets.merged_output_filename
        row_counts = self._merge_datasets(profiles, output_columns, merged_output_path)
        summary = MergeSummary(
            dataset_count=len(profiles),
            merged_rows=sum(row_counts.values()),
            merged_columns=output_columns,
            output_path=str(merged_output_path),
            schema_report_path=str(schema_report_path),
            mapping_report_path=str(mapping_report_path),
            dataset_row_counts=row_counts,
        )
        serialize_to_json(
            summary,
            self.config.paths.metadata_dir / self.config.datasets.merge_summary_filename,
        )
        self.logger.info(
            "Stage 2 completed: %s datasets merged into %s with %s total rows.",
            summary.dataset_count,
            merged_output_path.name,
            summary.merged_rows,
        )
        return summary

    def _prepare_schema_reports(
        self,
        profiles: list[DatasetProfile],
    ) -> tuple[list[dict[str, Any]], pd.DataFrame, list[str]]:
        """Build schema comparison and column mapping reports before merge."""

        mapping_records: list[dict[str, Any]] = []
        comparison_records: list[dict[str, Any]] = []
        extra_output_columns: set[str] = set()

        for profile in profiles:
            mapped_columns, missing_canonical, extra_columns = self._analyze_profile(profile)
            matched_canonical = sorted(
                {record["mapped_column"] for record in mapped_columns if record["is_canonical_match"]}
            )
            mapping_records.extend(mapped_columns)
            extra_output_columns.update(extra_columns)
            comparison_records.append(
                {
                    "dataset_name": profile.dataset_name,
                    "source_path": str(profile.path),
                    "delimiter": profile.delimiter,
                    "encoding": profile.encoding,
                    "file_size_mb": profile.file_size_mb,
                    "raw_column_count": len(profile.raw_columns),
                    "matched_canonical_count": len(matched_canonical),
                    "coverage_ratio": round(
                        len(matched_canonical) / max(len(self.config.datasets.canonical_columns), 1),
                        4,
                    ),
                    "missing_required_columns": ", ".join(
                        column
                        for column in self.config.datasets.required_columns
                        if column in missing_canonical
                    ),
                    "missing_canonical_columns": ", ".join(missing_canonical),
                    "extra_columns": ", ".join(sorted(extra_columns)),
                    "is_valid_for_merge": all(
                        required_column not in missing_canonical
                        for required_column in self.config.datasets.required_columns
                    ),
                }
            )

        comparison_frame = pd.DataFrame(comparison_records).sort_values(
            by=["is_valid_for_merge", "matched_canonical_count"],
            ascending=[False, False],
        )
        output_columns = [
            *self.config.datasets.canonical_columns,
            *sorted(extra_output_columns),
            *self.config.datasets.metadata_columns,
        ]
        return mapping_records, comparison_frame, output_columns

    def _analyze_profile(
        self,
        profile: DatasetProfile,
    ) -> tuple[list[dict[str, Any]], list[str], set[str]]:
        """Determine how each raw column maps into the unified schema."""

        mapping_records: list[dict[str, Any]] = []
        matched_canonical: set[str] = set()
        extra_columns: set[str] = set()

        for raw_column in profile.raw_columns:
            normalized = self.loader.normalize_column_name(raw_column)
            mapped_column = self.alias_lookup.get(normalized)
            if mapped_column is None:
                mapped_column = self._to_safe_extra_name(raw_column)
                is_canonical_match = False
                extra_columns.add(mapped_column)
            else:
                is_canonical_match = True
                matched_canonical.add(mapped_column)

            mapping_records.append(
                {
                    "dataset_name": profile.dataset_name,
                    "source_path": str(profile.path),
                    "raw_column": raw_column,
                    "normalized_column": normalized,
                    "mapped_column": mapped_column,
                    "is_canonical_match": is_canonical_match,
                }
            )

        missing_canonical = [
            column for column in self.config.datasets.canonical_columns
            if column not in matched_canonical
        ]
        return mapping_records, missing_canonical, extra_columns

    def _merge_datasets(
        self,
        profiles: list[DatasetProfile],
        output_columns: list[str],
        merged_output_path: Path,
    ) -> dict[str, int]:
        """Stream all datasets into a single harmonized CSV."""

        merged_output_path.parent.mkdir(parents=True, exist_ok=True)
        if merged_output_path.exists():
            merged_output_path.unlink()

        dataset_row_counts: dict[str, int] = {}
        write_header = True

        for profile in profiles:
            dataset_rows = 0
            for chunk_index, chunk in enumerate(
                self.loader.read_in_chunks(profile.path, profile.delimiter, profile.encoding),
                start=1,
            ):
                harmonized_chunk = self._harmonize_chunk(
                    dataset_name=profile.dataset_name,
                    source_path=profile.path,
                    chunk=chunk,
                    output_columns=output_columns,
                    row_offset=dataset_rows,
                )
                harmonized_chunk.to_csv(
                    merged_output_path,
                    mode="a",
                    index=False,
                    header=write_header,
                    encoding="utf-8-sig",
                )
                write_header = False
                dataset_rows += len(harmonized_chunk)
                self.logger.info(
                    "Merged chunk %s from dataset '%s' (%s rows processed).",
                    chunk_index,
                    profile.dataset_name,
                    dataset_rows,
                )

            dataset_row_counts[profile.dataset_name] = dataset_rows

        return dataset_row_counts

    def _harmonize_chunk(
        self,
        dataset_name: str,
        source_path: Path,
        chunk: pd.DataFrame,
        output_columns: list[str],
        row_offset: int,
    ) -> pd.DataFrame:
        """Rename columns, add metadata, and align each chunk to the output schema."""

        rename_map: dict[str, str] = {}
        for column in chunk.columns:
            normalized = self.loader.normalize_column_name(str(column))
            rename_map[str(column)] = self.alias_lookup.get(normalized, self._to_safe_extra_name(str(column)))

        harmonized = chunk.rename(columns=rename_map)
        harmonized = harmonized.loc[:, ~harmonized.columns.duplicated()]
        harmonized["Source dataset"] = dataset_name
        harmonized["Source file"] = source_path.name
        harmonized["Source row number"] = range(row_offset + 1, row_offset + len(harmonized) + 1)

        for column in output_columns:
            if column not in harmonized.columns:
                harmonized[column] = pd.NA

        ordered = harmonized.reindex(columns=output_columns)
        return ordered

    def _build_alias_lookup(self) -> dict[str, str]:
        """Prepare a normalized alias dictionary for schema matching."""

        alias_lookup: dict[str, str] = {}
        for canonical_column, aliases in self.config.datasets.column_aliases.items():
            alias_lookup[self.loader.normalize_column_name(canonical_column)] = canonical_column
            for alias in aliases:
                alias_lookup[self.loader.normalize_column_name(alias)] = canonical_column
        return alias_lookup

    @staticmethod
    def _to_safe_extra_name(column_name: str) -> str:
        """Convert unmapped raw columns into safe supplementary column names."""

        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(column_name).strip().lower())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return f"extra_{cleaned}" if cleaned else "extra_unnamed_column"
