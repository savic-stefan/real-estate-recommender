from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config.config import DatasetConfig
from src.utils.helpers import discover_csv_files
from src.utils.logger import get_logger


@dataclass(slots=True)
class DatasetProfile:
    """Structural metadata captured for a raw CSV dataset before merge."""

    dataset_name: str
    path: Path
    delimiter: str
    encoding: str
    raw_columns: list[str]
    normalized_columns: list[str]
    file_size_mb: float
    sample_rows: list[dict[str, Any]]


class RealEstateDataLoader:
    """Discovers, inspects, and streams CSV datasets for schema harmonization."""

    def __init__(self, dataset_config: DatasetConfig) -> None:
        self.config = dataset_config
        self.logger = get_logger(self.__class__.__name__)

    def discover_dataset_paths(self, project_root: Path, raw_data_dir: Path) -> list[Path]:
        """Discover dataset files in configured locations and sort them deterministically."""

        discovered = discover_csv_files((raw_data_dir, project_root), self.config.source_patterns)
        filtered = [
            path for path in discovered
            if path.suffix.lower() in self.config.accepted_extensions
        ]
        preferred_order = {name: idx for idx, name in enumerate(self.config.preferred_filenames)}
        filtered.sort(key=lambda path: (preferred_order.get(path.name, 10_000), path.name.lower()))
        self.logger.info("Discovered %s CSV dataset(s) for ingestion.", len(filtered))
        return filtered

    def inspect_datasets(self, dataset_paths: list[Path]) -> list[DatasetProfile]:
        """Build schema profiles for all discovered datasets."""

        profiles = [self.inspect_dataset(path) for path in dataset_paths]
        self.logger.info("Prepared structural profiles for %s dataset(s).", len(profiles))
        return profiles

    def inspect_dataset(self, dataset_path: Path) -> DatasetProfile:
        """Inspect delimiter, encoding, columns, and sample rows for a single file."""

        encoding = self._detect_encoding(dataset_path)
        delimiter = self._detect_delimiter(dataset_path, encoding)
        sample_frame = pd.read_csv(
            dataset_path,
            sep=delimiter,
            encoding=encoding,
            nrows=self.config.profile_sample_rows,
            on_bad_lines="skip",
        )
        raw_columns = [str(column) for column in sample_frame.columns]
        normalized_columns = [self.normalize_column_name(column) for column in raw_columns]
        profile = DatasetProfile(
            dataset_name=dataset_path.stem,
            path=dataset_path.resolve(),
            delimiter=delimiter,
            encoding=encoding,
            raw_columns=raw_columns,
            normalized_columns=normalized_columns,
            file_size_mb=round(dataset_path.stat().st_size / (1024 * 1024), 3),
            sample_rows=sample_frame.to_dict(orient="records"),
        )
        self.logger.info(
            "Inspected dataset '%s': %s columns, delimiter='%s', encoding=%s.",
            profile.dataset_name,
            len(profile.raw_columns),
            profile.delimiter,
            profile.encoding,
        )
        return profile

    def read_in_chunks(self, dataset_path: Path, delimiter: str, encoding: str) -> pd.io.parsers.TextFileReader:
        """Yield dataset chunks to support merging large CSV files without full in-memory load."""

        return pd.read_csv(
            dataset_path,
            sep=delimiter,
            encoding=encoding,
            chunksize=self.config.merge_chunk_size,
            low_memory=False,
            on_bad_lines="skip",
        )

    def _detect_encoding(self, dataset_path: Path) -> str:
        """Pick the first working text encoding from the configured candidates."""

        for encoding in self.config.encoding_candidates:
            try:
                with dataset_path.open("r", encoding=encoding, errors="strict") as file_obj:
                    file_obj.read(self.config.sniff_bytes)
                return encoding
            except UnicodeDecodeError:
                continue
        self.logger.warning(
            "Failed to confidently detect encoding for %s. Falling back to utf-8.",
            dataset_path.name,
        )
        return "utf-8"

    def _detect_delimiter(self, dataset_path: Path, encoding: str) -> str:
        """Infer the delimiter using the standard CSV sniffer with a safe fallback."""

        with dataset_path.open("r", encoding=encoding, errors="replace", newline="") as file_obj:
            sample = file_obj.read(self.config.sniff_bytes)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return str(dialect.delimiter)
        except csv.Error:
            self.logger.warning(
                "Delimiter sniffing failed for %s. Falling back to comma.",
                dataset_path.name,
            )
            return ","

    @staticmethod
    def normalize_column_name(column_name: str) -> str:
        """Normalize column names to compare semantically similar attributes."""

        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(column_name).strip().lower())
        return re.sub(r"_+", "_", normalized).strip("_")
