from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.config import AppConfig
from src.utils.helpers import serialize_to_json
from src.utils.logger import get_logger
from src.visualization.visualizer import RealEstateVisualizer


@dataclass(slots=True)
class EDASummary:
    """Metadata and high-level statistics produced by the EDA stage."""

    input_path: str
    total_rows: int
    total_columns: int
    analysis_sample_size: int
    missing_cells: int
    duplicate_rows_in_sample: int
    plots_created: list[str]
    tables_created: list[str]


class RealEstateEDAAnalyzer:
    """Performs reproducible EDA over the merged real-estate dataset."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.input_path = self.config.paths.processed_data_dir / self.config.eda.input_filename
        self.visualizer = RealEstateVisualizer(self.config.paths.plots_dir, self.config.eda)

    def execute(self) -> EDASummary:
        """Run structural analysis, descriptive statistics, and visualization."""

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Merged dataset was not found at {self.input_path}. Run stage 2 first."
            )

        sample_frame, stream_stats = self._stream_dataset_statistics()
        overview_frame = self._build_overview_table(
            sample_frame=sample_frame,
            total_rows=stream_stats["total_rows"],
            dtypes=stream_stats["dtypes"],
            missing_counts=stream_stats["missing_counts"],
        )
        missing_frame = self._build_missing_table(
            total_rows=stream_stats["total_rows"],
            missing_counts=stream_stats["missing_counts"],
        )
        numeric_summary = self._build_numeric_summary(sample_frame)
        categorical_summary = self._build_categorical_summary(sample_frame)
        correlation_matrix = self._build_correlation_matrix(sample_frame)
        source_price_summary = self._build_source_price_summary(
            source_accumulator=stream_stats["source_accumulator"],
            sample_frame=sample_frame,
        )
        region_distribution = self._build_region_distribution(stream_stats["region_counts"])
        metro_distribution = self._build_metro_distribution(stream_stats["metro_counts"])
        duplicate_frame = self._build_duplicate_summary(sample_frame)

        tables_created = self._save_tables(
            overview_frame=overview_frame,
            missing_frame=missing_frame,
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
            correlation_matrix=correlation_matrix,
            source_price_summary=source_price_summary,
            region_distribution=region_distribution,
            metro_distribution=metro_distribution,
            duplicate_frame=duplicate_frame,
        )

        metro_plot_frame = self._prepare_metro_plot_frame(sample_frame, metro_distribution)
        plots_created = self._create_plots(
            sample_frame=sample_frame,
            correlation_matrix=correlation_matrix,
            source_price_summary=source_price_summary,
            region_distribution=region_distribution,
            metro_plot_frame=metro_plot_frame,
        )

        summary = EDASummary(
            input_path=str(self.input_path),
            total_rows=int(stream_stats["total_rows"]),
            total_columns=int(len(stream_stats["dtypes"])),
            analysis_sample_size=int(len(sample_frame)),
            missing_cells=int(sum(stream_stats["missing_counts"].values())),
            duplicate_rows_in_sample=int(duplicate_frame["duplicate_rows_in_sample"].iloc[0]),
            plots_created=plots_created,
            tables_created=tables_created,
        )
        serialize_to_json(
            summary,
            self.config.paths.metadata_dir / self.config.eda.report_metadata_filename,
        )
        self.logger.info(
            "EDA stage completed for %s rows and %s columns.",
            summary.total_rows,
            summary.total_columns,
        )
        return summary

    def _stream_dataset_statistics(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Scan the dataset once to collect global stats and an analysis sample."""

        selected_columns = self._selected_columns()
        sample_parts: list[pd.DataFrame] = []
        missing_counts: dict[str, int] = {column: 0 for column in selected_columns}
        dtypes: dict[str, str] = {}
        source_accumulator: dict[str, dict[str, float | int | None]] = {}
        region_counts: dict[str, int] = {}
        metro_counts: dict[str, int] = {}
        total_rows = 0

        total_rows_hint = self._read_total_rows_hint()
        sample_fraction = min(1.0, self.config.eda.sample_size / max(total_rows_hint, 1))

        for chunk_index, chunk in enumerate(self._iter_chunks(selected_columns), start=1):
            if not dtypes:
                dtypes = {column: str(dtype) for column, dtype in chunk.dtypes.items()}

            chunk_rows = len(chunk)
            total_rows += chunk_rows

            for column in selected_columns:
                if column in chunk.columns:
                    missing_counts[column] += int(chunk[column].isna().sum())

            self._update_source_accumulator(source_accumulator, chunk)
            self._update_counter(region_counts, chunk.get("Region"), fill_value="Unknown region")
            self._update_counter(metro_counts, chunk.get("Metro station"), fill_value="Unknown metro")

            sampled_chunk = chunk.sample(frac=sample_fraction, random_state=self.config.eda.random_state + chunk_index)
            if not sampled_chunk.empty:
                sample_parts.append(sampled_chunk)

            self.logger.info(
                "EDA stream processed chunk %s with %s rows; cumulative rows=%s.",
                chunk_index,
                chunk_rows,
                total_rows,
            )

        if not sample_parts:
            raise ValueError("EDA sampling produced an empty frame. Check the merged dataset integrity.")

        sample_frame = pd.concat(sample_parts, ignore_index=True)
        if len(sample_frame) > self.config.eda.sample_size:
            sample_frame = sample_frame.sample(
                n=self.config.eda.sample_size,
                random_state=self.config.eda.random_state,
            ).reset_index(drop=True)

        self.logger.info(
            "Constructed EDA sample with %s rows from %s total rows.",
            len(sample_frame),
            total_rows,
        )
        return sample_frame, {
            "total_rows": total_rows,
            "missing_counts": missing_counts,
            "dtypes": dtypes,
            "source_accumulator": source_accumulator,
            "region_counts": region_counts,
            "metro_counts": metro_counts,
        }

    def _build_overview_table(
        self,
        sample_frame: pd.DataFrame,
        total_rows: int,
        dtypes: dict[str, str],
        missing_counts: dict[str, int],
    ) -> pd.DataFrame:
        """Summarize dataset structure, dtypes, and completeness."""

        records: list[dict[str, Any]] = []
        for column in sample_frame.columns:
            series = sample_frame[column]
            missing_count = int(missing_counts.get(column, 0))
            non_null_count = total_rows - missing_count
            sample_unique_count = int(series.nunique(dropna=True))
            missing_ratio = round(missing_count / total_rows, 6) if total_rows else 0.0
            records.append(
                {
                    "column": column,
                    "dtype": dtypes.get(column, str(series.dtype)),
                    "non_null_count": non_null_count,
                    "missing_count": missing_count,
                    "missing_ratio": missing_ratio,
                    "sample_unique_count": sample_unique_count,
                }
            )
        return pd.DataFrame(records).sort_values(by=["missing_ratio", "sample_unique_count"], ascending=[False, False])

    def _build_missing_table(
        self,
        total_rows: int,
        missing_counts: dict[str, int],
    ) -> pd.DataFrame:
        """Build a ranked missing-value report."""

        records = [
            {
                "column": column,
                "missing_count": int(missing_count),
                "missing_ratio": round(missing_count / total_rows, 6) if total_rows else 0.0,
            }
            for column, missing_count in missing_counts.items()
        ]
        return pd.DataFrame(records).sort_values(by="missing_ratio", ascending=False)

    def _build_numeric_summary(self, sample_frame: pd.DataFrame) -> pd.DataFrame:
        """Describe the numeric feature space."""

        numeric_frame = sample_frame.loc[
            :,
            [column for column in self.config.eda.numeric_columns if column in sample_frame.columns],
        ]
        numeric_frame = numeric_frame.apply(pd.to_numeric, errors="coerce")
        summary = numeric_frame.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).transpose()
        summary["missing_count"] = numeric_frame.isna().sum()
        summary["missing_ratio"] = numeric_frame.isna().mean().round(6)
        summary.reset_index(names="feature", inplace=True)
        return summary

    def _build_categorical_summary(self, sample_frame: pd.DataFrame) -> pd.DataFrame:
        """Summarize the categorical feature space."""

        records: list[dict[str, Any]] = []
        for column in self.config.eda.categorical_columns:
            if column not in sample_frame.columns:
                continue
            mode_series = sample_frame[column].mode(dropna=True)
            records.append(
                {
                    "feature": column,
                    "unique_values": int(sample_frame[column].nunique(dropna=True)),
                    "top_value": mode_series.iloc[0] if not mode_series.empty else pd.NA,
                    "top_value_frequency": int(sample_frame[column].value_counts(dropna=True).iloc[0])
                    if sample_frame[column].dropna().shape[0] > 0
                    else 0,
                    "missing_count": int(sample_frame[column].isna().sum()),
                    "missing_ratio": round(float(sample_frame[column].isna().mean()), 6),
                }
            )
        return pd.DataFrame(records).sort_values(by="unique_values", ascending=False)

    def _build_correlation_matrix(self, sample_frame: pd.DataFrame) -> pd.DataFrame:
        """Compute Pearson correlations for key numeric attributes."""

        numeric_frame = sample_frame.loc[
            :,
            [column for column in self.config.eda.numeric_columns if column in sample_frame.columns],
        ]
        numeric_frame = numeric_frame.apply(pd.to_numeric, errors="coerce")
        correlation_matrix = numeric_frame.corr(numeric_only=True).round(4)
        correlation_matrix.index.name = "feature"
        return correlation_matrix

    def _build_source_price_summary(
        self,
        source_accumulator: dict[str, dict[str, float | int | None]],
        sample_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compare sources by price level and record count."""

        sample_price = sample_frame.loc[:, ["Source dataset", "Price"]].copy()
        sample_price["Price"] = pd.to_numeric(sample_price["Price"], errors="coerce")
        sample_medians = (
            sample_price.groupby("Source dataset", dropna=False)["Price"]
            .median()
            .round(2)
            .rename("sample_median_price")
            .reset_index()
        )

        records: list[dict[str, Any]] = []
        for source_name, stats in source_accumulator.items():
            price_count = int(stats["price_count"])
            price_sum = float(stats["price_sum"])
            records.append(
                {
                    "Source dataset": source_name,
                    "listing_count": int(stats["listing_count"]),
                    "non_null_price_count": price_count,
                    "mean_price": round(price_sum / price_count, 2) if price_count else np.nan,
                    "min_price": stats["min_price"],
                    "max_price": stats["max_price"],
                }
            )

        summary = pd.DataFrame(records)
        summary = summary.merge(sample_medians, on="Source dataset", how="left")
        return summary.sort_values(by="mean_price", ascending=False)

    def _build_region_distribution(self, region_counts: dict[str, int]) -> pd.DataFrame:
        """Rank regions by listing count."""

        ordered = sorted(region_counts.items(), key=lambda item: item[1], reverse=True)
        return pd.DataFrame(ordered[: self.config.eda.top_region_count], columns=["Region", "listing_count"])

    def _build_metro_distribution(self, metro_counts: dict[str, int]) -> pd.DataFrame:
        """Rank metro stations by sample frequency."""

        ordered = sorted(metro_counts.items(), key=lambda item: item[1], reverse=True)
        return pd.DataFrame(ordered[: self.config.eda.top_metro_count], columns=["Metro station", "listing_count"])

    def _build_duplicate_summary(self, sample_frame: pd.DataFrame) -> pd.DataFrame:
        """Estimate duplicate prevalence using business-key duplicates on the analysis frame."""

        subset_columns = [column for column in self.config.eda.duplicate_subset_columns if column in sample_frame.columns]
        duplicate_mask = sample_frame.duplicated(subset=subset_columns, keep=False)
        duplicate_rows = int(duplicate_mask.sum())
        duplicate_ratio = round(duplicate_rows / len(sample_frame), 6) if len(sample_frame) else 0.0
        return pd.DataFrame(
            [
                {
                    "analysis_scope": "analysis_sample",
                    "subset_columns": ", ".join(subset_columns),
                    "duplicate_rows_in_sample": duplicate_rows,
                    "duplicate_ratio_in_sample": duplicate_ratio,
                }
            ]
        )

    def _prepare_metro_plot_frame(
        self,
        sample_frame: pd.DataFrame,
        metro_distribution: pd.DataFrame,
    ) -> pd.DataFrame:
        """Restrict metro plot to the most representative stations."""

        if metro_distribution.empty:
            return pd.DataFrame(columns=["Metro station", "Price"])
        metro_names = metro_distribution["Metro station"].tolist()
        metro_frame = sample_frame.loc[
            sample_frame["Metro station"].isin(metro_names),
            ["Metro station", "Price"],
        ].copy()
        metro_frame["Price"] = pd.to_numeric(metro_frame["Price"], errors="coerce")
        metro_frame = metro_frame.dropna(subset=["Price"])
        return metro_frame

    def _create_plots(
        self,
        sample_frame: pd.DataFrame,
        correlation_matrix: pd.DataFrame,
        source_price_summary: pd.DataFrame,
        region_distribution: pd.DataFrame,
        metro_plot_frame: pd.DataFrame,
    ) -> list[str]:
        """Create all required EDA visualizations."""

        plots = [
            self.visualizer.plot_correlation_heatmap(correlation_matrix),
            self.visualizer.plot_price_histogram(sample_frame),
            self.visualizer.plot_price_boxplot(sample_frame),
            self.visualizer.plot_price_vs_area(sample_frame),
            self.visualizer.plot_price_vs_rooms(sample_frame),
            self.visualizer.plot_average_price_by_source(source_price_summary),
            self.visualizer.plot_region_distribution(region_distribution),
        ]
        if not metro_plot_frame.empty:
            plots.append(self.visualizer.plot_price_vs_metro(metro_plot_frame))
        return [str(path) for path in plots]

    def _save_tables(
        self,
        overview_frame: pd.DataFrame,
        missing_frame: pd.DataFrame,
        numeric_summary: pd.DataFrame,
        categorical_summary: pd.DataFrame,
        correlation_matrix: pd.DataFrame,
        source_price_summary: pd.DataFrame,
        region_distribution: pd.DataFrame,
        metro_distribution: pd.DataFrame,
        duplicate_frame: pd.DataFrame,
    ) -> list[str]:
        """Persist analytical tables for the diploma report."""

        outputs = {
            self.config.eda.overview_table_filename: overview_frame,
            self.config.eda.missing_table_filename: missing_frame,
            self.config.eda.numeric_summary_filename: numeric_summary,
            self.config.eda.categorical_summary_filename: categorical_summary,
            self.config.eda.correlation_table_filename: correlation_matrix.reset_index(),
            self.config.eda.source_price_table_filename: source_price_summary,
            self.config.eda.region_distribution_table_filename: region_distribution,
            self.config.eda.metro_distribution_table_filename: metro_distribution,
            self.config.eda.duplicates_table_filename: duplicate_frame,
        }
        saved_paths: list[str] = []
        for filename, dataframe in outputs.items():
            output_path = self.config.paths.tables_dir / filename
            dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
            saved_paths.append(str(output_path))
            self.logger.info("Saved table: %s", output_path.name)
        return saved_paths

    def _selected_columns(self) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    *self.config.eda.numeric_columns,
                    *self.config.eda.categorical_columns,
                    *self.config.eda.duplicate_subset_columns,
                ]
            )
        )

    def _iter_chunks(self, selected_columns: list[str]):
        return pd.read_csv(
            self.input_path,
            usecols=lambda column: column in selected_columns,
            chunksize=self.config.eda.chunk_size,
            low_memory=False,
        )

    def _read_total_rows_hint(self) -> int:
        metadata_path = self.config.paths.metadata_dir / self.config.datasets.merge_summary_filename
        if metadata_path.exists():
            metadata = pd.read_json(metadata_path, typ="series")
            total_rows = int(metadata.get("merged_rows", 0))
            if total_rows > 0:
                return total_rows
        self.logger.warning("Merge summary metadata is unavailable; falling back to sample size hint.")
        return self.config.eda.sample_size

    @staticmethod
    def _update_counter(counter: dict[str, int], series: pd.Series | None, fill_value: str) -> None:
        if series is None:
            return
        counts = series.fillna(fill_value).value_counts()
        for key, value in counts.items():
            counter[str(key)] = counter.get(str(key), 0) + int(value)

    @staticmethod
    def _update_source_accumulator(
        accumulator: dict[str, dict[str, float | int | None]],
        chunk: pd.DataFrame,
    ) -> None:
        source_price = chunk.loc[:, ["Source dataset", "Price"]].copy()
        source_price["Price"] = pd.to_numeric(source_price["Price"], errors="coerce")

        grouped = source_price.groupby("Source dataset", dropna=False)
        for source_name, group in grouped:
            source_key = str(source_name)
            if source_key not in accumulator:
                accumulator[source_key] = {
                    "listing_count": 0,
                    "price_count": 0,
                    "price_sum": 0.0,
                    "min_price": None,
                    "max_price": None,
                }

            state = accumulator[source_key]
            state["listing_count"] = int(state["listing_count"]) + int(len(group))

            price_values = group["Price"].dropna()
            if not price_values.empty:
                state["price_count"] = int(state["price_count"]) + int(price_values.count())
                state["price_sum"] = float(state["price_sum"]) + float(price_values.sum())
                current_min = float(price_values.min())
                current_max = float(price_values.max())
                state["min_price"] = (
                    current_min
                    if state["min_price"] is None
                    else min(float(state["min_price"]), current_min)
                )
                state["max_price"] = (
                    current_max
                    if state["max_price"] is None
                    else max(float(state["max_price"]), current_max)
                )
