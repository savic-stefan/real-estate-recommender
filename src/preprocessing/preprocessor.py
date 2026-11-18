from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config.config import AppConfig, PreprocessingConfig
from src.utils.helpers import serialize_to_json
from src.utils.logger import get_logger
from src.visualization.visualizer import RealEstateVisualizer


@dataclass(slots=True)
class PreprocessingSummary:
    """Metadata produced by the full preprocessing stage."""

    input_path: str
    clean_output_path: str
    train_output_path: str
    test_output_path: str
    original_rows: int
    cleaned_rows: int
    removed_rows: int
    modeling_rows: int
    train_rows: int
    test_rows: int
    numeric_features: list[str]
    categorical_features: list[str]
    plots_created: list[str]
    tables_created: list[str]


class RealEstatePreprocessor:
    """Cleans the merged dataset and builds a model-ready tabular dataset."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.preprocessing_config = config.preprocessing
        self.logger = get_logger(self.__class__.__name__)
        self.input_path = self.config.paths.processed_data_dir / self.preprocessing_config.input_filename
        self.clean_output_path = self.config.paths.processed_data_dir / self.preprocessing_config.clean_output_filename
        self.modeling_output_path = self.config.paths.processed_data_dir / self.preprocessing_config.modeling_sample_filename
        self.train_output_path = self.config.paths.processed_data_dir / self.preprocessing_config.train_output_filename
        self.test_output_path = self.config.paths.processed_data_dir / self.preprocessing_config.test_output_filename
        self.visualizer = RealEstateVisualizer(self.config.paths.plots_dir, self.config.eda)

    def execute(self) -> PreprocessingSummary:
        """Run the full cleaning and model-ready dataset preparation pipeline."""

        if not self.input_path.exists():
            raise FileNotFoundError(f"Merged dataset was not found at {self.input_path}.")

        stats = self._collect_global_statistics()
        clean_rows = self._clean_full_dataset(stats)
        modeling_frame, encoding_report, scaler_report = self._prepare_modeling_dataset()
        train_frame, test_frame = self._split_modeling_dataset(modeling_frame)
        plots_created = self._create_before_after_plots()
        tables_created = self._save_reports(stats, encoding_report, scaler_report, clean_rows)

        summary = PreprocessingSummary(
            input_path=str(self.input_path),
            clean_output_path=str(self.clean_output_path),
            train_output_path=str(self.train_output_path),
            test_output_path=str(self.test_output_path),
            original_rows=int(stats["total_rows"]),
            cleaned_rows=int(clean_rows),
            removed_rows=int(stats["total_rows"] - clean_rows),
            modeling_rows=int(len(modeling_frame)),
            train_rows=int(len(train_frame)),
            test_rows=int(len(test_frame)),
            numeric_features=[column for column in self.preprocessing_config.numeric_columns if column in modeling_frame.columns],
            categorical_features=[
                column for column in modeling_frame.columns
                if column.endswith("_encoded") or column.endswith("_scaled")
            ],
            plots_created=plots_created,
            tables_created=tables_created,
        )
        serialize_to_json(
            summary,
            self.config.paths.metadata_dir / self.preprocessing_config.summary_filename,
        )
        self.logger.info(
            "Preprocessing completed: %s -> %s rows, modeling dataset size=%s.",
            summary.original_rows,
            summary.cleaned_rows,
            summary.modeling_rows,
        )
        return summary

    def _collect_global_statistics(self) -> dict[str, Any]:
        """First pass over the full dataset to compute cleaning statistics."""

        selected_columns = list(self.preprocessing_config.columns_to_keep)
        total_rows = 0
        cleaned_candidate_rows = 0
        missing_before = {column: 0 for column in selected_columns}
        missing_after = {column: 0 for column in selected_columns}
        numeric_values: dict[str, list[np.ndarray]] = {
            column: []
            for column in self.preprocessing_config.numeric_columns
            if column in selected_columns
        }
        categorical_modes: dict[str, dict[str, int]] = {
            column: {}
            for column in self.preprocessing_config.categorical_columns
            if column in selected_columns
        }
        outlier_report = {
            "invalid_price": 0,
            "invalid_area": 0,
            "invalid_rooms": 0,
            "invalid_floor": 0,
            "invalid_floors": 0,
            "invalid_kitchen": 0,
            "invalid_living_area": 0,
            "floor_exceeds_building": 0,
        }

        for chunk_index, chunk in enumerate(self._iter_chunks(selected_columns), start=1):
            total_rows += len(chunk)
            prepared_chunk = self._prepare_chunk(chunk)
            for column in selected_columns:
                missing_before[column] += int(prepared_chunk[column].isna().sum())

            cleaned_chunk, chunk_outliers = self._apply_row_filters(prepared_chunk)
            cleaned_candidate_rows += len(cleaned_chunk)

            for key, value in chunk_outliers.items():
                outlier_report[key] += value

            for column in selected_columns:
                missing_after[column] += int(cleaned_chunk[column].isna().sum())

            for column in numeric_values:
                values = cleaned_chunk[column].dropna().to_numpy(dtype=float)
                if values.size > 0:
                    numeric_values[column].append(values)

            for column in categorical_modes:
                counts = cleaned_chunk[column].fillna("Unknown").astype(str).value_counts()
                for category, count in counts.items():
                    category_key = str(category)
                    categorical_modes[column][category_key] = categorical_modes[column].get(category_key, 0) + int(count)

            self.logger.info(
                "Preprocessing stats pass chunk %s: original=%s, retained=%s.",
                chunk_index,
                len(prepared_chunk),
                len(cleaned_chunk),
            )

        numeric_imputers = self._build_numeric_imputers(numeric_values)
        categorical_imputers = {
            column: max(counts.items(), key=lambda item: item[1])[0] if counts else "Unknown"
            for column, counts in categorical_modes.items()
        }

        return {
            "total_rows": total_rows,
            "cleaned_candidate_rows": cleaned_candidate_rows,
            "missing_before": missing_before,
            "missing_after": missing_after,
            "numeric_imputers": numeric_imputers,
            "categorical_imputers": categorical_imputers,
            "outlier_report": outlier_report,
        }

    def _clean_full_dataset(self, stats: dict[str, Any]) -> int:
        """Second pass: clean the full dataset and persist it to clean_data.csv."""

        if self.clean_output_path.exists():
            self.clean_output_path.unlink()

        write_header = True
        total_clean_rows = 0
        selected_columns = list(self.preprocessing_config.columns_to_keep)

        for chunk_index, chunk in enumerate(self._iter_chunks(selected_columns), start=1):
            prepared_chunk = self._prepare_chunk(chunk)
            cleaned_chunk, _ = self._apply_row_filters(prepared_chunk)
            imputed_chunk = self._apply_imputation(cleaned_chunk, stats)
            imputed_chunk.to_csv(
                self.clean_output_path,
                mode="a",
                index=False,
                header=write_header,
                encoding="utf-8-sig",
            )
            write_header = False
            total_clean_rows += len(imputed_chunk)
            self.logger.info(
                "Clean dataset write chunk %s: wrote %s rows, cumulative=%s.",
                chunk_index,
                len(imputed_chunk),
                total_clean_rows,
            )

        return total_clean_rows

    def _prepare_modeling_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Build an encoded, scaled, and normalized modeling dataset from the cleaned data."""

        clean_frame = pd.read_csv(self.clean_output_path, low_memory=False)
        modeling_frame = clean_frame.copy()

        modeling_frame = self._encode_categorical_columns(modeling_frame)
        modeling_frame = self._add_scaled_features(modeling_frame)
        modeling_frame = self._normalize_numeric_features(modeling_frame)
        modeling_frame = self._select_numeric_modeling_frame(modeling_frame)

        modeling_frame = modeling_frame.sample(
            n=min(self.preprocessing_config.modeling_sample_size, len(modeling_frame)),
            random_state=self.preprocessing_config.random_state,
        ).reset_index(drop=True)
        modeling_frame.to_csv(self.modeling_output_path, index=False, encoding="utf-8-sig")

        encoding_report = self._build_encoding_report(clean_frame, modeling_frame)
        scaler_report = self._build_scaler_report(modeling_frame)
        return modeling_frame, encoding_report, scaler_report

    def _split_modeling_dataset(self, modeling_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Perform train/test split over the cleaned and encoded modeling dataset."""

        train_frame, test_frame = train_test_split(
            modeling_frame,
            test_size=self.preprocessing_config.test_size,
            random_state=self.preprocessing_config.random_state,
            shuffle=True,
        )
        train_frame = train_frame.reset_index(drop=True)
        test_frame = test_frame.reset_index(drop=True)
        train_frame.to_csv(self.train_output_path, index=False, encoding="utf-8-sig")
        test_frame.to_csv(self.test_output_path, index=False, encoding="utf-8-sig")
        return train_frame, test_frame

    def _create_before_after_plots(self) -> list[str]:
        """Generate before/after diagnostic plots for Price."""

        raw_sample = pd.read_csv(
            self.input_path,
            usecols=["Price"],
            low_memory=False,
        ).sample(
            n=min(self.preprocessing_config.modeling_sample_size, 300_000),
            random_state=self.preprocessing_config.random_state,
        )
        clean_sample = pd.read_csv(
            self.clean_output_path,
            usecols=["Price"],
            low_memory=False,
        ).sample(
            n=min(self.preprocessing_config.modeling_sample_size, 300_000),
            random_state=self.preprocessing_config.random_state,
        )

        raw_hist = self.visualizer.plot_custom_histogram(
            raw_sample,
            value_column="Price",
            title="Price distribution before cleaning",
            output_filename=self.preprocessing_config.pre_clean_histogram_filename,
        )
        clean_hist = self.visualizer.plot_log_price_distribution(
            clean_sample,
            output_filename=self.preprocessing_config.post_clean_histogram_filename,
        )
        raw_box = self.visualizer.plot_custom_boxplot(
            raw_sample,
            value_column="Price",
            title="Price range before cleaning",
            output_filename=self.preprocessing_config.pre_clean_boxplot_filename,
        )
        clean_box = self.visualizer.plot_custom_boxplot(
            clean_sample,
            value_column="Price",
            title="Price range after cleaning",
            output_filename=self.preprocessing_config.post_clean_boxplot_filename,
        )
        return [str(path) for path in (raw_hist, clean_hist, raw_box, clean_box)]

    def _save_reports(
        self,
        stats: dict[str, Any],
        encoding_report: pd.DataFrame,
        scaler_report: pd.DataFrame,
        clean_rows: int,
    ) -> list[str]:
        """Persist preprocessing reports for the diploma appendix."""

        missing_report = pd.DataFrame(
            [
                {
                    "column": column,
                    "missing_before": stats["missing_before"][column],
                    "missing_after": stats["missing_after"][column],
                    "missing_before_ratio": round(stats["missing_before"][column] / stats["total_rows"], 6)
                    if stats["total_rows"]
                    else 0.0,
                    "missing_after_ratio": round(stats["missing_after"][column] / clean_rows, 6)
                    if clean_rows
                    else 0.0,
                }
                for column in self.preprocessing_config.columns_to_keep
            ]
        ).sort_values(by="missing_before_ratio", ascending=False)

        outlier_report = pd.DataFrame(
            [
                {"rule": key, "affected_rows": value}
                for key, value in stats["outlier_report"].items()
            ]
        ).sort_values(by="affected_rows", ascending=False)

        outputs = {
            self.preprocessing_config.missing_report_filename: missing_report,
            self.preprocessing_config.outlier_report_filename: outlier_report,
            self.preprocessing_config.encoding_report_filename: encoding_report,
            self.preprocessing_config.scaler_report_filename: scaler_report,
        }
        saved_paths: list[str] = []
        for filename, dataframe in outputs.items():
            output_path = self.config.paths.tables_dir / filename
            dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
            saved_paths.append(str(output_path))
        return saved_paths

    def _prepare_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Standardize dtypes before cleaning rules are applied."""

        frame = chunk.copy()
        for column in self.preprocessing_config.numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in self.preprocessing_config.categorical_columns:
            if column in frame.columns:
                frame[column] = frame[column].astype("string")
        return frame

    def _apply_row_filters(self, chunk: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        """Remove invalid rows and convert implausible values to missing where appropriate."""

        frame = chunk.copy()
        report = {
            "invalid_price": 0,
            "invalid_area": 0,
            "invalid_rooms": 0,
            "invalid_floor": 0,
            "invalid_floors": 0,
            "invalid_kitchen": 0,
            "invalid_living_area": 0,
            "floor_exceeds_building": 0,
        }

        invalid_price = ~frame["Price"].between(
            self.preprocessing_config.price_min_valid,
            self.preprocessing_config.price_max_valid,
            inclusive="both",
        )
        report["invalid_price"] = int(invalid_price.sum())
        frame = frame.loc[~invalid_price].copy()

        invalid_area = ~frame["Area"].between(
            self.preprocessing_config.area_min_valid,
            self.preprocessing_config.area_max_valid,
            inclusive="both",
        )
        report["invalid_area"] = int(invalid_area.sum())
        frame = frame.loc[~invalid_area].copy()

        invalid_rooms = ~frame["Number of rooms"].between(
            self.preprocessing_config.rooms_min_valid,
            self.preprocessing_config.rooms_max_valid,
            inclusive="both",
        )
        report["invalid_rooms"] = int(invalid_rooms.sum())
        frame = frame.loc[~invalid_rooms].copy()

        invalid_floor = ~frame["Floor"].between(
            self.preprocessing_config.floor_min_valid,
            self.preprocessing_config.floor_max_valid,
            inclusive="both",
        )
        report["invalid_floor"] = int(invalid_floor.sum())
        frame = frame.loc[~invalid_floor].copy()

        invalid_num_floors = ~frame["Number of floors"].between(
            self.preprocessing_config.num_floors_min_valid,
            self.preprocessing_config.num_floors_max_valid,
            inclusive="both",
        )
        report["invalid_floors"] = int(invalid_num_floors.sum())
        frame = frame.loc[~invalid_num_floors].copy()

        invalid_floor_relation = frame["Floor"] > frame["Number of floors"]
        report["floor_exceeds_building"] = int(invalid_floor_relation.sum())
        frame = frame.loc[~invalid_floor_relation].copy()

        if "Kitchen area" in frame.columns:
            invalid_kitchen = ~frame["Kitchen area"].between(
                self.preprocessing_config.kitchen_area_min_valid,
                self.preprocessing_config.kitchen_area_max_valid,
                inclusive="both",
            )
            report["invalid_kitchen"] = int(invalid_kitchen.sum())
            frame.loc[invalid_kitchen, "Kitchen area"] = np.nan

        if "Living area" in frame.columns:
            invalid_living = ~frame["Living area"].between(
                self.preprocessing_config.living_area_min_valid,
                self.preprocessing_config.living_area_max_valid,
                inclusive="both",
            )
            invalid_living = invalid_living & frame["Living area"].notna()
            report["invalid_living_area"] = int(invalid_living.sum())
            frame.loc[invalid_living, "Living area"] = np.nan

        if "Minutes to metro" in frame.columns:
            invalid_metro = ~frame["Minutes to metro"].between(
                self.preprocessing_config.minutes_to_metro_min_valid,
                self.preprocessing_config.minutes_to_metro_max_valid,
                inclusive="both",
            )
            frame.loc[invalid_metro, "Minutes to metro"] = np.nan

        return frame, report

    def _apply_imputation(self, chunk: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
        """Fill missing values using robust global imputers."""

        frame = chunk.copy()
        for column, median_value in stats["numeric_imputers"].items():
            if column in frame.columns:
                frame[column] = frame[column].fillna(median_value)
        for column, mode_value in stats["categorical_imputers"].items():
            if column in frame.columns:
                frame[column] = frame[column].fillna(mode_value)
        return frame

    def _build_numeric_imputers(self, numeric_values: dict[str, list[np.ndarray]]) -> dict[str, float]:
        """Compute median imputers over cleaned numeric distributions."""

        imputers: dict[str, float] = {}
        for column, arrays in numeric_values.items():
            if arrays:
                concatenated = np.concatenate(arrays)
                imputers[column] = float(np.nanmedian(concatenated))
            else:
                imputers[column] = 0.0
        return imputers

    def _encode_categorical_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply frequency encoding to categorical columns for scalable modeling."""

        encoded_frame = frame.copy()
        encoder_map: dict[str, dict[str, float]] = {}
        for column in self.preprocessing_config.categorical_columns:
            if column not in encoded_frame.columns:
                continue
            frequencies = encoded_frame[column].astype(str).value_counts(normalize=True)
            encoder_map[column] = frequencies.to_dict()
            encoded_frame[f"{column}_encoded"] = encoded_frame[column].astype(str).map(encoder_map[column]).fillna(0.0)

        joblib.dump(encoder_map, self.config.paths.models_dir / "categorical_frequency_encoders.joblib")
        return encoded_frame

    def _add_scaled_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit StandardScaler on the modeling frame and append standardized features."""

        scaled_frame = frame.copy()
        numeric_columns = [column for column in self.preprocessing_config.numeric_columns if column in scaled_frame.columns]
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(scaled_frame[numeric_columns])
        scaled_columns = [f"{column}_scaled" for column in numeric_columns]
        scaled_frame[scaled_columns] = scaled_values
        joblib.dump(
            {"columns": numeric_columns, "scaler": scaler},
            self.config.paths.models_dir / "standard_scaler.joblib",
        )
        return scaled_frame

    def _normalize_numeric_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit MinMaxScaler and append normalized features."""

        normalized_frame = frame.copy()
        numeric_columns = [column for column in self.preprocessing_config.numeric_columns if column in normalized_frame.columns]
        scaler = MinMaxScaler()
        normalized_values = scaler.fit_transform(normalized_frame[numeric_columns])
        normalized_columns = [f"{column}_normalized" for column in numeric_columns]
        normalized_frame[normalized_columns] = normalized_values
        joblib.dump(
            {"columns": numeric_columns, "scaler": scaler},
            self.config.paths.models_dir / "minmax_scaler.joblib",
        )
        return normalized_frame

    def _build_encoding_report(self, clean_frame: pd.DataFrame, modeling_frame: pd.DataFrame) -> pd.DataFrame:
        """Describe categorical encoding coverage."""

        records: list[dict[str, Any]] = []
        for column in self.preprocessing_config.categorical_columns:
            if column not in clean_frame.columns:
                continue
            encoded_column = f"{column}_encoded"
            records.append(
                {
                    "column": column,
                    "unique_values_in_clean_data": int(clean_frame[column].astype(str).nunique()),
                    "encoded_column": encoded_column,
                    "encoded_min": float(modeling_frame[encoded_column].min()),
                    "encoded_max": float(modeling_frame[encoded_column].max()),
                    "top_category": clean_frame[column].astype(str).mode().iloc[0],
                }
            )
        return pd.DataFrame(records)

    def _select_numeric_modeling_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep only target and numeric model-ready features in the modeling dataset."""

        base_numeric = [column for column in self.preprocessing_config.numeric_columns if column in frame.columns]
        encoded_columns = [f"{column}_encoded" for column in self.preprocessing_config.categorical_columns if f"{column}_encoded" in frame.columns]
        scaled_columns = [f"{column}_scaled" for column in self.preprocessing_config.numeric_columns if f"{column}_scaled" in frame.columns]
        normalized_columns = [f"{column}_normalized" for column in self.preprocessing_config.numeric_columns if f"{column}_normalized" in frame.columns]
        selected_columns = list(dict.fromkeys([*base_numeric, *encoded_columns, *scaled_columns, *normalized_columns]))
        return frame.loc[:, selected_columns].copy()

    def _build_scaler_report(self, modeling_frame: pd.DataFrame) -> pd.DataFrame:
        """Summarize the ranges of scaled and normalized features."""

        records: list[dict[str, Any]] = []
        for column in self.preprocessing_config.numeric_columns:
            scaled_column = f"{column}_scaled"
            normalized_column = f"{column}_normalized"
            if scaled_column not in modeling_frame.columns or normalized_column not in modeling_frame.columns:
                continue
            records.append(
                {
                    "feature": column,
                    "scaled_mean": round(float(modeling_frame[scaled_column].mean()), 6),
                    "scaled_std": round(float(modeling_frame[scaled_column].std()), 6),
                    "normalized_min": round(float(modeling_frame[normalized_column].min()), 6),
                    "normalized_max": round(float(modeling_frame[normalized_column].max()), 6),
                }
            )
        return pd.DataFrame(records)

    def _iter_chunks(self, selected_columns: list[str]):
        return pd.read_csv(
            self.input_path,
            usecols=lambda column: column in selected_columns,
            chunksize=self.preprocessing_config.chunk_size,
            low_memory=False,
        )
