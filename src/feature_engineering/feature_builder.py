from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config.config import AppConfig, FeatureEngineeringConfig
from src.utils.helpers import serialize_to_json
from src.utils.logger import get_logger
from src.visualization.plot_labels import (
    format_numeric_axis,
    setup_plot_style,
    boolean_label,
    feature_label,
    feature_set_label,
    wrap_ticklabels,
)


@dataclass(slots=True)
class FeatureEngineeringSummary:
    """Metadata produced by the feature engineering stage."""

    input_path: str
    output_path: str
    engineered_rows: int
    comparison_sample_rows: int
    district_profile_rows: int
    baseline_feature_count: int
    enhanced_feature_count: int
    leakage_excluded_features: list[str]
    tables_created: list[str]
    plots_created: list[str]


class RealEstateFeatureBuilder:
    """Builds market-aware housing features and measures their impact on model quality."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.feature_config: FeatureEngineeringConfig = config.feature_engineering
        self.logger = get_logger(self.__class__.__name__)
        self.input_path = self.config.paths.processed_data_dir / self.feature_config.input_filename
        self.output_path = self.config.paths.processed_data_dir / self.feature_config.output_filename

    def execute(self) -> FeatureEngineeringSummary:
        """Run domain feature construction and compare baseline vs enriched model quality."""

        if not self.input_path.exists():
            raise FileNotFoundError(f"Clean dataset was not found at {self.input_path}. Run stage 4 first.")

        total_rows = self._load_clean_row_count()
        district_profile = self._collect_district_profiles()
        engineered_sample, engineered_rows = self._build_engineered_dataset(
            district_profile=district_profile,
            total_rows=total_rows,
        )

        feature_summary = self._build_engineered_feature_summary(engineered_sample)
        impact_frame, importance_frame = self._compare_model_quality(engineered_sample)

        district_profile_path = self.config.paths.tables_dir / self.feature_config.district_profile_filename
        feature_summary_path = self.config.paths.tables_dir / self.feature_config.feature_summary_filename
        impact_table_path = self.config.paths.tables_dir / self.feature_config.model_impact_filename
        importance_table_path = self.config.paths.tables_dir / self.feature_config.importance_filename

        district_profile.to_csv(district_profile_path, index=False, encoding="utf-8-sig")
        feature_summary.to_csv(feature_summary_path, index=False, encoding="utf-8-sig")
        impact_frame.to_csv(impact_table_path, index=False, encoding="utf-8-sig")
        importance_frame.to_csv(importance_table_path, index=False, encoding="utf-8-sig")

        comparison_plot_path = self._plot_model_impact(impact_frame)
        importance_plot_path = self._plot_feature_importance(importance_frame)

        summary = FeatureEngineeringSummary(
            input_path=str(self.input_path),
            output_path=str(self.output_path),
            engineered_rows=engineered_rows,
            comparison_sample_rows=len(engineered_sample),
            district_profile_rows=len(district_profile),
            baseline_feature_count=int(impact_frame.loc[impact_frame["feature_set"] == "baseline", "feature_count"].iloc[0]),
            enhanced_feature_count=int(impact_frame.loc[impact_frame["feature_set"] == "enhanced", "feature_count"].iloc[0]),
            leakage_excluded_features=list(self.feature_config.leakage_sensitive_engineered_columns),
            tables_created=[
                str(district_profile_path),
                str(feature_summary_path),
                str(impact_table_path),
                str(importance_table_path),
            ],
            plots_created=[
                str(comparison_plot_path),
                str(importance_plot_path),
            ],
        )
        serialize_to_json(
            summary,
            self.config.paths.metadata_dir / self.feature_config.summary_filename,
        )
        self.logger.info(
            "Stage 7 completed: engineered %s rows, comparison sample size=%s.",
            summary.engineered_rows,
            summary.comparison_sample_rows,
        )
        return summary

    def _load_clean_row_count(self) -> int:
        summary_path = self.config.paths.metadata_dir / self.config.preprocessing.summary_filename
        with summary_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        return int(payload["cleaned_rows"])

    def _collect_district_profiles(self) -> pd.DataFrame:
        """Aggregate district-level demand and value signals from the full clean dataset."""

        accumulator: dict[int, dict[str, float]] = {}
        for chunk in self._iter_clean_chunks(usecols=["Price", "Area", "Region id"]):
            frame = chunk.copy()
            frame["Price"] = pd.to_numeric(frame["Price"], errors="coerce")
            frame["Area"] = pd.to_numeric(frame["Area"], errors="coerce")
            frame["Region id"] = pd.to_numeric(frame["Region id"], errors="coerce").round().astype("Int64")
            frame["price_per_m2"] = frame["Price"] / frame["Area"].clip(lower=1.0)
            frame = frame.dropna(subset=["Region id", "price_per_m2"])

            grouped = frame.groupby("Region id", dropna=True)
            for district_id, group in grouped:
                district_key = int(district_id)
                if district_key not in accumulator:
                    accumulator[district_key] = {
                        "listing_count": 0.0,
                        "price_per_m2_sum": 0.0,
                    }
                accumulator[district_key]["listing_count"] += float(len(group))
                accumulator[district_key]["price_per_m2_sum"] += float(group["price_per_m2"].sum())

        profile = pd.DataFrame(
            [
                {
                    "district_id": district_id,
                    "listing_count": int(values["listing_count"]),
                    "mean_price_per_m2": values["price_per_m2_sum"] / max(values["listing_count"], 1.0),
                }
                for district_id, values in accumulator.items()
            ]
        ).sort_values(by="listing_count", ascending=False)

        profile["listing_count_norm"] = self._minmax_scale(profile["listing_count"])
        profile["mean_price_per_m2_norm"] = self._minmax_scale(profile["mean_price_per_m2"])
        profile["district_popularity_score"] = (
            0.65 * profile["listing_count_norm"] + 0.35 * profile["mean_price_per_m2_norm"]
        ).round(6)
        return profile.reset_index(drop=True)

    def _build_engineered_dataset(
        self,
        district_profile: pd.DataFrame,
        total_rows: int,
    ) -> tuple[pd.DataFrame, int]:
        """Create engineered features over the full clean dataset and collect a modeling sample."""

        if self.output_path.exists():
            self.output_path.unlink()

        popularity_lookup = district_profile.set_index("district_id")["district_popularity_score"].to_dict()
        write_header = True
        sampled_chunks: list[pd.DataFrame] = []
        engineered_rows = 0
        sample_fraction = min(1.0, self.feature_config.comparison_sample_size / max(total_rows, 1))

        for chunk_index, chunk in enumerate(self._iter_clean_chunks(), start=1):
            enriched_chunk = self._enrich_chunk(chunk, popularity_lookup)
            enriched_chunk.to_csv(
                self.output_path,
                mode="a",
                index=False,
                header=write_header,
                encoding="utf-8-sig",
            )
            write_header = False
            engineered_rows += len(enriched_chunk)

            sampled_chunk = enriched_chunk.sample(
                frac=sample_fraction,
                random_state=self.config.experiments.random_state + chunk_index,
            )
            if not sampled_chunk.empty:
                sampled_chunks.append(sampled_chunk)

            self.logger.info(
                "Feature engineering chunk %s processed: chunk_rows=%s, cumulative_rows=%s.",
                chunk_index,
                len(enriched_chunk),
                engineered_rows,
            )

        sample_frame = pd.concat(sampled_chunks, ignore_index=True)
        if len(sample_frame) > self.feature_config.comparison_sample_size:
            sample_frame = sample_frame.sample(
                n=self.feature_config.comparison_sample_size,
                random_state=self.config.experiments.random_state,
            ).reset_index(drop=True)
        return sample_frame, engineered_rows

    def _enrich_chunk(
        self,
        chunk: pd.DataFrame,
        popularity_lookup: dict[int, float],
    ) -> pd.DataFrame:
        """Create domain-driven real-estate features for one chunk."""

        frame = chunk.copy()
        numeric_columns = [
            "Price",
            "Area",
            "Floor",
            "Number of floors",
            "Minutes to metro",
            "Number of rooms",
            "Region id",
        ]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame["price_per_m2"] = frame["Price"] / frame["Area"].clip(lower=1.0)
        frame["floor_ratio"] = (frame["Floor"] / frame["Number of floors"].clip(lower=1.0)).clip(0.0, 1.0)

        center_preference = (1.0 - (frame["floor_ratio"] - 0.55).abs() / 0.55).clip(0.0, 1.0)
        first_floor_penalty = np.where(frame["Floor"] <= 1.0, 0.15, 0.0)
        top_floor_penalty = np.where(frame["Floor"] >= frame["Number of floors"], 0.10, 0.0)
        frame["floor_comfort_score"] = (center_preference - first_floor_penalty - top_floor_penalty).clip(0.0, 1.0)

        station_presence = frame["Metro station"].fillna("").astype(str).str.strip().ne("").astype(float)
        minutes_component = (1.0 - frame["Minutes to metro"].clip(lower=0.0, upper=60.0) / 60.0).clip(0.0, 1.0)
        frame["transport_score"] = (0.75 * minutes_component + 0.25 * station_presence).clip(0.0, 1.0)

        frame["renovation_score"] = self._map_renovation_score(frame["Renovation"])

        district_ids = frame["Region id"].round().astype("Int64")
        frame["district_popularity_score"] = district_ids.map(popularity_lookup).fillna(float(np.mean(list(popularity_lookup.values()))))

        room_scores = frame["Number of rooms"].map(
            {
                0.0: 0.95,
                1.0: 1.00,
                2.0: 0.90,
                3.0: 0.75,
                4.0: 0.55,
                5.0: 0.45,
                6.0: 0.40,
                7.0: 0.35,
                8.0: 0.30,
            }
        ).fillna(0.50)
        size_liquidity = (1.0 - (frame["Area"] - 55.0).abs() / 55.0).clip(0.0, 1.0)
        frame["liquidity_score"] = (
            0.35 * frame["transport_score"]
            + 0.25 * room_scores
            + 0.20 * frame["floor_comfort_score"]
            + 0.20 * frame["district_popularity_score"]
        ).clip(0.0, 1.0)

        frame["investment_score"] = (
            0.30 * frame["district_popularity_score"]
            + 0.25 * frame["transport_score"]
            + 0.20 * frame["liquidity_score"]
            + 0.15 * frame["renovation_score"]
            + 0.10 * frame["floor_comfort_score"]
        ).clip(0.0, 1.0)
        return frame

    def _build_engineered_feature_summary(self, sample_frame: pd.DataFrame) -> pd.DataFrame:
        """Summarize the newly created features on the comparison sample."""

        feature_columns = list(self.feature_config.engineered_numeric_columns)
        summary = sample_frame.loc[:, feature_columns].describe(
            percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        ).transpose()
        summary.reset_index(names="feature_name", inplace=True)
        return summary

    def _compare_model_quality(self, sample_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compare baseline and enriched feature spaces using the stage-6 best model family."""

        train_frame, test_frame = train_test_split(
            sample_frame,
            test_size=self.feature_config.test_size,
            random_state=self.config.experiments.random_state,
            shuffle=True,
        )
        train_frame = train_frame.reset_index(drop=True)
        test_frame = test_frame.reset_index(drop=True)

        baseline_train, baseline_test = self._prepare_feature_sets(train_frame, test_frame, use_engineered=False)
        enhanced_train, enhanced_test = self._prepare_feature_sets(train_frame, test_frame, use_engineered=True)

        baseline_metrics, _ = self._fit_and_score_random_forest("baseline", baseline_train, baseline_test)
        enhanced_metrics, importance_frame = self._fit_and_score_random_forest("enhanced", enhanced_train, enhanced_test)

        impact_frame = pd.DataFrame([baseline_metrics, enhanced_metrics])
        impact_frame["delta_rmse_vs_baseline"] = impact_frame["rmse"] - float(baseline_metrics["rmse"])
        impact_frame["delta_mae_vs_baseline"] = impact_frame["mae"] - float(baseline_metrics["mae"])
        impact_frame["delta_r2_vs_baseline"] = impact_frame["r2"] - float(baseline_metrics["r2"])
        return impact_frame, importance_frame

    def _prepare_feature_sets(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        use_engineered: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Encode and scale baseline features, optionally appending engineered columns."""

        base_numeric_columns = [
            "Minutes to metro",
            "Number of rooms",
            "Area",
            "Living area",
            "Kitchen area",
            "Floor",
            "Number of floors",
            "Region id",
            "Geo latitude",
            "Geo longitude",
        ]
        categorical_columns = list(self.config.preprocessing.categorical_columns)
        engineered_columns = [
            column
            for column in self.feature_config.engineered_numeric_columns
            if column not in self.feature_config.leakage_sensitive_engineered_columns
        ]

        train_encoded = self._frequency_encode(train_frame, train_frame, categorical_columns)
        test_encoded = self._frequency_encode(train_frame, test_frame, categorical_columns)

        baseline_train = pd.DataFrame(index=train_frame.index)
        baseline_test = pd.DataFrame(index=test_frame.index)

        for column in categorical_columns:
            baseline_train[f"{column}_encoded"] = train_encoded[f"{column}_encoded"]
            baseline_test[f"{column}_encoded"] = test_encoded[f"{column}_encoded"]

        scaler = StandardScaler()
        baseline_train_scaled = scaler.fit_transform(train_frame[base_numeric_columns])
        baseline_test_scaled = scaler.transform(test_frame[base_numeric_columns])
        for idx, column in enumerate(base_numeric_columns):
            scaled_name = f"{column}_scaled"
            baseline_train[scaled_name] = baseline_train_scaled[:, idx]
            baseline_test[scaled_name] = baseline_test_scaled[:, idx]

        if use_engineered:
            engineered_scaler = StandardScaler()
            engineered_train_scaled = engineered_scaler.fit_transform(train_frame[engineered_columns])
            engineered_test_scaled = engineered_scaler.transform(test_frame[engineered_columns])
            for idx, column in enumerate(engineered_columns):
                scaled_name = f"{column}_scaled"
                baseline_train[scaled_name] = engineered_train_scaled[:, idx]
                baseline_test[scaled_name] = engineered_test_scaled[:, idx]

        return (
            {
                "X": baseline_train,
                "y": train_frame["Price"].to_numpy(dtype=float),
                "feature_names": baseline_train.columns.tolist(),
            },
            {
                "X": baseline_test,
                "y": test_frame["Price"].to_numpy(dtype=float),
                "feature_names": baseline_test.columns.tolist(),
            },
        )

    def _fit_and_score_random_forest(
        self,
        feature_set_name: str,
        train_bundle: dict[str, Any],
        test_bundle: dict[str, Any],
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Fit a controlled Random Forest and compute before/after metrics."""

        started_at = perf_counter()
        estimator = RandomForestRegressor(**self.config.price_modeling.random_forest_params)
        estimator.fit(train_bundle["X"], train_bundle["y"])
        predictions = estimator.predict(test_bundle["X"])
        elapsed = round(perf_counter() - started_at, 4)

        mse = float(mean_squared_error(test_bundle["y"], predictions))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(test_bundle["y"], predictions))
        r2 = float(r2_score(test_bundle["y"], predictions))

        feature_importance = pd.DataFrame(
            {
                "feature_name": train_bundle["feature_names"],
                "raw_importance": estimator.feature_importances_,
            }
        )
        feature_importance["relative_importance"] = feature_importance["raw_importance"] / feature_importance["raw_importance"].sum()
        feature_importance["is_engineered"] = feature_importance["feature_name"].str.contains(
            "floor_ratio|floor_comfort_score|transport_score|renovation_score|investment_score|liquidity_score|district_popularity_score"
        )
        feature_importance = feature_importance.sort_values(by="relative_importance", ascending=False).reset_index(drop=True)

        metrics = {
            "feature_set": feature_set_name,
            "feature_count": len(train_bundle["feature_names"]),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mse": round(mse, 4),
            "r2": round(r2, 6),
            "training_seconds": elapsed,
        }
        return metrics, feature_importance.head(20)

    def _frequency_encode(
        self,
        reference_frame: pd.DataFrame,
        target_frame: pd.DataFrame,
        categorical_columns: list[str],
    ) -> pd.DataFrame:
        encoded = target_frame.copy()
        for column in categorical_columns:
            frequencies = reference_frame[column].astype(str).value_counts(normalize=True)
            encoded[f"{column}_encoded"] = target_frame[column].astype(str).map(frequencies).fillna(0.0)
        return encoded

    def _plot_model_impact(self, impact_frame: pd.DataFrame) -> Path:
        figure, axes = plt.subplots(2, 2, figsize=(18, 12))
        metric_specs = [
            ("mae", "MAE", "#4c78a8"),
            ("rmse", "RMSE", "#f58518"),
            ("mse", "MSE", "#54a24b"),
            ("r2", "R²", "#e45756"),
        ]
        plot_frame = impact_frame.copy()
        plot_frame["feature_set_label"] = plot_frame["feature_set"].map(feature_set_label)
        sns.set_theme(style="whitegrid", context="talk")
        setup_plot_style()
        for axis, (column, title, color) in zip(axes.flatten(), metric_specs, strict=False):
            sns.barplot(data=plot_frame, x="feature_set_label", y=column, color=color, ax=axis)
            axis.set_title(f"{title}:\nbaseline vs. engineered features")
            axis.set_xlabel("Feature set")
            wrap_ticklabels(axis, axis_name="x", width=14, rotation=0, ha="center", fontsize=12)
            format_numeric_axis(axis, y=True, nbins=5)
        output_path = self.config.paths.plots_dir / self.feature_config.comparison_plot_filename
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.feature_config.plot_dpi, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def _plot_feature_importance(self, importance_frame: pd.DataFrame) -> Path:
        plot_frame = importance_frame.sort_values(by="relative_importance", ascending=True).copy()
        plot_frame["feature_name_label"] = plot_frame["feature_name"].map(feature_label)
        plot_frame["is_engineered_label"] = plot_frame["is_engineered"].map(boolean_label)
        figure, axis = plt.subplots(figsize=(12, 8))
        sns.set_theme(style="whitegrid", context="talk")
        setup_plot_style()
        sns.barplot(
            data=plot_frame,
            x="relative_importance",
            y="feature_name_label",
            hue="is_engineered_label",
            dodge=False,
            palette={"Yes": "#f58518", "No": "#4c78a8"},
            ax=axis,
        )
        axis.set_title("Feature importance in the engineered Random Forest model")
        axis.set_xlabel("Relative importance")
        axis.set_ylabel("Feature")
        legend = axis.get_legend()
        if legend is not None:
            legend.set_title("Engineered feature")
        output_path = self.config.paths.plots_dir / self.feature_config.importance_plot_filename
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.feature_config.plot_dpi, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def _iter_clean_chunks(self, usecols: list[str] | None = None):
        return pd.read_csv(
            self.input_path,
            usecols=usecols,
            chunksize=self.feature_config.chunk_size,
            low_memory=False,
        )

    @staticmethod
    def _map_renovation_score(renovation_series: pd.Series) -> pd.Series:
        values = renovation_series.fillna("unknown").astype(str).str.lower()
        conditions = [
            values.str.contains("designer", regex=False),
            values.str.contains("european", regex=False),
            values.str.contains("cosmetic", regex=False),
            values.str.contains("without", regex=False),
            values.str.contains("no renovation", regex=False),
        ]
        scores = [1.0, 0.8, 0.55, 0.2, 0.2]
        return pd.Series(np.select(conditions, scores, default=0.45), index=renovation_series.index, dtype=float)

    @staticmethod
    def _minmax_scale(series: pd.Series) -> pd.Series:
        minimum = float(series.min())
        maximum = float(series.max())
        if maximum - minimum == 0:
            return pd.Series(np.ones(len(series)), index=series.index, dtype=float)
        return ((series - minimum) / (maximum - minimum)).astype(float)
