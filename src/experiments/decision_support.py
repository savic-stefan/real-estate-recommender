from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.config import AppConfig, DecisionSupportConfig
from src.models.price_model import RandomForestPriceModel
from src.utils.helpers import serialize_to_json
from src.utils.logger import get_logger
from src.visualization.decision_support_visualizer import DecisionSupportVisualizer


@dataclass(slots=True)
class DecisionSupportProfile:
    """Represents the end-user profile for the integrated DSS shortlist."""

    budget: float
    desired_area: float
    area_tolerance: float
    desired_rooms: float
    room_tolerance: float
    preferred_metro_station: str
    preferred_renovation: str
    target_investment_score: float
    target_liquidity_score: float
    min_price_per_m2: float
    min_investment_score: float
    min_liquidity_score: float
    min_transport_score: float


@dataclass(slots=True)
class DecisionSupportSummary:
    """Metadata produced by the integrated decision-support stage."""

    input_path: str
    sampled_rows: int
    candidate_rows: int
    shortlisted_rows: int
    price_model_artifact_path: str
    recommendation_method: str
    tables_created: list[str]
    plots_created: list[str]


class DecisionSupportEngine:
    """Builds an end-to-end shortlist by combining price modeling and recommendation logic."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.ds_config: DecisionSupportConfig = config.decision_support
        self.logger = get_logger(self.__class__.__name__)
        self.input_path = self.config.paths.processed_data_dir / self.ds_config.input_filename
        self.visualizer = DecisionSupportVisualizer(self.config.paths.plots_dir, self.ds_config)

    def execute(self) -> DecisionSupportSummary:
        """Run the full decision-support workflow and persist the final shortlist."""

        if not self.input_path.exists():
            raise FileNotFoundError(f"Engineered dataset was not found at {self.input_path}. Run stage 7 first.")

        sample_frame = self._load_sample()
        sample_frame = self._prepare_sample(sample_frame)
        profile = self._build_profile()
        candidate_frame, diagnostics_frame = self._build_candidate_pool(sample_frame, profile)
        content_frame = self._compute_content_scores(candidate_frame, profile)
        scored_frame = self._predict_market_prices(content_frame)
        shortlist_frame = self._build_decision_scores(scored_frame)
        shortlist_frame = shortlist_frame.head(self.ds_config.top_k).reset_index(drop=True)
        shortlist_frame.insert(1, "rank", np.arange(1, len(shortlist_frame) + 1))
        shortlist_frame["explanation"] = shortlist_frame.apply(self._build_explanation, axis=1)

        component_frame = shortlist_frame.loc[
            :,
            [
                "rank",
                "listing_id",
                "preference_score",
                "valuation_score",
                "investment_score_component",
                "liquidity_score_component",
                "decision_support_score",
            ],
        ].copy()

        shortlist_path = self.config.paths.tables_dir / self.ds_config.shortlist_filename
        component_path = self.config.paths.tables_dir / self.ds_config.component_filename
        diagnostics_path = self.config.paths.tables_dir / self.ds_config.diagnostics_filename
        profile_path = self.config.paths.metadata_dir / self.ds_config.profile_filename

        shortlist_frame.to_csv(shortlist_path, index=False, encoding="utf-8-sig")
        component_frame.to_csv(component_path, index=False, encoding="utf-8-sig")
        diagnostics_frame.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
        serialize_to_json(profile, profile_path)

        value_gap_plot_path = self.visualizer.plot_value_gap(
            shortlist_frame,
            self.ds_config.value_gap_plot_filename,
        )
        component_plot_path = self.visualizer.plot_component_scores(
            shortlist_frame,
            self.ds_config.component_plot_filename,
        )

        artifact_path = self.config.paths.models_dir / "random_forest_regressor_price_model.joblib"
        summary = DecisionSupportSummary(
            input_path=str(self.input_path),
            sampled_rows=len(sample_frame),
            candidate_rows=len(candidate_frame),
            shortlisted_rows=len(shortlist_frame),
            price_model_artifact_path=str(artifact_path),
            recommendation_method="content_weighted_on_feasible_pool",
            tables_created=[
                str(shortlist_path),
                str(component_path),
                str(diagnostics_path),
                str(profile_path),
            ],
            plots_created=[
                str(value_gap_plot_path),
                str(component_plot_path),
            ],
        )
        serialize_to_json(
            summary,
            self.config.paths.metadata_dir / self.ds_config.summary_filename,
        )
        self.logger.info(
            "Stage 13 completed: sampled=%s, candidates=%s, shortlisted=%s.",
            summary.sampled_rows,
            summary.candidate_rows,
            summary.shortlisted_rows,
        )
        return summary

    def _load_sample(self) -> pd.DataFrame:
        sampled_chunks: list[pd.DataFrame] = []
        target_sample = self.ds_config.sample_size
        sample_fraction = min(1.0, target_sample / max(self._engineered_row_count_hint(), 1))
        for chunk_index, chunk in enumerate(self._iter_engineered_chunks(), start=1):
            sampled_chunk = chunk.sample(
                frac=sample_fraction,
                random_state=self.config.experiments.random_state + chunk_index,
            )
            if not sampled_chunk.empty:
                sampled_chunks.append(sampled_chunk)
        frame = pd.concat(sampled_chunks, ignore_index=True)
        if len(frame) > target_sample:
            frame = frame.sample(
                n=target_sample,
                random_state=self.config.experiments.random_state,
            ).reset_index(drop=True)
        self.logger.info("Prepared decision-support sample with %s rows.", len(frame))
        return frame

    def _prepare_sample(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        prepared["listing_id"] = np.arange(1, len(prepared) + 1)
        numeric_columns = [
            "Price",
            "Area",
            "Living area",
            "Kitchen area",
            "Floor",
            "Number of floors",
            "Number of rooms",
            "Minutes to metro",
            "Region id",
            "Geo latitude",
            "Geo longitude",
            "price_per_m2",
            "transport_score",
            "investment_score",
            "liquidity_score",
        ]
        for column in numeric_columns:
            if column in prepared.columns:
                prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

        categorical_columns = [
            "Apartment type",
            "Metro station",
            "Region",
            "Renovation",
            "Building type",
            "Object type",
            "Date",
            "Time",
            "Source dataset",
        ]
        for column in categorical_columns:
            if column in prepared.columns:
                prepared[column] = prepared[column].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})

        return prepared.dropna(
            subset=[
                "Price",
                "Area",
                "Number of rooms",
                "price_per_m2",
                "investment_score",
                "liquidity_score",
                "transport_score",
            ]
        ).reset_index(drop=True)

    def _build_profile(self) -> DecisionSupportProfile:
        return DecisionSupportProfile(
            budget=self.ds_config.budget,
            desired_area=self.ds_config.desired_area,
            area_tolerance=self.ds_config.area_tolerance,
            desired_rooms=self.ds_config.desired_rooms,
            room_tolerance=self.ds_config.room_tolerance,
            preferred_metro_station=self.ds_config.preferred_metro_station,
            preferred_renovation=self.ds_config.preferred_renovation,
            target_investment_score=self.ds_config.target_investment_score,
            target_liquidity_score=self.ds_config.target_liquidity_score,
            min_price_per_m2=self.ds_config.min_price_per_m2,
            min_investment_score=self.ds_config.min_investment_score,
            min_liquidity_score=self.ds_config.min_liquidity_score,
            min_transport_score=self.ds_config.min_transport_score,
        )

    def _build_candidate_pool(
        self,
        sample_frame: pd.DataFrame,
        profile: DecisionSupportProfile,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        diagnostics: list[dict[str, Any]] = []
        current = sample_frame.copy()
        diagnostics.append({"stage": "initial_pool", "remaining_rows": len(current), "description": "Initial sampled pool"})

        rules = [
            (
                "budget_with_slack",
                current["Price"] <= profile.budget * (1.0 + self.ds_config.max_budget_slack_ratio),
                f"Price <= budget * {1.0 + self.ds_config.max_budget_slack_ratio}",
            ),
            (
                "area_band",
                current["Area"].between(
                    profile.desired_area - profile.area_tolerance,
                    profile.desired_area + profile.area_tolerance,
                ),
                f"Area in [{profile.desired_area - profile.area_tolerance}, {profile.desired_area + profile.area_tolerance}]",
            ),
            (
                "rooms_band",
                current["Number of rooms"].between(
                    profile.desired_rooms - profile.room_tolerance,
                    profile.desired_rooms + profile.room_tolerance,
                ),
                f"Rooms in [{profile.desired_rooms - profile.room_tolerance}, {profile.desired_rooms + profile.room_tolerance}]",
            ),
            (
                "minimum_price_per_m2",
                current["price_per_m2"] >= profile.min_price_per_m2,
                f"price_per_m2 >= {profile.min_price_per_m2}",
            ),
            (
                "minimum_investment",
                current["investment_score"] >= profile.min_investment_score,
                f"investment_score >= {profile.min_investment_score}",
            ),
            (
                "minimum_liquidity",
                current["liquidity_score"] >= profile.min_liquidity_score,
                f"liquidity_score >= {profile.min_liquidity_score}",
            ),
            (
                "minimum_transport",
                current["transport_score"] >= profile.min_transport_score,
                f"transport_score >= {profile.min_transport_score}",
            ),
        ]

        mask = pd.Series(True, index=current.index)
        for stage_name, condition, description in rules:
            mask &= condition
            diagnostics.append(
                {
                    "stage": stage_name,
                    "remaining_rows": int(mask.sum()),
                    "description": description,
                }
            )
        candidate_frame = current.loc[mask].copy().reset_index(drop=True)
        return candidate_frame, pd.DataFrame(diagnostics)

    def _compute_content_scores(
        self,
        candidate_frame: pd.DataFrame,
        profile: DecisionSupportProfile,
    ) -> pd.DataFrame:
        frame = candidate_frame.copy()
        weights = self.config.content_based.weighted_similarity_weights
        price_sim = self._bounded_closeness(frame["Price"], profile.budget, scale=max(profile.budget, 1.0))
        area_sim = self._bounded_closeness(frame["Area"], profile.desired_area, scale=max(profile.desired_area, 1.0))
        rooms_sim = self._bounded_closeness(frame["Number of rooms"], profile.desired_rooms, scale=max(profile.desired_rooms, 1.0))
        metro_sim = (frame["Metro station"] == profile.preferred_metro_station).astype(float)
        renovation_sim = (frame["Renovation"] == profile.preferred_renovation).astype(float)
        investment_sim = self._bounded_closeness(frame["investment_score"], profile.target_investment_score, scale=max(profile.target_investment_score, 0.1))
        liquidity_sim = self._bounded_closeness(frame["liquidity_score"], profile.target_liquidity_score, scale=max(profile.target_liquidity_score, 0.1))
        frame["content_weighted_score"] = (
            weights["Price"] * price_sim
            + weights["Area"] * area_sim
            + weights["Number of rooms"] * rooms_sim
            + weights["Metro station"] * metro_sim
            + weights["Renovation"] * renovation_sim
            + weights["investment_score"] * investment_sim
            + weights["liquidity_score"] * liquidity_sim
        )
        return frame

    def _predict_market_prices(self, candidate_frame: pd.DataFrame) -> pd.DataFrame:
        encoders = joblib.load(self.config.paths.models_dir / "categorical_frequency_encoders.joblib")
        scaler_bundle = joblib.load(self.config.paths.models_dir / "standard_scaler.joblib")
        model = RandomForestPriceModel.load(self.config.paths.models_dir / "random_forest_regressor_price_model.joblib")

        numeric_columns = list(scaler_bundle["columns"])
        scaler = scaler_bundle["scaler"]

        feature_frame = pd.DataFrame(index=candidate_frame.index)
        for column, mapping in encoders.items():
            feature_frame[f"{column}_encoded"] = candidate_frame[column].astype(str).map(mapping).fillna(0.0)

        numeric_values = candidate_frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
        scaled_values = scaler.transform(numeric_values)
        for index, column in enumerate(numeric_columns):
            feature_frame[f"{column}_scaled"] = scaled_values[:, index]

        feature_frame = feature_frame.loc[:, model.feature_columns]
        predicted_prices = model.predict(feature_frame)

        scored = candidate_frame.copy()
        scored["predicted_market_price"] = predicted_prices
        scored["absolute_price_gap"] = scored["predicted_market_price"] - scored["Price"]
        scored["relative_discount_to_market"] = np.where(
            scored["predicted_market_price"] != 0,
            scored["absolute_price_gap"] / scored["predicted_market_price"],
            0.0,
        )
        return scored

    def _build_decision_scores(self, scored_frame: pd.DataFrame) -> pd.DataFrame:
        frame = scored_frame.copy()
        frame["preference_score"] = self._normalize_series(frame["content_weighted_score"])
        frame["valuation_score"] = self._normalize_series(frame["relative_discount_to_market"].clip(lower=-0.25, upper=0.40))
        frame["investment_score_component"] = self._normalize_series(frame["investment_score"])
        frame["liquidity_score_component"] = self._normalize_series(frame["liquidity_score"])
        frame["decision_support_score"] = (
            0.42 * frame["preference_score"]
            + 0.30 * frame["valuation_score"]
            + 0.16 * frame["investment_score_component"]
            + 0.12 * frame["liquidity_score_component"]
        )
        frame = frame.sort_values(
            by=["decision_support_score", "relative_discount_to_market", "content_weighted_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        return frame

    def _build_explanation(self, row: pd.Series) -> str:
        reasons: list[str] = []
        if row["relative_discount_to_market"] > 0.10:
            reasons.append("below_estimated_market_price")
        if row["content_weighted_score"] >= 0.70:
            reasons.append("strong_preference_match")
        if row["Metro station"] == self.ds_config.preferred_metro_station:
            reasons.append("preferred_metro_match")
        if row["Renovation"] == self.ds_config.preferred_renovation:
            reasons.append("preferred_renovation_match")
        if row["investment_score"] >= 0.70:
            reasons.append("high_investment_potential")
        if row["liquidity_score"] >= 0.80:
            reasons.append("high_liquidity")
        return "; ".join(reasons) if reasons else "balanced_candidate"

    def _iter_engineered_chunks(self):
        return pd.read_csv(
            self.input_path,
            chunksize=self.ds_config.chunk_size,
            low_memory=False,
        )

    def _engineered_row_count_hint(self) -> int:
        summary_path = self.config.paths.metadata_dir / self.config.feature_engineering.summary_filename
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            return int(payload["engineered_rows"])
        return self.ds_config.sample_size

    @staticmethod
    def _bounded_closeness(values: pd.Series, target: float, scale: float) -> pd.Series:
        return (1.0 / (1.0 + (values - target).abs() / max(scale, 1e-9))).clip(0.0, 1.0)

    @staticmethod
    def _normalize_series(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
        minimum = float(values.min())
        maximum = float(values.max())
        if maximum - minimum == 0:
            return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
        return (values - minimum) / (maximum - minimum)
