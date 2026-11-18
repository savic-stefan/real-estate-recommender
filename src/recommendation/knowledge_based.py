from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config.config import AppConfig, KnowledgeBasedConfig
from src.utils.helpers import serialize_to_json
from src.utils.logger import get_logger
from src.visualization.plot_labels import (
    format_numeric_axis,
    setup_plot_style,
    boolean_label,
    method_label,
    rule_stage_label,
    wrap_ticklabels,
)


@dataclass(slots=True)
class KnowledgePreferenceProfile:
    """Represents explicit user constraints for rule-based housing recommendation."""

    max_budget: float
    target_area: float
    area_tolerance: float
    target_rooms: float
    room_tolerance: float
    preferred_metro_station: str
    preferred_renovation: str
    min_price_per_m2: float
    min_investment_score: float
    min_liquidity_score: float
    min_transport_score: float


@dataclass(slots=True)
class KnowledgeBasedSummary:
    """Metadata produced by the knowledge-based recommendation stage."""

    input_path: str
    sampled_rows: int
    hard_filtered_rows: int
    final_candidate_rows: int
    top_k: int
    user_profile: dict[str, Any]
    optional_rules_applied: list[str]
    tables_created: list[str]
    plots_created: list[str]


class KnowledgeBasedRecommender:
    """Builds explainable rule-based recommendations under explicit user constraints."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.kb_config: KnowledgeBasedConfig = config.knowledge_based
        self.logger = get_logger(self.__class__.__name__)
        self.input_path = self.config.paths.processed_data_dir / self.kb_config.input_filename

    def execute(self) -> KnowledgeBasedSummary:
        """Run filtering, rule-based ranking, and comparison with content-based recommendations."""

        if not self.input_path.exists():
            raise FileNotFoundError(f"Engineered dataset was not found at {self.input_path}. Run stage 7 first.")

        sample_frame = self._load_recommendation_sample()
        sample_frame = self._prepare_sample(sample_frame)
        profile = self._build_user_profile()

        hard_filtered, diagnostics_frame = self._apply_hard_filters(sample_frame, profile)
        final_candidates, optional_rules_applied, diagnostics_frame = self._apply_optional_rules(
            hard_filtered,
            profile,
            diagnostics_frame,
        )
        ranked_frame = self._rank_candidates(final_candidates, profile)
        top_recommendations = ranked_frame.head(self.kb_config.top_k).reset_index(drop=True)

        content_based_reference = self._load_content_based_reference()
        comparison_frame = self._build_comparison_frame(top_recommendations, content_based_reference)

        recommendations_path = self.config.paths.tables_dir / self.kb_config.recommendations_filename
        diagnostics_path = self.config.paths.tables_dir / self.kb_config.diagnostics_filename
        comparison_path = self.config.paths.tables_dir / self.kb_config.comparison_filename

        top_recommendations.to_csv(recommendations_path, index=False, encoding="utf-8-sig")
        diagnostics_frame.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
        comparison_frame.to_csv(comparison_path, index=False, encoding="utf-8-sig")

        funnel_plot_path = self._plot_filter_funnel(diagnostics_frame)
        comparison_plot_path = self._plot_comparison(comparison_frame)

        summary = KnowledgeBasedSummary(
            input_path=str(self.input_path),
            sampled_rows=len(sample_frame),
            hard_filtered_rows=len(hard_filtered),
            final_candidate_rows=len(final_candidates),
            top_k=self.kb_config.top_k,
            user_profile=asdict(profile),
            optional_rules_applied=optional_rules_applied,
            tables_created=[
                str(recommendations_path),
                str(diagnostics_path),
                str(comparison_path),
            ],
            plots_created=[
                str(funnel_plot_path),
                str(comparison_plot_path),
            ],
        )
        serialize_to_json(
            summary,
            self.config.paths.metadata_dir / self.kb_config.summary_filename,
        )
        self.logger.info(
            "Stage 10 completed: sampled=%s, hard_filtered=%s, final_candidates=%s.",
            summary.sampled_rows,
            summary.hard_filtered_rows,
            summary.final_candidate_rows,
        )
        return summary

    def _load_recommendation_sample(self) -> pd.DataFrame:
        sampled_chunks: list[pd.DataFrame] = []
        target_sample = self.kb_config.sample_size
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
        self.logger.info("Prepared knowledge-based sample with %s rows.", len(frame))
        return frame

    def _prepare_sample(self, sample_frame: pd.DataFrame) -> pd.DataFrame:
        frame = sample_frame.copy()
        frame["listing_id"] = np.arange(1, len(frame) + 1)

        numeric_columns = [
            "Price",
            "Area",
            "Number of rooms",
            "price_per_m2",
            "transport_score",
            "investment_score",
            "liquidity_score",
        ]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame["Metro station"] = (
            frame["Metro station"].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
        )
        frame["Renovation"] = (
            frame["Renovation"].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
        )
        frame = frame.dropna(subset=numeric_columns).reset_index(drop=True)
        return frame

    def _build_user_profile(self) -> KnowledgePreferenceProfile:
        return KnowledgePreferenceProfile(
            max_budget=self.kb_config.default_max_budget,
            target_area=self.kb_config.default_target_area,
            area_tolerance=self.kb_config.default_area_tolerance,
            target_rooms=self.kb_config.default_target_rooms,
            room_tolerance=self.kb_config.default_room_tolerance,
            preferred_metro_station=self.kb_config.default_preferred_metro_station,
            preferred_renovation=self.kb_config.default_preferred_renovation,
            min_price_per_m2=self.kb_config.min_price_per_m2,
            min_investment_score=self.kb_config.min_investment_score,
            min_liquidity_score=self.kb_config.min_liquidity_score,
            min_transport_score=self.kb_config.min_transport_score,
        )

    def _apply_hard_filters(
        self,
        sample_frame: pd.DataFrame,
        profile: KnowledgePreferenceProfile,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        diagnostics: list[dict[str, Any]] = []
        current = sample_frame.copy()
        diagnostics.append(
            {
                "stage": "initial_pool",
                "remaining_rows": len(current),
                "applied": True,
                "description": "Initial sampled candidate pool",
            }
        )

        rule_specs = [
            (
                "max_budget",
                current["Price"] <= profile.max_budget,
                f"Price <= {profile.max_budget}",
            ),
            (
                "target_area_band",
                current["Area"].between(
                    profile.target_area - profile.area_tolerance,
                    profile.target_area + profile.area_tolerance,
                ),
                f"Area in [{profile.target_area - profile.area_tolerance}, {profile.target_area + profile.area_tolerance}]",
            ),
            (
                "target_rooms_band",
                current["Number of rooms"].between(
                    profile.target_rooms - profile.room_tolerance,
                    profile.target_rooms + profile.room_tolerance,
                ),
                f"Rooms in [{profile.target_rooms - profile.room_tolerance}, {profile.target_rooms + profile.room_tolerance}]",
            ),
            (
                "minimum_price_per_m2",
                current["price_per_m2"] >= profile.min_price_per_m2,
                f"price_per_m2 >= {profile.min_price_per_m2}",
            ),
            (
                "minimum_investment_score",
                current["investment_score"] >= profile.min_investment_score,
                f"investment_score >= {profile.min_investment_score}",
            ),
            (
                "minimum_liquidity_score",
                current["liquidity_score"] >= profile.min_liquidity_score,
                f"liquidity_score >= {profile.min_liquidity_score}",
            ),
            (
                "minimum_transport_score",
                current["transport_score"] >= profile.min_transport_score,
                f"transport_score >= {profile.min_transport_score}",
            ),
        ]

        running_mask = pd.Series(True, index=current.index)
        for stage_name, condition, description in rule_specs:
            running_mask &= condition
            diagnostics.append(
                {
                    "stage": stage_name,
                    "remaining_rows": int(running_mask.sum()),
                    "applied": True,
                    "description": description,
                }
            )

        filtered = current.loc[running_mask].copy()
        return filtered.reset_index(drop=True), pd.DataFrame(diagnostics)

    def _apply_optional_rules(
        self,
        candidate_frame: pd.DataFrame,
        profile: KnowledgePreferenceProfile,
        diagnostics_frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
        current = candidate_frame.copy()
        optional_rules_applied: list[str] = []
        diagnostics_records = diagnostics_frame.to_dict("records")

        if len(current) < self.kb_config.min_candidates_after_hard_filters:
            diagnostics_records.append(
                {
                    "stage": "optional_rules_skipped",
                    "remaining_rows": len(current),
                    "applied": False,
                    "description": "Optional rules skipped because hard-filtered pool is small",
                }
            )
            return current.reset_index(drop=True), optional_rules_applied, pd.DataFrame(diagnostics_records)

        optional_specs = [
            (
                "preferred_metro_station",
                current["Metro station"] == profile.preferred_metro_station,
                f"Metro station == {profile.preferred_metro_station}",
            ),
            (
                "preferred_renovation",
                current["Renovation"] == profile.preferred_renovation,
                f"Renovation == {profile.preferred_renovation}",
            ),
            (
                "known_metro_value",
                current["Metro station"] != "Unknown",
                "Metro station is known",
            ),
            (
                "known_renovation_value",
                current["Renovation"] != "Unknown",
                "Renovation is known",
            ),
        ]

        for rule_name, base_condition, description in optional_specs:
            condition = base_condition.loc[current.index]
            retained_rows = int(condition.sum())
            should_apply = retained_rows >= self.kb_config.min_candidates_after_optional_filters
            if should_apply:
                current = current.loc[condition].copy()
                optional_rules_applied.append(rule_name)
            diagnostics_records.append(
                {
                    "stage": rule_name,
                    "remaining_rows": len(current) if should_apply else retained_rows,
                    "applied": should_apply,
                    "description": description,
                }
            )

        return current.reset_index(drop=True), optional_rules_applied, pd.DataFrame(diagnostics_records)

    def _rank_candidates(
        self,
        candidate_frame: pd.DataFrame,
        profile: KnowledgePreferenceProfile,
    ) -> pd.DataFrame:
        ranked = candidate_frame.copy()
        ranked["budget_fit"] = (1.0 - (ranked["Price"] / profile.max_budget)).clip(0.0, 1.0)
        ranked["area_fit"] = self._bounded_closeness(
            ranked["Area"],
            profile.target_area,
            scale=max(profile.area_tolerance, 1.0),
        )
        ranked["rooms_fit"] = self._bounded_closeness(
            ranked["Number of rooms"],
            profile.target_rooms,
            scale=max(profile.room_tolerance, 1.0),
        )
        ranked["metro_fit"] = np.where(
            ranked["Metro station"] == profile.preferred_metro_station,
            1.0,
            np.where(ranked["Metro station"] == "Unknown", 0.15, 0.45),
        )
        ranked["renovation_fit"] = np.where(
            ranked["Renovation"] == profile.preferred_renovation,
            1.0,
            np.where(ranked["Renovation"] == "Unknown", 0.20, 0.55),
        )
        ranked["knowledge_score"] = (
            0.18 * ranked["budget_fit"]
            + 0.16 * ranked["area_fit"]
            + 0.12 * ranked["rooms_fit"]
            + 0.12 * ranked["metro_fit"]
            + 0.10 * ranked["renovation_fit"]
            + 0.12 * ranked["transport_score"]
            + 0.10 * ranked["investment_score"]
            + 0.10 * ranked["liquidity_score"]
        )
        ranked["hard_rules_satisfied"] = 7
        ranked["soft_preference_matches"] = (
            (ranked["Metro station"] == profile.preferred_metro_station).astype(int)
            + (ranked["Renovation"] == profile.preferred_renovation).astype(int)
        )
        ranked["method"] = "knowledge_based"
        ranked = ranked.sort_values(
            by=["knowledge_score", "investment_score", "liquidity_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        ranked.insert(1, "rank", np.arange(1, len(ranked) + 1))
        return ranked.loc[
            :,
            [
                "listing_id",
                "rank",
                "Price",
                "Area",
                "Number of rooms",
                "Metro station",
                "Renovation",
                "transport_score",
                "investment_score",
                "liquidity_score",
                "budget_fit",
                "area_fit",
                "rooms_fit",
                "metro_fit",
                "renovation_fit",
                "knowledge_score",
                "hard_rules_satisfied",
                "soft_preference_matches",
                "method",
            ],
        ]

    def _load_content_based_reference(self) -> pd.DataFrame:
        content_path = self.config.paths.tables_dir / self.config.content_based.top_recommendations_filename
        if not content_path.exists():
            raise FileNotFoundError("Content-based recommendations are missing. Run stage 8 first.")
        frame = pd.read_csv(content_path)
        reference = frame.loc[frame["method"] == "weighted_similarity"].copy()
        reference["hard_rules_satisfied"] = self._count_hard_rule_satisfaction(reference)
        reference["soft_preference_matches"] = (
            (reference["Metro station"] == self.kb_config.default_preferred_metro_station).astype(int)
            + (reference["Renovation"] == self.kb_config.default_preferred_renovation).astype(int)
        )
        reference["method"] = "content_based_reference"
        return reference.reset_index(drop=True)

    def _build_comparison_frame(
        self,
        knowledge_based: pd.DataFrame,
        content_based_reference: pd.DataFrame,
    ) -> pd.DataFrame:
        content_frame = content_based_reference.copy()
        if "knowledge_score" not in content_frame.columns:
            content_frame["knowledge_score"] = np.nan
        comparison = pd.concat([knowledge_based, content_frame], ignore_index=True, sort=False)
        comparison["rank_position"] = comparison.groupby("method").cumcount() + 1
        return comparison

    def _plot_filter_funnel(self, diagnostics_frame: pd.DataFrame) -> Path:
        plot_frame = diagnostics_frame.copy()
        plot_frame["stage_label"] = plot_frame["stage"].map(rule_stage_label)
        plot_frame["applied_label"] = plot_frame["applied"].map(boolean_label)
        sns.set_theme(style="whitegrid", context="talk")
        setup_plot_style()
        figure, axis = plt.subplots(figsize=(22, 9))
        sns.barplot(
            data=plot_frame,
            x="stage_label",
            y="remaining_rows",
            hue="applied_label",
            dodge=False,
            palette={"Yes": "#4c78a8", "No": "#b8b8b8"},
            ax=axis,
        )
        axis.set_title("Knowledge-based filtering funnel")
        axis.set_xlabel("Filter stage")
        axis.set_ylabel("Remaining properties")
        wrap_ticklabels(axis, axis_name="x", width=14, rotation=18, ha="right", fontsize=10)
        format_numeric_axis(axis, y=True, nbins=6)
        legend = axis.get_legend()
        if legend is not None:
            legend.set_title("Applied")
        output_path = self.config.paths.plots_dir / self.kb_config.funnel_plot_filename
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.kb_config.plot_dpi, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def _plot_comparison(self, comparison_frame: pd.DataFrame) -> Path:
        plot_frame = comparison_frame.copy()
        plot_frame["comparison_score"] = np.where(
            plot_frame["method"] == "knowledge_based",
            plot_frame["knowledge_score"],
            plot_frame["score"],
        )
        plot_frame["method_label"] = plot_frame["method"].map(method_label)
        sns.set_theme(style="whitegrid", context="talk")
        setup_plot_style()
        figure, axis = plt.subplots(figsize=(13.5, 7.5))
        sns.lineplot(
            data=plot_frame,
            x="rank_position",
            y="comparison_score",
            hue="method_label",
            marker="o",
            linewidth=2.5,
            ax=axis,
        )
        axis.set_title("Knowledge-based and content-based rankings")
        axis.set_xlabel("Rank")
        axis.set_ylabel("Ranking score")
        legend = axis.get_legend()
        if legend is not None:
            legend.set_title("Method")
            sns.move_legend(axis, "upper left", bbox_to_anchor=(1.02, 1.0), title="Method")
        output_path = self.config.paths.plots_dir / self.kb_config.comparison_plot_filename
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.kb_config.plot_dpi, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def _iter_engineered_chunks(self):
        return pd.read_csv(
            self.input_path,
            chunksize=self.kb_config.chunk_size,
            low_memory=False,
        )

    def _engineered_row_count_hint(self) -> int:
        summary_path = self.config.paths.metadata_dir / self.config.feature_engineering.summary_filename
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            return int(payload["engineered_rows"])
        return self.kb_config.sample_size

    def _count_hard_rule_satisfaction(self, frame: pd.DataFrame) -> pd.Series:
        price_per_m2_component = (
            (pd.to_numeric(frame["price_per_m2"], errors="coerce") >= self.kb_config.min_price_per_m2).astype(int)
            if "price_per_m2" in frame.columns
            else pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)
        )
        transport_component = (
            (pd.to_numeric(frame["transport_score"], errors="coerce") >= self.kb_config.min_transport_score).astype(int)
            if "transport_score" in frame.columns
            else pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)
        )
        return (
            (pd.to_numeric(frame["Price"], errors="coerce") <= self.kb_config.default_max_budget).astype(int)
            + pd.to_numeric(frame["Area"], errors="coerce").between(
                self.kb_config.default_target_area - self.kb_config.default_area_tolerance,
                self.kb_config.default_target_area + self.kb_config.default_area_tolerance,
            ).astype(int)
            + pd.to_numeric(frame["Number of rooms"], errors="coerce").between(
                self.kb_config.default_target_rooms - self.kb_config.default_room_tolerance,
                self.kb_config.default_target_rooms + self.kb_config.default_room_tolerance,
            ).astype(int)
            + price_per_m2_component
            + (pd.to_numeric(frame["investment_score"], errors="coerce") >= self.kb_config.min_investment_score).astype(int)
            + (pd.to_numeric(frame["liquidity_score"], errors="coerce") >= self.kb_config.min_liquidity_score).astype(int)
            + transport_component
        )

    @staticmethod
    def _bounded_closeness(values: pd.Series, target: float, scale: float) -> pd.Series:
        return (1.0 / (1.0 + (values - target).abs() / max(scale, 1e-9))).clip(0.0, 1.0)
