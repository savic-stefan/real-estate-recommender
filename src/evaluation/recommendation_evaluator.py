from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.config import AppConfig, RecommendationEvaluationConfig
from src.utils.helpers import serialize_to_json
from src.utils.logger import get_logger
from src.visualization.recommendation_visualizer import RecommendationEvaluationVisualizer


@dataclass(slots=True)
class RecommendationBenchmarkProfile:
    """Represents the common benchmark profile used for offline recommender evaluation."""

    budget: float
    desired_area: float
    area_tolerance: float
    desired_rooms: float
    room_tolerance: float
    preferred_metro_station: str
    preferred_renovation: str
    min_investment_score: float
    min_liquidity_score: float


@dataclass(slots=True)
class RecommendationEvaluationSummary:
    """Metadata produced by the recommender offline-evaluation stage."""

    best_method_name: str
    best_method_selection_metric: str
    evaluated_methods: list[str]
    benchmark_profile: dict[str, Any]
    tables_created: list[str]
    plots_created: list[str]


class RecommendationEvaluator:
    """Evaluates recommendation algorithms through a shared offline utility benchmark."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.eval_config: RecommendationEvaluationConfig = config.recommendation_evaluation
        self.logger = get_logger(self.__class__.__name__)
        self.tables_dir = self.config.paths.tables_dir
        self.visualizer = RecommendationEvaluationVisualizer(self.config.paths.plots_dir, self.eval_config)

    def execute(self) -> RecommendationEvaluationSummary:
        """Run unified offline comparison across all recommendation algorithms."""

        benchmark_profile = self._build_benchmark_profile()
        method_frames = self._load_method_frames()
        item_level_frame = self._build_item_level_evaluation(method_frames, benchmark_profile)
        comparison_frame = self._aggregate_method_metrics(item_level_frame)
        comparison_frame = comparison_frame.sort_values(
            by=self.eval_config.best_method_selection_metric,
            ascending=False,
        ).reset_index(drop=True)

        comparison_path = self.tables_dir / self.eval_config.comparison_table_filename
        item_level_path = self.tables_dir / self.eval_config.item_level_table_filename

        comparison_frame.to_csv(comparison_path, index=False, encoding="utf-8-sig")
        item_level_frame.to_csv(item_level_path, index=False, encoding="utf-8-sig")

        overall_plot_path = self.visualizer.plot_overall_scores(
            comparison_frame,
            self.eval_config.overall_score_plot_filename,
        )
        heatmap_path = self.visualizer.plot_metric_heatmap(
            comparison_frame,
            self.eval_config.metric_heatmap_filename,
        )

        best_method_name = str(comparison_frame.iloc[0]["method_name"])
        summary = RecommendationEvaluationSummary(
            best_method_name=best_method_name,
            best_method_selection_metric=self.eval_config.best_method_selection_metric,
            evaluated_methods=comparison_frame["method_name"].tolist(),
            benchmark_profile=asdict(benchmark_profile),
            tables_created=[
                str(comparison_path),
                str(item_level_path),
            ],
            plots_created=[
                str(overall_plot_path),
                str(heatmap_path),
            ],
        )
        serialize_to_json(
            summary,
            self.config.paths.metadata_dir / self.eval_config.summary_filename,
        )
        self.logger.info(
            "Stage 12 completed. Best recommendation method by %s: %s.",
            self.eval_config.best_method_selection_metric,
            best_method_name,
        )
        return summary

    def _build_benchmark_profile(self) -> RecommendationBenchmarkProfile:
        return RecommendationBenchmarkProfile(
            budget=self.config.hybrid.default_budget,
            desired_area=self.config.hybrid.default_area,
            area_tolerance=self.config.hybrid.default_area_tolerance,
            desired_rooms=self.config.hybrid.default_rooms,
            room_tolerance=self.config.hybrid.default_room_tolerance,
            preferred_metro_station=self.config.hybrid.default_metro_station,
            preferred_renovation=self.config.hybrid.default_renovation,
            min_investment_score=self.config.hybrid.min_investment_score,
            min_liquidity_score=self.config.hybrid.min_liquidity_score,
        )

    def _load_method_frames(self) -> dict[str, pd.DataFrame]:
        content_path = self.tables_dir / self.eval_config.content_input_filename
        collaborative_path = self.tables_dir / self.eval_config.collaborative_input_filename
        knowledge_path = self.tables_dir / self.eval_config.knowledge_input_filename
        hybrid_path = self.tables_dir / self.eval_config.hybrid_input_filename

        required_paths = [content_path, collaborative_path, knowledge_path, hybrid_path]
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "Required recommendation outputs are missing: "
                + ", ".join(missing_paths)
                + ". Run stages 8-11 first."
            )

        content_frame = pd.read_csv(content_path)
        collaborative_frame = pd.read_csv(collaborative_path)
        knowledge_frame = pd.read_csv(knowledge_path)
        hybrid_frame = pd.read_csv(hybrid_path)

        frames = {
            "content_cosine": content_frame.loc[content_frame["method"] == "cosine_similarity"].copy(),
            "content_euclidean": content_frame.loc[content_frame["method"] == "euclidean_similarity"].copy(),
            "content_manhattan": content_frame.loc[content_frame["method"] == "manhattan_similarity"].copy(),
            "content_weighted": content_frame.loc[content_frame["method"] == "weighted_similarity"].copy(),
            "collaborative_user_based": collaborative_frame.loc[collaborative_frame["method"] == "user_based_cf"].copy(),
            "collaborative_item_based": collaborative_frame.loc[collaborative_frame["method"] == "item_based_cf"].copy(),
            "knowledge_based": knowledge_frame.copy(),
            "hybrid": hybrid_frame.copy(),
        }
        return {
            method_name: self._prepare_method_frame(frame.head(self.eval_config.top_k).copy(), method_name)
            for method_name, frame in frames.items()
            if not frame.empty
        }

    def _prepare_method_frame(self, frame: pd.DataFrame, method_name: str) -> pd.DataFrame:
        prepared = frame.copy()
        prepared["method_name"] = method_name
        rank_series = pd.to_numeric(prepared["rank"], errors="coerce")
        fallback_rank = pd.Series(np.arange(1, len(prepared) + 1), index=prepared.index, dtype=float)
        prepared["rank"] = rank_series.where(rank_series.notna(), fallback_rank)
        for column in ("Price", "Area", "Number of rooms", "investment_score", "liquidity_score"):
            if column in prepared.columns:
                prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        if "Metro station" in prepared.columns:
            prepared["Metro station"] = prepared["Metro station"].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
        if "Renovation" in prepared.columns:
            prepared["Renovation"] = prepared["Renovation"].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
        return prepared.reset_index(drop=True)

    def _build_item_level_evaluation(
        self,
        method_frames: dict[str, pd.DataFrame],
        profile: RecommendationBenchmarkProfile,
    ) -> pd.DataFrame:
        evaluated_frames: list[pd.DataFrame] = []
        for method_name, frame in method_frames.items():
            evaluated = frame.copy()
            evaluated["price_fit"] = self._bounded_closeness(
                evaluated["Price"],
                profile.budget,
                scale=max(profile.budget, 1.0),
            )
            evaluated["area_fit"] = self._bounded_closeness(
                evaluated["Area"],
                profile.desired_area,
                scale=max(profile.area_tolerance, 1.0),
            )
            evaluated["rooms_fit"] = self._bounded_closeness(
                evaluated["Number of rooms"],
                profile.desired_rooms,
                scale=max(profile.room_tolerance, 1.0),
            )
            evaluated["metro_match"] = (evaluated["Metro station"] == profile.preferred_metro_station).astype(float)
            evaluated["renovation_match"] = (evaluated["Renovation"] == profile.preferred_renovation).astype(float)
            evaluated["known_metro"] = (evaluated["Metro station"] != "Unknown").astype(float)
            evaluated["known_renovation"] = (evaluated["Renovation"] != "Unknown").astype(float)

            evaluated["budget_rule"] = (evaluated["Price"] <= profile.budget).astype(float)
            evaluated["area_rule"] = evaluated["Area"].between(
                profile.desired_area - profile.area_tolerance,
                profile.desired_area + profile.area_tolerance,
            ).astype(float)
            evaluated["rooms_rule"] = evaluated["Number of rooms"].between(
                profile.desired_rooms - profile.room_tolerance,
                profile.desired_rooms + profile.room_tolerance,
            ).astype(float)
            evaluated["investment_rule"] = (
                evaluated["investment_score"] >= profile.min_investment_score
            ).astype(float)
            evaluated["liquidity_rule"] = (
                evaluated["liquidity_score"] >= profile.min_liquidity_score
            ).astype(float)
            evaluated["hard_constraint_fraction"] = (
                evaluated["budget_rule"]
                + evaluated["area_rule"]
                + evaluated["rooms_rule"]
                + evaluated["investment_rule"]
                + evaluated["liquidity_rule"]
            ) / 5.0
            evaluated["all_hard_constraints_satisfied"] = (evaluated["hard_constraint_fraction"] == 1.0).astype(float)

            evaluated["item_utility"] = (
                0.18 * evaluated["price_fit"]
                + 0.16 * evaluated["area_fit"]
                + 0.12 * evaluated["rooms_fit"]
                + 0.12 * evaluated["metro_match"]
                + 0.10 * evaluated["renovation_match"]
                + 0.14 * evaluated["investment_score"]
                + 0.10 * evaluated["liquidity_score"]
                + 0.08 * evaluated["hard_constraint_fraction"]
            )
            evaluated["discount_weight"] = 1.0 / np.log2(evaluated["rank"] + 1.0)
            evaluated["dcg_component"] = evaluated["item_utility"] * evaluated["discount_weight"]
            evaluated_frames.append(evaluated)

        return pd.concat(evaluated_frames, ignore_index=True)

    def _aggregate_method_metrics(self, item_level_frame: pd.DataFrame) -> pd.DataFrame:
        grouped = item_level_frame.groupby("method_name", dropna=False)
        records: list[dict[str, Any]] = []
        for method_name, frame in grouped:
            top1_utility = float(frame.sort_values("rank").iloc[0]["item_utility"])
            records.append(
                {
                    "method_name": str(method_name),
                    "evaluated_items": int(len(frame)),
                    "avg_item_utility": round(float(frame["item_utility"].mean()), 6),
                    "top1_utility": round(top1_utility, 6),
                    "dcg_utility": round(float(frame["dcg_component"].sum()), 6),
                    "hard_constraint_rate": round(float(frame["all_hard_constraints_satisfied"].mean()), 6),
                    "mean_hard_constraint_fraction": round(float(frame["hard_constraint_fraction"].mean()), 6),
                    "mean_price_fit": round(float(frame["price_fit"].mean()), 6),
                    "mean_area_fit": round(float(frame["area_fit"].mean()), 6),
                    "mean_rooms_fit": round(float(frame["rooms_fit"].mean()), 6),
                    "metro_match_rate": round(float(frame["metro_match"].mean()), 6),
                    "renovation_match_rate": round(float(frame["renovation_match"].mean()), 6),
                    "known_metro_rate": round(float(frame["known_metro"].mean()), 6),
                    "known_renovation_rate": round(float(frame["known_renovation"].mean()), 6),
                    "mean_investment_score": round(float(frame["investment_score"].mean()), 6),
                    "mean_liquidity_score": round(float(frame["liquidity_score"].mean()), 6),
                }
            )
        return pd.DataFrame(records)

    @staticmethod
    def _bounded_closeness(values: pd.Series, target: float, scale: float) -> pd.Series:
        return (1.0 / (1.0 + (pd.to_numeric(values, errors="coerce") - target).abs() / max(scale, 1e-9))).clip(0.0, 1.0)
