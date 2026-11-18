from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config.config import AppConfig, HybridRecommendationConfig
from src.utils.helpers import serialize_to_json
from src.utils.logger import get_logger
from src.visualization.plot_labels import (
    setup_plot_style,
    component_label,
    method_label,
)


@dataclass(slots=True)
class HybridPreferenceProfile:
    """Unified preference profile reused across content, collaborative, and hybrid scoring."""

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
class HybridSummary:
    """Metadata produced by the hybrid recommendation stage."""

    input_path: str
    sampled_rows: int
    candidate_rows: int
    interaction_rows: int
    top_k: int
    user_profile: dict[str, Any]
    weights: dict[str, float]
    methods_compared: list[str]
    tables_created: list[str]
    plots_created: list[str]


class HybridRecommender:
    """Combines knowledge filters, content similarity, and collaborative signals into one ranking."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.hybrid_config: HybridRecommendationConfig = config.hybrid
        self.logger = get_logger(self.__class__.__name__)
        self.input_path = self.config.paths.processed_data_dir / self.hybrid_config.input_filename

    def execute(self) -> HybridSummary:
        """Run the end-to-end hybrid recommendation experiment on a shared sample."""

        if not self.input_path.exists():
            raise FileNotFoundError(f"Engineered dataset was not found at {self.input_path}. Run stage 7 first.")

        sample_frame = self._load_sample()
        sample_frame = self._prepare_sample(sample_frame)
        profile = self._build_profile()
        candidate_frame = self._apply_knowledge_gate(sample_frame, profile)

        content_frame = self._build_content_scores(candidate_frame, profile)
        interactions = self._generate_mock_interactions(candidate_frame, profile)
        collaborative_scores = self._build_collaborative_scores(candidate_frame, interactions)

        candidate_frame = candidate_frame.merge(
            content_frame.loc[:, ["listing_id", "content_score"]],
            on="listing_id",
            how="left",
        )
        candidate_frame = candidate_frame.merge(
            collaborative_scores,
            on="listing_id",
            how="left",
        )
        candidate_frame["content_score"] = candidate_frame["content_score"].fillna(0.0)
        candidate_frame["user_based_score"] = candidate_frame["user_based_score"].fillna(0.0)
        candidate_frame["item_based_score"] = candidate_frame["item_based_score"].fillna(0.0)
        candidate_frame["collaborative_score"] = candidate_frame["collaborative_score"].fillna(0.0)

        knowledge_frame = self._build_knowledge_scores(candidate_frame, profile)
        hybrid_frame = self._build_hybrid_scores(knowledge_frame, profile)

        hybrid_top = self._build_method_frame(
            hybrid_frame.sort_values(by="hybrid_score", ascending=False),
            method_name="hybrid_recommendation",
            score_column="hybrid_score",
        )
        content_top = self._build_method_frame(
            hybrid_frame.sort_values(by="content_score", ascending=False),
            method_name="content_based_reference",
            score_column="content_score",
        )
        collaborative_top = self._build_method_frame(
            hybrid_frame.sort_values(by="collaborative_score", ascending=False),
            method_name="collaborative_reference",
            score_column="collaborative_score",
        )
        knowledge_top = self._build_method_frame(
            hybrid_frame.sort_values(by="knowledge_score", ascending=False),
            method_name="knowledge_based_reference",
            score_column="knowledge_score",
        )

        comparison_frame = pd.concat(
            [hybrid_top, content_top, collaborative_top, knowledge_top],
            ignore_index=True,
        )
        overlap_frame = self._build_overlap_matrix(
            {
                "hybrid_recommendation": hybrid_top,
                "content_based_reference": content_top,
                "collaborative_reference": collaborative_top,
                "knowledge_based_reference": knowledge_top,
            }
        )
        component_frame = self._build_component_contributions(hybrid_top)
        component_summary = self._build_component_summary(hybrid_top)

        recommendations_path = self.config.paths.tables_dir / self.hybrid_config.recommendations_filename
        interactions_path = self.config.paths.tables_dir / self.hybrid_config.interactions_filename
        comparison_path = self.config.paths.tables_dir / self.hybrid_config.comparison_filename
        overlap_path = self.config.paths.tables_dir / self.hybrid_config.overlap_filename
        component_path = self.config.paths.tables_dir / self.hybrid_config.component_table_filename
        component_summary_path = self.config.paths.tables_dir / self.hybrid_config.component_summary_filename

        hybrid_top.to_csv(recommendations_path, index=False, encoding="utf-8-sig")
        interactions.to_csv(interactions_path, index=False, encoding="utf-8-sig")
        comparison_frame.to_csv(comparison_path, index=False, encoding="utf-8-sig")
        overlap_frame.to_csv(overlap_path, index=False, encoding="utf-8-sig")
        component_frame.to_csv(component_path, index=False, encoding="utf-8-sig")
        component_summary.to_csv(component_summary_path, index=False, encoding="utf-8-sig")

        ranking_plot_path = self._plot_ranking_comparison(comparison_frame)
        contribution_plot_path = self._plot_component_contributions(component_summary)

        summary = HybridSummary(
            input_path=str(self.input_path),
            sampled_rows=len(sample_frame),
            candidate_rows=len(candidate_frame),
            interaction_rows=len(interactions),
            top_k=self.hybrid_config.top_k,
            user_profile=asdict(profile),
            weights=self.config.recommendation.as_dict(),
            methods_compared=[
                "hybrid_recommendation",
                "content_based_reference",
                "collaborative_reference",
                "knowledge_based_reference",
            ],
            tables_created=[
                str(recommendations_path),
                str(interactions_path),
                str(comparison_path),
                str(overlap_path),
                str(component_path),
                str(component_summary_path),
            ],
            plots_created=[
                str(ranking_plot_path),
                str(contribution_plot_path),
            ],
        )
        serialize_to_json(
            summary,
            self.config.paths.metadata_dir / self.hybrid_config.summary_filename,
        )
        self.logger.info(
            "Stage 11 completed: sample=%s, candidates=%s, top_k=%s.",
            summary.sampled_rows,
            summary.candidate_rows,
            summary.top_k,
        )
        return summary

    def _load_sample(self) -> pd.DataFrame:
        sampled_chunks: list[pd.DataFrame] = []
        target_sample = self.hybrid_config.sample_size
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
        self.logger.info("Prepared hybrid sample with %s rows.", len(frame))
        return frame

    def _prepare_sample(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        prepared["listing_id"] = np.arange(1, len(prepared) + 1)
        numeric_columns = [
            "Price",
            "Area",
            "Number of rooms",
            "price_per_m2",
            "transport_score",
            "renovation_score",
            "investment_score",
            "floor_comfort_score",
            "liquidity_score",
        ]
        for column in numeric_columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared["Metro station"] = (
            prepared["Metro station"].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
        )
        prepared["Renovation"] = (
            prepared["Renovation"].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
        )
        return prepared.dropna(subset=numeric_columns).reset_index(drop=True)

    def _build_profile(self) -> HybridPreferenceProfile:
        return HybridPreferenceProfile(
            budget=self.hybrid_config.default_budget,
            desired_area=self.hybrid_config.default_area,
            area_tolerance=self.hybrid_config.default_area_tolerance,
            desired_rooms=self.hybrid_config.default_rooms,
            room_tolerance=self.hybrid_config.default_room_tolerance,
            preferred_metro_station=self.hybrid_config.default_metro_station,
            preferred_renovation=self.hybrid_config.default_renovation,
            target_investment_score=self.hybrid_config.default_investment_score,
            target_liquidity_score=self.hybrid_config.default_liquidity_score,
            min_price_per_m2=self.hybrid_config.min_price_per_m2,
            min_investment_score=self.hybrid_config.min_investment_score,
            min_liquidity_score=self.hybrid_config.min_liquidity_score,
            min_transport_score=self.hybrid_config.min_transport_score,
        )

    def _build_content_scores(
        self,
        sample_frame: pd.DataFrame,
        profile: HybridPreferenceProfile,
    ) -> pd.DataFrame:
        weights = self.config.content_based.weighted_similarity_weights
        frame = sample_frame.loc[
            :,
            [
                "listing_id",
                "Price",
                "Area",
                "Number of rooms",
                "Metro station",
                "Renovation",
                "investment_score",
                "liquidity_score",
            ],
        ].copy()
        price_sim = self._bounded_closeness(frame["Price"], profile.budget, scale=max(profile.budget, 1.0))
        area_sim = self._bounded_closeness(frame["Area"], profile.desired_area, scale=max(profile.desired_area, 1.0))
        rooms_sim = self._bounded_closeness(
            frame["Number of rooms"],
            profile.desired_rooms,
            scale=max(profile.desired_rooms, 1.0),
        )
        metro_sim = (frame["Metro station"] == profile.preferred_metro_station).astype(float)
        renovation_sim = (frame["Renovation"] == profile.preferred_renovation).astype(float)
        investment_sim = self._bounded_closeness(
            frame["investment_score"],
            profile.target_investment_score,
            scale=max(profile.target_investment_score, 0.1),
        )
        liquidity_sim = self._bounded_closeness(
            frame["liquidity_score"],
            profile.target_liquidity_score,
            scale=max(profile.target_liquidity_score, 0.1),
        )
        frame["content_score"] = (
            weights["Price"] * price_sim
            + weights["Area"] * area_sim
            + weights["Number of rooms"] * rooms_sim
            + weights["Metro station"] * metro_sim
            + weights["Renovation"] * renovation_sim
            + weights["investment_score"] * investment_sim
            + weights["liquidity_score"] * liquidity_sim
        )
        return frame.loc[:, ["listing_id", "content_score"]]

    def _generate_mock_interactions(
        self,
        sample_frame: pd.DataFrame,
        profile: HybridPreferenceProfile,
    ) -> pd.DataFrame:
        user_profiles = self._build_mock_user_profiles(sample_frame, profile)
        records: list[dict[str, Any]] = []
        rng = np.random.default_rng(self.config.experiments.random_state)
        for user_profile in user_profiles:
            scores = self._interaction_preference_scores(sample_frame, user_profile)
            top_pool_size = min(max(self.hybrid_config.interactions_per_user * 8, 80), len(sample_frame))
            top_pool_indices = np.argsort(scores)[::-1][:top_pool_size]
            top_pool_scores = scores[top_pool_indices]
            probability_vector = top_pool_scores / top_pool_scores.sum() if top_pool_scores.sum() else None
            chosen_indices = rng.choice(
                top_pool_indices,
                size=min(self.hybrid_config.interactions_per_user, len(top_pool_indices)),
                replace=False,
                p=probability_vector,
            )
            chosen_scores = sorted(
                ((int(item_index), float(scores[item_index])) for item_index in chosen_indices),
                key=lambda item: item[1],
                reverse=True,
            )
            for rank_index, (item_index, score_value) in enumerate(chosen_scores, start=1):
                listing = sample_frame.iloc[item_index]
                rating_noise = float(rng.uniform(-0.08, 0.08))
                rating = float(np.clip(3.0 + 2.2 * score_value - 0.015 * (rank_index - 1) + rating_noise, 1.0, 5.0))
                records.append(
                    {
                        "user_id": user_profile["user_id"],
                        "listing_id": int(listing["listing_id"]),
                        "rating": round(rating, 4),
                    }
                )
        return pd.DataFrame(records).sort_values(by=["user_id", "rating"], ascending=[True, False]).reset_index(drop=True)

    def _build_mock_user_profiles(
        self,
        sample_frame: pd.DataFrame,
        profile: HybridPreferenceProfile,
    ) -> list[dict[str, Any]]:
        metro_choices = sample_frame["Metro station"].value_counts().head(20).index.tolist()
        renovation_choices = sample_frame["Renovation"].value_counts().head(4).index.tolist()
        budgets = sample_frame["Price"].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
        areas = sample_frame["Area"].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
        rng = np.random.default_rng(self.config.experiments.random_state)

        profiles: list[dict[str, Any]] = [
            {
                "user_id": "user_0001",
                "budget": profile.budget,
                "desired_area": profile.desired_area,
                "desired_rooms": profile.desired_rooms,
                "preferred_metro_station": profile.preferred_metro_station,
                "preferred_renovation": profile.preferred_renovation,
                "target_investment_score": profile.target_investment_score,
                "target_liquidity_score": profile.target_liquidity_score,
            }
        ]
        for user_index in range(1, self.hybrid_config.n_mock_users):
            budget_anchor = budgets[user_index % len(budgets)]
            area_anchor = areas[(user_index // len(budgets)) % len(areas)]
            room_anchor = float((user_index % 4) + 1)
            profiles.append(
                {
                    "user_id": f"user_{user_index + 1:04d}",
                    "budget": float(max(2_500_000.0, budget_anchor * rng.uniform(0.85, 1.15))),
                    "desired_area": float(max(18.0, area_anchor * rng.uniform(0.8, 1.2))),
                    "desired_rooms": float(min(room_anchor, 4.0)),
                    "preferred_metro_station": str(metro_choices[user_index % len(metro_choices)]),
                    "preferred_renovation": str(renovation_choices[user_index % len(renovation_choices)]),
                    "target_investment_score": float(rng.uniform(0.45, 0.85)),
                    "target_liquidity_score": float(rng.uniform(0.45, 0.85)),
                }
            )
        return profiles

    def _interaction_preference_scores(
        self,
        sample_frame: pd.DataFrame,
        profile: dict[str, Any],
    ) -> np.ndarray:
        price_sim = self._bounded_closeness(sample_frame["Price"], profile["budget"], scale=max(profile["budget"], 1.0))
        area_sim = self._bounded_closeness(sample_frame["Area"], profile["desired_area"], scale=max(profile["desired_area"], 1.0))
        rooms_sim = self._bounded_closeness(
            sample_frame["Number of rooms"],
            profile["desired_rooms"],
            scale=max(profile["desired_rooms"], 1.0),
        )
        metro_sim = (sample_frame["Metro station"] == profile["preferred_metro_station"]).astype(float)
        renovation_sim = (sample_frame["Renovation"] == profile["preferred_renovation"]).astype(float)
        investment_sim = self._bounded_closeness(sample_frame["investment_score"], profile["target_investment_score"], scale=1.0)
        liquidity_sim = self._bounded_closeness(sample_frame["liquidity_score"], profile["target_liquidity_score"], scale=1.0)
        return (
            0.22 * price_sim
            + 0.18 * area_sim
            + 0.14 * rooms_sim
            + 0.15 * metro_sim
            + 0.11 * renovation_sim
            + 0.10 * investment_sim
            + 0.10 * liquidity_sim
        ).to_numpy(dtype=float)

    def _build_collaborative_scores(
        self,
        sample_frame: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> pd.DataFrame:
        matrix = interactions.pivot_table(
            index="user_id",
            columns="listing_id",
            values="rating",
            aggfunc="mean",
            fill_value=0.0,
        )
        target_user_id = "user_0001"

        user_similarity = cosine_similarity(matrix)
        similarity_frame = pd.DataFrame(user_similarity, index=matrix.index, columns=matrix.index)
        user_sim = similarity_frame.loc[target_user_id].drop(index=target_user_id)
        top_neighbors = user_sim.sort_values(ascending=False).head(20)
        weighted_ratings = matrix.loc[top_neighbors.index].T.dot(top_neighbors)
        normalization = top_neighbors.abs().sum()
        user_scores = weighted_ratings / normalization if normalization else weighted_ratings
        seen_items = matrix.loc[target_user_id]
        user_scores = user_scores[seen_items == 0.0]

        item_similarity = cosine_similarity(matrix.T)
        item_similarity_frame = pd.DataFrame(item_similarity, index=matrix.columns, columns=matrix.columns)
        target_vector = matrix.loc[target_user_id]
        seen_items_only = target_vector[target_vector > 0.0]
        item_scores_map: dict[int, float] = {}
        for seen_item_id, rating in seen_items_only.items():
            neighbors = item_similarity_frame.loc[seen_item_id].drop(index=seen_item_id)
            neighbors = neighbors[target_vector[neighbors.index] == 0.0].sort_values(ascending=False).head(40)
            for item_id, similarity_value in neighbors.items():
                item_scores_map[int(item_id)] = item_scores_map.get(int(item_id), 0.0) + float(similarity_value * rating)
        item_scores = pd.Series(item_scores_map, dtype=float)

        score_frame = pd.DataFrame({"listing_id": sample_frame["listing_id"]})
        score_frame["user_based_score"] = score_frame["listing_id"].map(user_scores.to_dict()).fillna(0.0)
        score_frame["item_based_score"] = score_frame["listing_id"].map(item_scores.to_dict()).fillna(0.0)
        score_frame["user_based_score"] = self._normalize_series(score_frame["user_based_score"])
        score_frame["item_based_score"] = self._normalize_series(score_frame["item_based_score"])
        score_frame["collaborative_score"] = 0.5 * (
            score_frame["user_based_score"] + score_frame["item_based_score"]
        )
        return score_frame

    def _apply_knowledge_gate(
        self,
        sample_frame: pd.DataFrame,
        profile: HybridPreferenceProfile,
    ) -> pd.DataFrame:
        mask = (
            (sample_frame["Price"] <= profile.budget)
            & sample_frame["Area"].between(
                profile.desired_area - profile.area_tolerance,
                profile.desired_area + profile.area_tolerance,
            )
            & sample_frame["Number of rooms"].between(
                profile.desired_rooms - profile.room_tolerance,
                profile.desired_rooms + profile.room_tolerance,
            )
            & (sample_frame["price_per_m2"] >= profile.min_price_per_m2)
            & (sample_frame["investment_score"] >= profile.min_investment_score)
            & (sample_frame["liquidity_score"] >= profile.min_liquidity_score)
            & (sample_frame["transport_score"] >= profile.min_transport_score)
        )
        gated = sample_frame.loc[mask].copy().reset_index(drop=True)
        self.logger.info("Knowledge gate retained %s candidate rows.", len(gated))
        return gated

    def _build_knowledge_scores(
        self,
        candidate_frame: pd.DataFrame,
        profile: HybridPreferenceProfile,
    ) -> pd.DataFrame:
        frame = candidate_frame.copy()
        frame["budget_fit"] = self._bounded_closeness(frame["Price"], profile.budget, scale=max(profile.budget, 1.0))
        frame["area_fit"] = self._bounded_closeness(frame["Area"], profile.desired_area, scale=max(profile.area_tolerance, 1.0))
        frame["rooms_fit"] = self._bounded_closeness(
            frame["Number of rooms"],
            profile.desired_rooms,
            scale=max(profile.room_tolerance, 1.0),
        )
        frame["metro_fit"] = np.where(
            frame["Metro station"] == profile.preferred_metro_station,
            1.0,
            np.where(frame["Metro station"] == "Unknown", 0.15, 0.45),
        )
        frame["renovation_fit"] = np.where(
            frame["Renovation"] == profile.preferred_renovation,
            1.0,
            np.where(frame["Renovation"] == "Unknown", 0.20, 0.55),
        )
        frame["knowledge_score"] = (
            0.18 * frame["budget_fit"]
            + 0.16 * frame["area_fit"]
            + 0.12 * frame["rooms_fit"]
            + 0.12 * frame["metro_fit"]
            + 0.10 * frame["renovation_fit"]
            + 0.12 * frame["transport_score"]
            + 0.10 * frame["investment_score"]
            + 0.10 * frame["liquidity_score"]
        )
        return frame

    def _build_hybrid_scores(
        self,
        candidate_frame: pd.DataFrame,
        profile: HybridPreferenceProfile,
    ) -> pd.DataFrame:
        frame = candidate_frame.copy()
        frame["price_component"] = self._normalize_series(
            self._bounded_closeness(frame["Price"], profile.budget, scale=max(profile.budget, 1.0))
        )
        frame["transport_component"] = self._normalize_series(frame["transport_score"])
        frame["renovation_component"] = self._normalize_series(frame["renovation_score"])
        frame["investment_component"] = self._normalize_series(frame["investment_score"])
        frame["floor_component"] = self._normalize_series(frame["floor_comfort_score"])
        frame["liquidity_component"] = self._normalize_series(frame["liquidity_score"])
        frame["similarity_component"] = self._normalize_series(frame["content_score"])
        frame["collaborative_component"] = self._normalize_series(frame["collaborative_score"])

        weights = self.config.recommendation.as_dict()
        frame["hybrid_score"] = (
            weights["price"] * frame["price_component"]
            + weights["transport"] * frame["transport_component"]
            + weights["renovation"] * frame["renovation_component"]
            + weights["investment"] * frame["investment_component"]
            + weights["floor"] * frame["floor_component"]
            + weights["liquidity"] * frame["liquidity_component"]
            + weights["similarity"] * frame["similarity_component"]
            + weights["collaborative"] * frame["collaborative_component"]
        )
        return frame

    def _build_method_frame(
        self,
        ranked_frame: pd.DataFrame,
        method_name: str,
        score_column: str,
    ) -> pd.DataFrame:
        top_frame = ranked_frame.head(self.hybrid_config.top_k).copy().reset_index(drop=True)
        top_frame.insert(1, "rank", np.arange(1, len(top_frame) + 1))
        top_frame["method"] = method_name
        top_frame["score"] = top_frame[score_column]
        top_frame["score_normalized"] = self._normalize_series(top_frame["score"])
        return top_frame

    def _build_overlap_matrix(self, ranking_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        method_sets = {
            method: set(frame["listing_id"].tolist())
            for method, frame in ranking_frames.items()
        }
        methods = list(ranking_frames.keys())
        for method_a in methods:
            row: dict[str, Any] = {"method": method_a}
            for method_b in methods:
                intersection = len(method_sets[method_a] & method_sets[method_b])
                union = len(method_sets[method_a] | method_sets[method_b])
                row[method_b] = round(intersection / union if union else 0.0, 4)
            records.append(row)
        return pd.DataFrame(records)

    def _build_component_contributions(self, hybrid_top: pd.DataFrame) -> pd.DataFrame:
        weights = self.config.recommendation.as_dict()
        frame = hybrid_top.copy()
        frame["price_contribution"] = weights["price"] * frame["price_component"]
        frame["transport_contribution"] = weights["transport"] * frame["transport_component"]
        frame["renovation_contribution"] = weights["renovation"] * frame["renovation_component"]
        frame["investment_contribution"] = weights["investment"] * frame["investment_component"]
        frame["floor_contribution"] = weights["floor"] * frame["floor_component"]
        frame["liquidity_contribution"] = weights["liquidity"] * frame["liquidity_component"]
        frame["similarity_contribution"] = weights["similarity"] * frame["similarity_component"]
        frame["collaborative_contribution"] = weights["collaborative"] * frame["collaborative_component"]
        return frame

    def _build_component_summary(self, hybrid_top: pd.DataFrame) -> pd.DataFrame:
        component_columns = {
            "price_component": "Price",
            "transport_component": "Transport",
            "renovation_component": "Renovation",
            "investment_component": "Investment",
            "floor_component": "Floor",
            "liquidity_component": "Liquidity",
            "similarity_component": "Content similarity",
            "collaborative_component": "Collaborative",
        }
        return pd.DataFrame(
            [
                {
                    "component": label,
                    "mean_score": round(float(hybrid_top[column].mean()), 6),
                }
                for column, label in component_columns.items()
            ]
        ).sort_values(by="mean_score", ascending=False)

    def _plot_ranking_comparison(self, comparison_frame: pd.DataFrame) -> Path:
        plot_frame = comparison_frame.copy()
        plot_frame["method_label"] = plot_frame["method"].map(method_label)
        sns.set_theme(style="whitegrid", context="talk")
        setup_plot_style()
        figure, axis = plt.subplots(figsize=(13.5, 7.5))
        sns.lineplot(
            data=plot_frame,
            x="rank",
            y="score_normalized",
            hue="method_label",
            marker="o",
            linewidth=2.5,
            ax=axis,
        )
        axis.set_title("Hybrid and component recommendation rankings")
        axis.set_xlabel("Rank")
        axis.set_ylabel("Normalized score")
        legend = axis.get_legend()
        if legend is not None:
            sns.move_legend(axis, "upper left", bbox_to_anchor=(1.02, 1.0), title="Method")
        output_path = self.config.paths.plots_dir / self.hybrid_config.ranking_plot_filename
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.hybrid_config.plot_dpi, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def _plot_component_contributions(self, component_summary: pd.DataFrame) -> Path:
        plot_frame = component_summary.sort_values(by="mean_score", ascending=True).copy()
        plot_frame["component_label"] = plot_frame["component"].map(component_label)
        sns.set_theme(style="whitegrid", context="talk")
        setup_plot_style()
        figure, axis = plt.subplots(figsize=(11, 7))
        sns.barplot(
            data=plot_frame,
            x="mean_score",
            y="component_label",
            color="#4c78a8",
            ax=axis,
        )
        axis.set_title("Mean component scores in the hybrid top K")
        axis.set_xlabel("Mean normalized component score")
        axis.set_ylabel("Component")
        output_path = self.config.paths.plots_dir / self.hybrid_config.contribution_plot_filename
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.hybrid_config.plot_dpi, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def _iter_engineered_chunks(self):
        return pd.read_csv(
            self.input_path,
            chunksize=self.hybrid_config.chunk_size,
            low_memory=False,
        )

    def _engineered_row_count_hint(self) -> int:
        summary_path = self.config.paths.metadata_dir / self.config.feature_engineering.summary_filename
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            return int(payload["engineered_rows"])
        return self.hybrid_config.sample_size

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
