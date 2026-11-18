from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config.config import AppConfig, CollaborativeFilteringConfig
from src.recommendation.content_based import UserPreferenceProfile
from src.utils.helpers import serialize_to_json
from src.utils.logger import get_logger
from src.visualization.plot_labels import setup_plot_style, method_label


@dataclass(slots=True)
class CollaborativeSummary:
    """Metadata produced by the collaborative filtering stage."""

    input_path: str
    sampled_rows: int
    n_mock_users: int
    interactions_count: int
    target_user_id: str
    methods_compared: list[str]
    tables_created: list[str]
    plots_created: list[str]


class CollaborativeFilteringRecommender:
    """Implements user-based and item-based collaborative filtering with mock interactions."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cf_config: CollaborativeFilteringConfig = config.collaborative
        self.logger = get_logger(self.__class__.__name__)
        self.input_path = self.config.paths.processed_data_dir / self.cf_config.input_filename

    def execute(self) -> CollaborativeSummary:
        """Run collaborative filtering and compare it against content-based recommendations."""

        if not self.input_path.exists():
            raise FileNotFoundError(f"Engineered dataset was not found at {self.input_path}. Run stage 7 first.")

        item_frame = self._load_item_sample()
        item_frame = self._prepare_items(item_frame)
        user_profiles = self._build_mock_user_profiles(item_frame)
        interactions = self._generate_mock_interactions(item_frame, user_profiles)
        interaction_matrix = self._build_interaction_matrix(interactions)

        target_user_id = self.cf_config.default_target_user_id
        if target_user_id not in set(interaction_matrix.index):
            target_user_id = str(interaction_matrix.index[0])

        user_based = self._user_based_recommendations(item_frame, interaction_matrix, target_user_id)
        item_based = self._item_based_recommendations(item_frame, interaction_matrix, target_user_id)
        content_based = self._load_content_based_reference()
        comparison_frame = self._build_comparison_frame(user_based, item_based, content_based)

        interactions_path = self.config.paths.tables_dir / self.cf_config.mock_interactions_filename
        recommendations_path = self.config.paths.tables_dir / self.cf_config.recommendations_filename
        comparison_path = self.config.paths.tables_dir / self.cf_config.comparison_filename

        interactions.to_csv(interactions_path, index=False, encoding="utf-8-sig")
        pd.concat([user_based, item_based], ignore_index=True).to_csv(recommendations_path, index=False, encoding="utf-8-sig")
        comparison_frame.to_csv(comparison_path, index=False, encoding="utf-8-sig")

        ranking_plot_path = self._plot_comparison(comparison_frame)

        summary = CollaborativeSummary(
            input_path=str(self.input_path),
            sampled_rows=len(item_frame),
            n_mock_users=len(user_profiles),
            interactions_count=len(interactions),
            target_user_id=target_user_id,
            methods_compared=["user_based_cf", "item_based_cf", "content_based_reference"],
            tables_created=[
                str(interactions_path),
                str(recommendations_path),
                str(comparison_path),
            ],
            plots_created=[str(ranking_plot_path)],
        )
        serialize_to_json(
            summary,
            self.config.paths.metadata_dir / self.cf_config.summary_filename,
        )
        self.logger.info(
            "Stage 9 completed: generated %s interactions for %s users.",
            summary.interactions_count,
            summary.n_mock_users,
        )
        return summary

    def _load_item_sample(self) -> pd.DataFrame:
        sampled_chunks: list[pd.DataFrame] = []
        target_sample = self.cf_config.sample_size
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
        self.logger.info("Prepared collaborative sample with %s items.", len(frame))
        return frame

    def _prepare_items(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        prepared["listing_id"] = np.arange(1, len(prepared) + 1)
        numeric_columns = ["Price", "Area", "Number of rooms", "investment_score", "liquidity_score"]
        for column in numeric_columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared["Metro station"] = prepared["Metro station"].fillna("").astype(str).str.strip()
        prepared["Renovation"] = prepared["Renovation"].fillna("").astype(str).str.strip()
        prepared = prepared.dropna(subset=numeric_columns).reset_index(drop=True)
        return prepared

    def _build_mock_user_profiles(self, item_frame: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate realistic synthetic buyer profiles around real market distributions."""

        metro_choices = item_frame["Metro station"].replace("", "Unknown").value_counts().head(20).index.tolist()
        renovation_choices = item_frame["Renovation"].replace("", "Unknown").value_counts().head(4).index.tolist()
        budgets = item_frame["Price"].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
        areas = item_frame["Area"].quantile([0.2, 0.4, 0.6, 0.8]).tolist()

        rng = np.random.default_rng(self.config.experiments.random_state)
        profiles: list[dict[str, Any]] = []
        for user_index in range(self.cf_config.n_mock_users):
            budget_anchor = budgets[user_index % len(budgets)]
            area_anchor = areas[(user_index // len(budgets)) % len(areas)]
            room_anchor = float((user_index % 4) + 1)
            profile = {
                "user_id": f"user_{user_index + 1:04d}",
                "budget": float(max(2_500_000.0, budget_anchor * rng.uniform(0.85, 1.15))),
                "desired_area": float(max(18.0, area_anchor * rng.uniform(0.8, 1.2))),
                "desired_rooms": float(room_anchor if room_anchor <= 4 else 4.0),
                "preferred_metro_station": str(metro_choices[user_index % len(metro_choices)]),
                "preferred_renovation": str(renovation_choices[user_index % len(renovation_choices)]),
                "target_investment_score": float(rng.uniform(0.45, 0.85)),
                "target_liquidity_score": float(rng.uniform(0.45, 0.85)),
            }
            profiles.append(profile)
        return profiles

    def _generate_mock_interactions(
        self,
        item_frame: pd.DataFrame,
        user_profiles: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Generate synthetic but preference-driven explicit ratings."""

        interaction_records: list[dict[str, Any]] = []
        for profile in user_profiles:
            scores = self._weighted_similarity_scores(item_frame, profile)
            top_indices = np.argsort(scores)[::-1][: self.cf_config.interactions_per_user]
            for rank_index, item_index in enumerate(top_indices, start=1):
                listing = item_frame.iloc[item_index]
                rating = float(np.clip(3.0 + 2.2 * scores[item_index] - 0.015 * (rank_index - 1), 1.0, 5.0))
                interaction_records.append(
                    {
                        "user_id": profile["user_id"],
                        "listing_id": int(listing["listing_id"]),
                        "rating": round(rating, 4),
                        "score_signal": round(float(scores[item_index]), 6),
                    }
                )
        interactions = pd.DataFrame(interaction_records)
        return interactions.sort_values(by=["user_id", "rating"], ascending=[True, False]).reset_index(drop=True)

    def _weighted_similarity_scores(self, item_frame: pd.DataFrame, profile: dict[str, Any]) -> np.ndarray:
        price_sim = self._bounded_closeness(item_frame["Price"], profile["budget"], scale=max(profile["budget"], 1.0))
        area_sim = self._bounded_closeness(item_frame["Area"], profile["desired_area"], scale=max(profile["desired_area"], 1.0))
        rooms_sim = self._bounded_closeness(item_frame["Number of rooms"], profile["desired_rooms"], scale=max(profile["desired_rooms"], 1.0))
        metro_sim = (item_frame["Metro station"] == profile["preferred_metro_station"]).astype(float)
        renovation_sim = (item_frame["Renovation"] == profile["preferred_renovation"]).astype(float)
        invest_sim = self._bounded_closeness(item_frame["investment_score"], profile["target_investment_score"], scale=1.0)
        liquidity_sim = self._bounded_closeness(item_frame["liquidity_score"], profile["target_liquidity_score"], scale=1.0)

        return (
            0.22 * price_sim
            + 0.18 * area_sim
            + 0.14 * rooms_sim
            + 0.15 * metro_sim
            + 0.11 * renovation_sim
            + 0.10 * invest_sim
            + 0.10 * liquidity_sim
        ).to_numpy(dtype=float)

    def _build_interaction_matrix(self, interactions: pd.DataFrame) -> pd.DataFrame:
        matrix = interactions.pivot_table(
            index="user_id",
            columns="listing_id",
            values="rating",
            aggfunc="mean",
            fill_value=0.0,
        )
        return matrix

    def _user_based_recommendations(
        self,
        item_frame: pd.DataFrame,
        interaction_matrix: pd.DataFrame,
        target_user_id: str,
    ) -> pd.DataFrame:
        similarity_matrix = cosine_similarity(interaction_matrix)
        similarity_frame = pd.DataFrame(
            similarity_matrix,
            index=interaction_matrix.index,
            columns=interaction_matrix.index,
        )
        user_sim = similarity_frame.loc[target_user_id].drop(index=target_user_id)
        top_neighbors = user_sim.sort_values(ascending=False).head(20)

        weighted_ratings = interaction_matrix.loc[top_neighbors.index].T.dot(top_neighbors)
        normalization = top_neighbors.abs().sum()
        candidate_scores = weighted_ratings / normalization if normalization else weighted_ratings
        seen_items = interaction_matrix.loc[target_user_id]
        candidate_scores = candidate_scores[seen_items == 0.0].sort_values(ascending=False).head(self.cf_config.top_k)

        return self._assemble_recommendation_frame(
            item_frame=item_frame,
            score_series=candidate_scores,
            method_name="user_based_cf",
        )

    def _item_based_recommendations(
        self,
        item_frame: pd.DataFrame,
        interaction_matrix: pd.DataFrame,
        target_user_id: str,
    ) -> pd.DataFrame:
        item_similarity = cosine_similarity(interaction_matrix.T)
        item_similarity_frame = pd.DataFrame(
            item_similarity,
            index=interaction_matrix.columns,
            columns=interaction_matrix.columns,
        )
        user_vector = interaction_matrix.loc[target_user_id]
        seen_items = user_vector[user_vector > 0.0]

        candidate_scores: dict[int, float] = {}
        for seen_item_id, rating in seen_items.items():
            neighbors = item_similarity_frame.loc[seen_item_id].drop(index=seen_item_id)
            neighbors = neighbors[user_vector[neighbors.index] == 0.0].sort_values(ascending=False).head(40)
            for item_id, sim_value in neighbors.items():
                candidate_scores[int(item_id)] = candidate_scores.get(int(item_id), 0.0) + float(sim_value * rating)

        score_series = pd.Series(candidate_scores).sort_values(ascending=False).head(self.cf_config.top_k)
        return self._assemble_recommendation_frame(
            item_frame=item_frame,
            score_series=score_series,
            method_name="item_based_cf",
        )

    def _assemble_recommendation_frame(
        self,
        item_frame: pd.DataFrame,
        score_series: pd.Series,
        method_name: str,
    ) -> pd.DataFrame:
        recommendations = item_frame.loc[
            item_frame["listing_id"].isin(score_series.index),
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
        recommendations["score"] = recommendations["listing_id"].map(score_series.to_dict())
        recommendations["method"] = method_name
        recommendations = recommendations.sort_values(by="score", ascending=False).reset_index(drop=True)
        recommendations.insert(1, "rank", np.arange(1, len(recommendations) + 1))
        return recommendations

    def _load_content_based_reference(self) -> pd.DataFrame:
        content_path = self.config.paths.tables_dir / self.config.content_based.top_recommendations_filename
        if not content_path.exists():
            raise FileNotFoundError("Content-based recommendations are missing. Run stage 8 first.")
        frame = pd.read_csv(content_path)
        reference = frame.loc[frame["method"] == "weighted_similarity"].copy()
        reference["method"] = "content_based_reference"
        return reference.reset_index(drop=True)

    def _build_comparison_frame(
        self,
        user_based: pd.DataFrame,
        item_based: pd.DataFrame,
        content_based: pd.DataFrame,
    ) -> pd.DataFrame:
        comparison = pd.concat([user_based, item_based, content_based], ignore_index=True)
        comparison["score_rank_norm"] = comparison.groupby("method")["score"].rank(method="first", ascending=False, pct=True)
        return comparison

    def _plot_comparison(self, comparison_frame: pd.DataFrame) -> Path:
        sns.set_theme(style="whitegrid", context="talk")
        setup_plot_style()
        plot_frame = comparison_frame.copy()
        plot_frame["method_label"] = plot_frame["method"].map(method_label)
        figure, axis = plt.subplots(figsize=(13.5, 7.5))
        sns.lineplot(
            data=plot_frame,
            x="rank",
            y="score",
            hue="method_label",
            marker="o",
            linewidth=2.5,
            ax=axis,
        )
        axis.set_title("Collaborative and content-based recommendations")
        axis.set_xlabel("Rank")
        axis.set_ylabel("Recommendation score")
        legend = axis.get_legend()
        if legend is not None:
            sns.move_legend(axis, "center left", bbox_to_anchor=(1.02, 0.5), title="Method")
        output_path = self.config.paths.plots_dir / self.cf_config.ranking_plot_filename
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.cf_config.plot_dpi, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def _iter_engineered_chunks(self):
        return pd.read_csv(
            self.input_path,
            chunksize=self.cf_config.chunk_size,
            low_memory=False,
        )

    def _engineered_row_count_hint(self) -> int:
        summary_path = self.config.paths.metadata_dir / self.config.feature_engineering.summary_filename
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            return int(payload["engineered_rows"])
        return self.cf_config.sample_size

    @staticmethod
    def _bounded_closeness(values: pd.Series, target: float, scale: float) -> pd.Series:
        return (1.0 / (1.0 + (values - target).abs() / max(scale, 1e-9))).clip(0.0, 1.0)
