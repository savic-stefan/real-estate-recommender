from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config.config import AppConfig, ContentBasedConfig
from src.utils.helpers import serialize_to_json
from src.utils.logger import get_logger
from src.visualization.plot_labels import setup_plot_style, method_label


@dataclass(slots=True)
class UserPreferenceProfile:
    """Represents a realistic buyer profile used for content-based recommendation."""

    budget: float
    desired_area: float
    desired_rooms: float
    preferred_metro_station: str
    preferred_renovation: str
    target_investment_score: float
    target_liquidity_score: float


@dataclass(slots=True)
class ContentBasedSummary:
    """Metadata produced by the content-based recommendation stage."""

    input_path: str
    sampled_rows: int
    top_k: int
    user_profile: dict[str, Any]
    methods_compared: list[str]
    tables_created: list[str]
    plots_created: list[str]


class ContentBasedRecommender:
    """Builds and compares several content-based recommendation strategies."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.reco_config: ContentBasedConfig = config.content_based
        self.logger = get_logger(self.__class__.__name__)
        self.input_path = self.config.paths.processed_data_dir / self.reco_config.input_filename

    def execute(self) -> ContentBasedSummary:
        """Run a full content-based recommendation experiment with ranking comparison."""

        if not self.input_path.exists():
            raise FileNotFoundError(f"Engineered dataset was not found at {self.input_path}. Run stage 7 first.")

        sample_frame = self._load_recommendation_sample()
        sample_frame = self._prepare_sample(sample_frame)
        profile = self._build_user_profile(sample_frame)

        vector_bundle = self._build_vector_space(sample_frame, profile)
        ranking_frames = self._rank_with_all_methods(sample_frame, profile, vector_bundle)
        overlap_matrix = self._build_overlap_matrix(ranking_frames)

        top_recommendations = pd.concat(ranking_frames.values(), ignore_index=True)
        recommendations_path = self.config.paths.tables_dir / self.reco_config.top_recommendations_filename
        overlap_path = self.config.paths.tables_dir / self.reco_config.overlap_filename

        top_recommendations.to_csv(recommendations_path, index=False, encoding="utf-8-sig")
        overlap_matrix.to_csv(overlap_path, index=False, encoding="utf-8-sig")

        ranking_plot_path = self._plot_ranking_comparison(ranking_frames)

        summary = ContentBasedSummary(
            input_path=str(self.input_path),
            sampled_rows=len(sample_frame),
            top_k=self.reco_config.top_k,
            user_profile=asdict(profile),
            methods_compared=list(ranking_frames.keys()),
            tables_created=[
                str(recommendations_path),
                str(overlap_path),
            ],
            plots_created=[str(ranking_plot_path)],
        )
        serialize_to_json(
            summary,
            self.config.paths.metadata_dir / self.reco_config.summary_filename,
        )
        self.logger.info(
            "Stage 8 completed: compared %s methods on %s candidate objects.",
            len(summary.methods_compared),
            summary.sampled_rows,
        )
        return summary

    def _load_recommendation_sample(self) -> pd.DataFrame:
        """Load a representative sample from the full engineered dataset."""

        sampled_chunks: list[pd.DataFrame] = []
        total_rows_seen = 0
        target_sample = self.reco_config.sample_size
        sample_fraction = min(1.0, target_sample / max(self._engineered_row_count_hint(), 1))

        for chunk_index, chunk in enumerate(self._iter_engineered_chunks(), start=1):
            total_rows_seen += len(chunk)
            sampled_chunk = chunk.sample(
                frac=sample_fraction,
                random_state=self.config.experiments.random_state + chunk_index,
            )
            if not sampled_chunk.empty:
                sampled_chunks.append(sampled_chunk)

        sample_frame = pd.concat(sampled_chunks, ignore_index=True)
        if len(sample_frame) > target_sample:
            sample_frame = sample_frame.sample(
                n=target_sample,
                random_state=self.config.experiments.random_state,
            ).reset_index(drop=True)
        self.logger.info("Prepared recommendation sample with %s rows.", len(sample_frame))
        return sample_frame

    def _prepare_sample(self, sample_frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize dtypes and create a stable listing identifier."""

        frame = sample_frame.copy()
        frame["listing_id"] = np.arange(1, len(frame) + 1)
        numeric_columns = [
            "Price",
            "Area",
            "Number of rooms",
            "investment_score",
            "liquidity_score",
        ]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame["Metro station"] = frame["Metro station"].fillna("").astype(str).str.strip()
        frame["Renovation"] = frame["Renovation"].fillna("").astype(str).str.strip()
        frame = frame.dropna(subset=numeric_columns).reset_index(drop=True)
        return frame

    def _build_user_profile(self, sample_frame: pd.DataFrame) -> UserPreferenceProfile:
        """Construct a realistic buyer preference profile for ranking experiments."""

        metro_station = self.reco_config.default_metro_station
        if metro_station not in set(sample_frame["Metro station"]):
            metro_station = str(sample_frame["Metro station"].mode().iloc[0])

        renovation = self.reco_config.default_renovation
        if renovation not in set(sample_frame["Renovation"]):
            renovation = str(sample_frame["Renovation"].mode().iloc[0])

        return UserPreferenceProfile(
            budget=self.reco_config.default_budget,
            desired_area=self.reco_config.default_area,
            desired_rooms=self.reco_config.default_rooms,
            preferred_metro_station=metro_station,
            preferred_renovation=renovation,
            target_investment_score=self.reco_config.default_investment_score,
            target_liquidity_score=self.reco_config.default_liquidity_score,
        )

    def _build_vector_space(
        self,
        sample_frame: pd.DataFrame,
        profile: UserPreferenceProfile,
    ) -> dict[str, Any]:
        """Create a joint item-profile vector space for distance-based similarities."""

        numeric_columns = ["Price", "Area", "Number of rooms", "investment_score", "liquidity_score"]
        categorical_columns = ["Metro station", "Renovation"]

        scaler = StandardScaler()
        numeric_matrix = scaler.fit_transform(sample_frame[numeric_columns])
        user_numeric = scaler.transform(
            pd.DataFrame(
                [
                    {
                        "Price": profile.budget,
                        "Area": profile.desired_area,
                        "Number of rooms": profile.desired_rooms,
                        "investment_score": profile.target_investment_score,
                        "liquidity_score": profile.target_liquidity_score,
                    }
                ]
            )
        )

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        categorical_matrix = encoder.fit_transform(sample_frame[categorical_columns])
        user_categorical = encoder.transform(
            pd.DataFrame(
                [
                    {
                        "Metro station": profile.preferred_metro_station,
                        "Renovation": profile.preferred_renovation,
                    }
                ]
            )
        )

        item_matrix = np.hstack([numeric_matrix, categorical_matrix])
        user_vector = np.hstack([user_numeric, user_categorical])
        return {
            "item_matrix": item_matrix,
            "user_vector": user_vector,
        }

    def _rank_with_all_methods(
        self,
        sample_frame: pd.DataFrame,
        profile: UserPreferenceProfile,
        vector_bundle: dict[str, Any],
    ) -> dict[str, pd.DataFrame]:
        """Compute recommendation rankings with all configured similarity methods."""

        item_matrix = vector_bundle["item_matrix"]
        user_vector = vector_bundle["user_vector"]

        cosine_scores = cosine_similarity(item_matrix, user_vector).ravel()
        euclidean_scores = 1.0 / (1.0 + euclidean_distances(item_matrix, user_vector).ravel())
        manhattan_scores = 1.0 / (1.0 + manhattan_distances(item_matrix, user_vector).ravel())
        weighted_scores = self._compute_weighted_similarity(sample_frame, profile)

        return {
            "cosine_similarity": self._build_ranked_frame(sample_frame, cosine_scores, "cosine_similarity"),
            "euclidean_similarity": self._build_ranked_frame(sample_frame, euclidean_scores, "euclidean_similarity"),
            "manhattan_similarity": self._build_ranked_frame(sample_frame, manhattan_scores, "manhattan_similarity"),
            "weighted_similarity": self._build_ranked_frame(sample_frame, weighted_scores, "weighted_similarity"),
        }

    def _compute_weighted_similarity(
        self,
        sample_frame: pd.DataFrame,
        profile: UserPreferenceProfile,
    ) -> np.ndarray:
        """Compute an interpretable weighted similarity tailored to housing choice."""

        weights = self.reco_config.weighted_similarity_weights
        price_sim = self._bounded_closeness(sample_frame["Price"], profile.budget, scale=max(profile.budget, 1.0))
        area_sim = self._bounded_closeness(sample_frame["Area"], profile.desired_area, scale=max(profile.desired_area, 1.0))
        rooms_sim = self._bounded_closeness(sample_frame["Number of rooms"], profile.desired_rooms, scale=max(profile.desired_rooms, 1.0))
        metro_sim = (sample_frame["Metro station"] == profile.preferred_metro_station).astype(float)
        renovation_sim = (sample_frame["Renovation"] == profile.preferred_renovation).astype(float)
        investment_sim = self._bounded_closeness(
            sample_frame["investment_score"],
            profile.target_investment_score,
            scale=max(profile.target_investment_score, 0.1),
        )
        liquidity_sim = self._bounded_closeness(
            sample_frame["liquidity_score"],
            profile.target_liquidity_score,
            scale=max(profile.target_liquidity_score, 0.1),
        )

        weighted_sum = (
            weights["Price"] * price_sim
            + weights["Area"] * area_sim
            + weights["Number of rooms"] * rooms_sim
            + weights["Metro station"] * metro_sim
            + weights["Renovation"] * renovation_sim
            + weights["investment_score"] * investment_sim
            + weights["liquidity_score"] * liquidity_sim
        )
        return weighted_sum.to_numpy(dtype=float)

    def _build_ranked_frame(
        self,
        sample_frame: pd.DataFrame,
        scores: np.ndarray,
        method_name: str,
    ) -> pd.DataFrame:
        """Attach similarity scores and return top-K recommendations."""

        ranked = sample_frame.loc[
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
        ranked["method"] = method_name
        ranked["score"] = scores
        ranked = ranked.sort_values(by="score", ascending=False).head(self.reco_config.top_k).reset_index(drop=True)
        ranked.insert(1, "rank", np.arange(1, len(ranked) + 1))
        return ranked

    def _build_overlap_matrix(self, ranking_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Quantify top-K overlap between different content-based methods."""

        methods = list(ranking_frames.keys())
        records: list[dict[str, Any]] = []
        method_sets = {
            method: set(frame["listing_id"].tolist())
            for method, frame in ranking_frames.items()
        }
        for method_a in methods:
            row: dict[str, Any] = {"method": method_a}
            for method_b in methods:
                intersection = len(method_sets[method_a] & method_sets[method_b])
                union = len(method_sets[method_a] | method_sets[method_b])
                jaccard = intersection / union if union else 0.0
                row[method_b] = round(jaccard, 4)
            records.append(row)
        return pd.DataFrame(records)

    def _plot_ranking_comparison(self, ranking_frames: dict[str, pd.DataFrame]) -> Path:
        """Visualize score decay across ranking positions for all similarity methods."""

        plot_frame = pd.concat(ranking_frames.values(), ignore_index=True)
        plot_frame["method_label"] = plot_frame["method"].map(method_label)
        sns.set_theme(style="whitegrid", context="talk")
        setup_plot_style()
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
        axis.set_title("Content-based recommendation rankings")
        axis.set_xlabel("Rank")
        axis.set_ylabel("Similarity score")
        legend = axis.get_legend()
        if legend is not None:
            sns.move_legend(axis, "center left", bbox_to_anchor=(1.02, 0.5), title="Method")
        output_path = self.config.paths.plots_dir / self.reco_config.ranking_plot_filename
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.reco_config.plot_dpi, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def _iter_engineered_chunks(self):
        return pd.read_csv(
            self.input_path,
            chunksize=self.reco_config.chunk_size,
            low_memory=False,
        )

    def _engineered_row_count_hint(self) -> int:
        summary_path = self.config.paths.metadata_dir / self.config.feature_engineering.summary_filename
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            return int(payload["engineered_rows"])
        return self.reco_config.sample_size

    @staticmethod
    def _bounded_closeness(values: pd.Series, target: float, scale: float) -> pd.Series:
        return (1.0 / (1.0 + (values - target).abs() / max(scale, 1e-9))).clip(0.0, 1.0)
