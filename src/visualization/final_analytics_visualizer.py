from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.config import FinalAnalyticsConfig
from src.utils.logger import get_logger
from src.visualization.plot_labels import (
    format_numeric_axis,
    setup_plot_style,
    method_label,
    model_label,
    wrap_ticklabels,
)


class FinalAnalyticsVisualizer:
    """Creates compact summary plots for the final analytical chapter."""

    def __init__(self, output_dir: Path, config: FinalAnalyticsConfig) -> None:
        self.output_dir = output_dir
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="talk")
        setup_plot_style()

    def plot_price_models(self, comparison_frame: pd.DataFrame, output_filename: str) -> Path:
        ordered = comparison_frame.sort_values(by="test_rmse", ascending=True).copy()
        ordered["model_name_label"] = ordered["model_name"].map(model_label)
        figure, axes = plt.subplots(1, 2, figsize=(17, 6.8))

        sns.barplot(data=ordered, x="model_name_label", y="test_rmse", color="#f58518", ax=axes[0])
        axes[0].set_title("Price models: Test RMSE")
        axes[0].set_xlabel("Model")
        axes[0].set_ylabel("RMSE")
        wrap_ticklabels(axes[0], axis_name="x", width=14, rotation=0, ha="center", fontsize=12)
        format_numeric_axis(axes[0], y=True, nbins=5)

        sns.barplot(data=ordered, x="model_name_label", y="test_r2", color="#4c78a8", ax=axes[1])
        axes[1].set_title("Price models: Test R²")
        axes[1].set_xlabel("Model")
        axes[1].set_ylabel("R²")
        wrap_ticklabels(axes[1], axis_name="x", width=14, rotation=0, ha="center", fontsize=12)

        return self._save_figure(figure, output_filename)

    def plot_recommendation_methods(self, comparison_frame: pd.DataFrame, output_filename: str) -> Path:
        ordered = comparison_frame.sort_values(by="dcg_utility", ascending=False).copy()
        ordered["method_name_label"] = ordered["method_name"].map(method_label)
        figure, axes = plt.subplots(1, 2, figsize=(22, 8.5))

        sns.barplot(data=ordered, x="method_name_label", y="dcg_utility", color="#54a24b", ax=axes[0])
        axes[0].set_title("Recommendation methods:\nDCG utility")
        axes[0].set_xlabel("Method")
        axes[0].set_ylabel("DCG utility")
        wrap_ticklabels(axes[0], axis_name="x", width=14, rotation=18, ha="right", fontsize=10)

        sns.barplot(data=ordered, x="method_name_label", y="hard_constraint_rate", color="#e45756", ax=axes[1])
        axes[1].set_title("Recommendation methods:\nhard-constraint rate")
        axes[1].set_xlabel("Method")
        axes[1].set_ylabel("Hard-constraint rate")
        wrap_ticklabels(axes[1], axis_name="x", width=14, rotation=18, ha="right", fontsize=10)

        return self._save_figure(figure, output_filename)

    def plot_shortlist_discounts(self, shortlist_frame: pd.DataFrame, output_filename: str) -> Path:
        preview = shortlist_frame.head(self.config.shortlist_preview_rows).copy()
        figure, axes = plt.subplots(1, 2, figsize=(15.5, 6.8))

        sns.barplot(data=preview, x="rank", y="relative_discount_to_market", color="#72b7b2", ax=axes[0])
        axes[0].set_title(
            "Top shortlist properties:\nrelative discount to market",
            fontsize=15,
            pad=10,
        )
        axes[0].set_xlabel("Rank")
        axes[0].set_ylabel("Relative discount to market")

        sns.barplot(data=preview, x="rank", y="decision_support_score", color="#b279a2", ax=axes[1])
        axes[1].set_title(
            "Top shortlist properties:\ndecision-support score",
            fontsize=15,
            pad=10,
        )
        axes[1].set_xlabel("Rank")
        axes[1].set_ylabel("Decision-support score")

        figure.subplots_adjust(wspace=0.2, top=0.84)

        return self._save_figure(figure, output_filename)

    def _save_figure(self, figure: plt.Figure, filename: str) -> Path:
        output_path = self.output_dir / filename
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close(figure)
        self.logger.info("Saved plot: %s", output_path.name)
        return output_path
