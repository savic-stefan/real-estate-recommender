from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.config import RecommendationEvaluationConfig
from src.utils.logger import get_logger
from src.visualization.plot_labels import (
    format_numeric_axis,
    setup_plot_style,
    method_label,
    metric_label,
    wrap_ticklabels,
)


class RecommendationEvaluationVisualizer:
    """Creates visual comparisons for recommendation-evaluation experiments."""

    def __init__(self, output_dir: Path, config: RecommendationEvaluationConfig) -> None:
        self.output_dir = output_dir
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="talk")
        setup_plot_style()

    def plot_overall_scores(
        self,
        comparison_frame: pd.DataFrame,
        output_filename: str,
    ) -> Path:
        plot_frame = comparison_frame.sort_values(by="dcg_utility", ascending=False).copy()
        plot_frame["method_name_label"] = plot_frame["method_name"].map(method_label)
        figure, axis = plt.subplots(figsize=(14.5, 8))
        sns.barplot(
            data=plot_frame,
            x="method_name_label",
            y="dcg_utility",
            color="#4c78a8",
            ax=axis,
        )
        axis.set_title("Recommendation methods by DCG utility")
        axis.set_xlabel("Method")
        axis.set_ylabel("DCG utility")
        wrap_ticklabels(axis, axis_name="x", width=18, rotation=0, ha="center", fontsize=12)
        format_numeric_axis(axis, y=True, nbins=5)
        return self._save_figure(figure, output_filename)

    def plot_metric_heatmap(
        self,
        comparison_frame: pd.DataFrame,
        output_filename: str,
    ) -> Path:
        metric_columns = [
            "avg_item_utility",
            "top1_utility",
            "dcg_utility",
            "hard_constraint_rate",
            "metro_match_rate",
            "renovation_match_rate",
            "known_metro_rate",
            "known_renovation_rate",
        ]
        heatmap_frame = comparison_frame.set_index("method_name").loc[:, metric_columns].copy()
        heatmap_frame.index = [method_label(name) for name in heatmap_frame.index]
        heatmap_frame.columns = [metric_label(name) for name in heatmap_frame.columns]
        figure, axis = plt.subplots(figsize=(15.5, 8.5))
        sns.heatmap(
            heatmap_frame,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            linewidths=0.5,
            cbar_kws={"shrink": 0.85},
            annot_kws={"size": 11},
            ax=axis,
        )
        axis.set_title("Recommendation metric heatmap")
        axis.set_xlabel("Metric")
        axis.set_ylabel("Method")
        axis.tick_params(axis="x", labelrotation=90)
        axis.tick_params(axis="y", labelrotation=0)
        return self._save_figure(figure, output_filename)

    def _save_figure(self, figure: plt.Figure, filename: str) -> Path:
        output_path = self.output_dir / filename
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close(figure)
        self.logger.info("Saved plot: %s", output_path.name)
        return output_path
