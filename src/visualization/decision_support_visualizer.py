from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.config import DecisionSupportConfig
from src.utils.logger import get_logger
from src.visualization.plot_labels import (
    setup_plot_style,
    component_label,
)


class DecisionSupportVisualizer:
    """Creates summary plots for the integrated decision-support shortlist."""

    def __init__(self, output_dir: Path, config: DecisionSupportConfig) -> None:
        self.output_dir = output_dir
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="talk")
        setup_plot_style()

    def plot_value_gap(self, shortlist_frame: pd.DataFrame, output_filename: str) -> Path:
        figure, axis = plt.subplots(figsize=(12, 7))
        sns.barplot(
            data=shortlist_frame.sort_values(by="decision_support_score", ascending=False),
            x="rank",
            y="relative_discount_to_market",
            color="#4c78a8",
            ax=axis,
        )
        axis.set_title("Discount relative to predicted market price")
        axis.set_xlabel("Recommendation rank")
        axis.set_ylabel("Relative discount to market")
        axis.ticklabel_format(style="plain", axis="y")
        return self._save_figure(figure, output_filename)

    def plot_component_scores(self, shortlist_frame: pd.DataFrame, output_filename: str) -> Path:
        plot_frame = shortlist_frame.loc[
            :,
            [
                "rank",
                "preference_score",
                "valuation_score",
                "investment_score_component",
                "liquidity_score_component",
            ],
        ].melt(
            id_vars="rank",
            var_name="component",
            value_name="score",
        )
        plot_frame["component_label"] = plot_frame["component"].map(component_label)
        figure, axis = plt.subplots(figsize=(14, 7.5))
        sns.lineplot(
            data=plot_frame,
            x="rank",
            y="score",
            hue="component_label",
            marker="o",
            linewidth=2.2,
            ax=axis,
        )
        axis.set_title("Decision-support scores for top-ranked properties")
        axis.set_xlabel("Recommendation rank")
        axis.set_ylabel("Score")
        legend = axis.get_legend()
        if legend is not None:
            sns.move_legend(axis, "center left", bbox_to_anchor=(1.02, 0.5), title="Component")
        return self._save_figure(figure, output_filename)

    def _save_figure(self, figure: plt.Figure, filename: str) -> Path:
        output_path = self.output_dir / filename
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close(figure)
        self.logger.info("Saved plot: %s", output_path.name)
        return output_path
