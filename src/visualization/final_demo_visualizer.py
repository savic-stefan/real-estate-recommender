from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.config import FinalDemoConfig
from src.utils.logger import get_logger
from src.visualization.plot_labels import (
    setup_plot_style,
    price_type_label,
)


class FinalDemoVisualizer:
    """Creates compact visuals for the final end-to-end demo package."""

    def __init__(self, output_dir: Path, config: FinalDemoConfig) -> None:
        self.output_dir = output_dir
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="talk")
        setup_plot_style()

    def plot_score_overview(self, shortlist_frame: pd.DataFrame, output_filename: str) -> Path:
        figure, axis = plt.subplots(figsize=(12, 7))
        sns.barplot(
            data=shortlist_frame,
            x="rank",
            y="decision_support_score",
            color="#4c78a8",
            ax=axis,
        )
        axis.set_title("Decision-support score by shortlist rank")
        axis.set_xlabel("Rank")
        axis.set_ylabel("Decision-support score")
        axis.ticklabel_format(style="plain", axis="y")
        return self._save_figure(figure, output_filename)

    def plot_price_vs_market(self, shortlist_frame: pd.DataFrame, output_filename: str) -> Path:
        plot_frame = shortlist_frame.melt(
            id_vars=["rank"],
            value_vars=["Price", "predicted_market_price"],
            var_name="price_type",
            value_name="value",
        )
        plot_frame["price_type_label"] = plot_frame["price_type"].map(price_type_label)
        figure, axis = plt.subplots(figsize=(12, 7))
        sns.lineplot(
            data=plot_frame,
            x="rank",
            y="value",
            hue="price_type_label",
            marker="o",
            linewidth=2.2,
            ax=axis,
        )
        axis.set_title("Asking price vs. predicted market price")
        axis.set_xlabel("Rank")
        axis.set_ylabel("Price")
        axis.ticklabel_format(style="plain", axis="y")
        legend = axis.get_legend()
        if legend is not None:
            legend.set_title("Price type")
        return self._save_figure(figure, output_filename)

    def _save_figure(self, figure: plt.Figure, filename: str) -> Path:
        output_path = self.output_dir / filename
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close(figure)
        self.logger.info("Saved plot: %s", output_path.name)
        return output_path
