from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.config import ModelEvaluationConfig
from src.utils.logger import get_logger
from src.visualization.plot_labels import (
    format_numeric_axis,
    setup_plot_style,
    feature_label,
    model_label,
)


class ModelEvaluationVisualizer:
    """Creates model-evaluation plots for regression diagnostics and comparison."""

    def __init__(self, output_dir: Path, config: ModelEvaluationConfig) -> None:
        self.output_dir = output_dir
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="talk")
        setup_plot_style()

    def plot_predicted_vs_actual(
        self,
        model_name: str,
        plot_frame: pd.DataFrame,
        output_filename: str,
    ) -> Path:
        figure, axis = plt.subplots(figsize=(11.5, 8.5))
        sns.scatterplot(
            data=plot_frame,
            x="actual_price",
            y="predicted_price",
            alpha=0.35,
            s=24,
            color="#1f77b4",
            ax=axis,
        )
        lower_bound = min(plot_frame["actual_price"].min(), plot_frame["predicted_price"].min())
        upper_bound = max(plot_frame["actual_price"].max(), plot_frame["predicted_price"].max())
        axis.plot([lower_bound, upper_bound], [lower_bound, upper_bound], linestyle="--", color="#d62728", linewidth=2)
        axis.set_title(f"Actual vs. predicted price: {model_label(model_name)}")
        axis.set_xlabel("Actual price")
        axis.set_ylabel("Predicted price")
        format_numeric_axis(axis, x=True, y=True, nbins=6)
        return self._save_figure(figure, output_filename)

    def plot_residuals(
        self,
        model_name: str,
        plot_frame: pd.DataFrame,
        output_filename: str,
    ) -> Path:
        figure, axis = plt.subplots(figsize=(11.5, 7.8))
        sns.scatterplot(
            data=plot_frame,
            x="predicted_price",
            y="residual",
            alpha=0.35,
            s=24,
            color="#2ca02c",
            ax=axis,
        )
        axis.axhline(0.0, linestyle="--", color="#d62728", linewidth=2)
        axis.set_title(f"Residual plot: {model_label(model_name)}")
        axis.set_xlabel("Predicted price")
        axis.set_ylabel("Residual")
        format_numeric_axis(axis, x=True, y=True, nbins=6)
        return self._save_figure(figure, output_filename)

    def plot_feature_importance(
        self,
        model_name: str,
        importance_frame: pd.DataFrame,
        output_filename: str,
    ) -> Path:
        ordered = importance_frame.sort_values(by="relative_importance", ascending=True).copy()
        ordered["feature_name_label"] = ordered["feature_name"].map(feature_label)
        figure, axis = plt.subplots(figsize=(12, 8))
        sns.barplot(
            data=ordered,
            x="relative_importance",
            y="feature_name_label",
            color="#ff7f0e",
            ax=axis,
        )
        axis.set_title(f"Feature importance: {model_label(model_name)}")
        axis.set_xlabel("Relative importance")
        axis.set_ylabel("Feature")
        return self._save_figure(figure, output_filename)

    def plot_model_comparison(
        self,
        comparison_frame: pd.DataFrame,
        output_filename: str,
    ) -> Path:
        figure, axes = plt.subplots(2, 2, figsize=(16, 12))
        metric_specs = [
            ("test_mae", "Test MAE", "#4c78a8"),
            ("test_rmse", "Test RMSE", "#f58518"),
            ("test_mse", "Test MSE", "#54a24b"),
            ("test_r2", "Test R²", "#e45756"),
        ]
        plot_frame = comparison_frame.copy()
        plot_frame["model_name_label"] = plot_frame["model_name"].map(model_label)
        for axis, (column, title, color) in zip(axes.flatten(), metric_specs, strict=False):
            sns.barplot(
                data=plot_frame,
                x="model_name_label",
                y=column,
                color=color,
                ax=axis,
            )
            axis.set_title(title)
            axis.set_xlabel("Model")
            axis.tick_params(axis="x", rotation=18)
            format_numeric_axis(axis, y=True, nbins=5)
        return self._save_figure(figure, output_filename)

    def _save_figure(self, figure: plt.Figure, filename: str) -> Path:
        output_path = self.output_dir / filename
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close(figure)
        self.logger.info("Saved plot: %s", output_path.name)
        return output_path
