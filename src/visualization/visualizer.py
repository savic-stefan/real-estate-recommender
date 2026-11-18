from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.config import EDAConfig
from src.utils.logger import get_logger
from src.visualization.plot_labels import (
    format_numeric_axis,
    setup_plot_style,
    column_label,
    wrap_ticklabels,
)


class RealEstateVisualizer:
    """Creates publication-ready EDA visualizations for the diploma project."""

    def __init__(self, output_dir: Path, config: EDAConfig) -> None:
        self.output_dir = output_dir
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="talk")
        setup_plot_style()

    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> Path:
        plot_matrix = correlation_matrix.copy()
        plot_matrix.index = [column_label(name) for name in plot_matrix.index]
        plot_matrix.columns = [column_label(name) for name in plot_matrix.columns]
        figure, axis = plt.subplots(figsize=(12, 9))
        sns.heatmap(
            plot_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.85},
            ax=axis,
        )
        axis.set_title("Correlation matrix of numeric property features")
        return self._save_figure(figure, "correlation_heatmap.png")

    def plot_price_histogram(self, sample_frame: pd.DataFrame) -> Path:
        plot_frame = sample_frame.dropna(subset=["Price"]).copy()
        figure, axis = plt.subplots(figsize=(12, 7))
        sns.histplot(
            plot_frame["Price"],
            bins=self.config.histogram_bins,
            kde=True,
            color="#1f77b4",
            ax=axis,
        )
        axis.set_title("Property price distribution")
        axis.set_xlabel("Price")
        axis.set_ylabel("Count")
        format_numeric_axis(axis, x=True, nbins=6)
        return self._save_figure(figure, "price_histogram.png")

    def plot_price_boxplot(self, sample_frame: pd.DataFrame) -> Path:
        plot_frame = sample_frame.dropna(subset=["Price"]).copy()
        figure, axis = plt.subplots(figsize=(12, 3.5))
        sns.boxplot(x=plot_frame["Price"], color="#2ca02c", ax=axis)
        axis.set_title("Property price range")
        axis.set_xlabel("Price")
        format_numeric_axis(axis, x=True, nbins=6)
        return self._save_figure(figure, "price_boxplot.png")

    def plot_price_vs_area(self, sample_frame: pd.DataFrame) -> Path:
        plot_frame = sample_frame.dropna(subset=["Price", "Area"]).copy()
        figure, axis = plt.subplots(figsize=(12, 7))
        sns.scatterplot(
            data=plot_frame,
            x="Area",
            y="Price",
            hue="Source dataset",
            alpha=0.35,
            s=22,
            ax=axis,
        )
        axis.set_title("Price vs. area")
        axis.set_xlabel("Area")
        axis.set_ylabel("Price")
        legend = axis.get_legend()
        if legend is not None:
            legend.set_title("Source dataset")
        format_numeric_axis(axis, y=True, nbins=6)
        return self._save_figure(figure, "price_vs_area.png")

    def plot_price_vs_metro(self, metro_frame: pd.DataFrame) -> Path:
        figure, axis = plt.subplots(figsize=(16, 9))
        sns.boxplot(
            data=metro_frame,
            x="Metro station",
            y="Price",
            color="#ff7f0e",
            ax=axis,
        )
        axis.set_title("Prices by metro station")
        axis.set_xlabel("Metro station")
        axis.set_ylabel("Price")
        wrap_ticklabels(axis, axis_name="x", width=12, rotation=35, ha="right", fontsize=12)
        format_numeric_axis(axis, y=True, nbins=6)
        return self._save_figure(figure, "price_vs_metro.png")

    def plot_price_vs_rooms(self, sample_frame: pd.DataFrame) -> Path:
        plot_frame = sample_frame.dropna(subset=["Price", "Number of rooms"]).copy()
        plot_frame["Rooms category"] = plot_frame["Number of rooms"].astype("Int64").astype(str) + " rooms"
        figure, axis = plt.subplots(figsize=(14, 8))
        sns.boxplot(
            data=plot_frame,
            x="Rooms category",
            y="Price",
            color="#d62728",
            ax=axis,
        )
        axis.set_title("Prices by number of rooms")
        axis.set_xlabel("Number of rooms")
        axis.set_ylabel("Price")
        wrap_ticklabels(axis, axis_name="x", width=8, rotation=20, ha="right", fontsize=12)
        format_numeric_axis(axis, y=True, nbins=6)
        return self._save_figure(figure, "price_vs_rooms.png")

    def plot_average_price_by_source(self, source_price_frame: pd.DataFrame) -> Path:
        figure, axis = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=source_price_frame,
            x="Source dataset",
            y="mean_price",
            color="#4c78a8",
            ax=axis,
        )
        axis.set_title("Average price by source dataset")
        axis.set_xlabel("Dataset")
        axis.set_ylabel("Average price")
        format_numeric_axis(axis, y=True, nbins=5)
        return self._save_figure(figure, "average_price_by_dataset.png")

    def plot_region_distribution(self, region_frame: pd.DataFrame) -> Path:
        plot_frame = region_frame.copy()
        plot_frame["Region"] = plot_frame["Region"].replace({"Unknown region": "Unknown region"})
        figure, axis = plt.subplots(figsize=(15, 8.5))
        sns.barplot(
            data=plot_frame,
            x="listing_count",
            y="Region",
            color="#72b7b2",
            ax=axis,
        )
        axis.set_title("Regions with the most listings")
        axis.set_xlabel("Number of listings")
        axis.set_ylabel("Region")
        format_numeric_axis(axis, x=True, nbins=6)
        return self._save_figure(figure, "region_distribution.png")

    def plot_custom_histogram(
        self,
        frame: pd.DataFrame,
        value_column: str,
        title: str,
        output_filename: str,
    ) -> Path:
        figure, axis = plt.subplots(figsize=(12, 7))
        sns.histplot(
            frame[value_column].dropna(),
            bins=self.config.histogram_bins,
            kde=True,
            color="#1f77b4",
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel(column_label(value_column))
        axis.set_ylabel("Count")
        format_numeric_axis(axis, x=True, nbins=6)
        return self._save_figure(figure, output_filename)

    def plot_log_price_distribution(
        self,
        frame: pd.DataFrame,
        output_filename: str,
    ) -> Path:
        plot_frame = frame.dropna(subset=["Price"]).copy()
        plot_frame = plot_frame.loc[plot_frame["Price"] > 0].copy()

        figure, axis = plt.subplots(figsize=(12, 7))
        min_price = float(plot_frame["Price"].min())
        max_price = float(plot_frame["Price"].max())
        log_bins = np.geomspace(min_price, max_price, self.config.histogram_bins)

        sns.histplot(
            plot_frame["Price"],
            bins=log_bins,
            color="#2f6ea5",
            edgecolor="white",
            linewidth=0.5,
            ax=axis,
        )

        q1, median, q3 = plot_frame["Price"].quantile([0.25, 0.5, 0.75]).tolist()
        axis.axvline(q1, color="#2e8b57", linestyle="--", linewidth=1.5, label="Lower quartile")
        axis.axvline(median, color="#c44e52", linestyle="-", linewidth=2.0, label="Median")
        axis.axvline(q3, color="#2e8b57", linestyle="--", linewidth=1.5, label="Upper quartile")

        axis.set_xscale("log")
        axis.set_title("Property prices after preprocessing")
        axis.set_xlabel("Property price")
        axis.set_ylabel("Number of properties")
        axis.legend(frameon=True)
        axis.grid(axis="y", alpha=0.3)
        axis.set_axisbelow(True)

        tick_values = [1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000, 300_000_000]
        tick_values = [value for value in tick_values if min_price <= value <= max_price]
        if tick_values:
            axis.set_xticks(tick_values)
            axis.set_xticklabels(
                [f"{int(value / 1_000_000)}M RUB" for value in tick_values],
                rotation=0,
            )

        return self._save_figure(figure, output_filename)

    def plot_before_after_log_price_comparison(
        self,
        raw_frame: pd.DataFrame,
        clean_frame: pd.DataFrame,
        output_filename: str,
    ) -> Path:
        raw_plot = raw_frame.dropna(subset=["Price"]).copy()
        raw_plot = raw_plot.loc[raw_plot["Price"] > 0].copy()
        clean_plot = clean_frame.dropna(subset=["Price"]).copy()
        clean_plot = clean_plot.loc[clean_plot["Price"] > 0].copy()

        global_min = float(min(raw_plot["Price"].min(), clean_plot["Price"].min()))
        global_max = float(max(raw_plot["Price"].max(), clean_plot["Price"].max()))
        log_bins = np.geomspace(global_min, global_max, self.config.histogram_bins)
        tick_values = [1_000_000, 10_000_000, 100_000_000]
        tick_values = [value for value in tick_values if global_min <= value <= global_max]

        figure, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
        plot_specs = (
            (axes[0], raw_plot, "Before cleaning", "#9c755f"),
            (axes[1], clean_plot, "After cleaning", "#4c78a8"),
        )

        for axis, plot_frame, title, color in plot_specs:
            sns.histplot(
                plot_frame["Price"],
                bins=log_bins,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                ax=axis,
            )
            median = float(plot_frame["Price"].median())
            axis.axvline(median, color="#c44e52", linestyle="-", linewidth=2.0, label="Median")
            axis.set_xscale("log")
            axis.set_title(title)
            axis.set_xlabel("Property price")
            axis.grid(axis="y", alpha=0.3)
            axis.set_axisbelow(True)
            if tick_values:
                axis.set_xticks(tick_values)
                axis.set_xticklabels(
                    [f"{int(value / 1_000_000)}M RUB" for value in tick_values],
                    rotation=15,
                    ha="right",
                )
                axis.tick_params(axis="x", labelsize=16)

        axes[0].set_ylabel("Number of properties")
        axes[1].set_ylabel("")
        axes[1].legend(frameon=True)
        figure.suptitle("Property price distribution before and after cleaning", fontsize=22)
        return self._save_figure(figure, output_filename)

    def plot_price_quantile_comparison(
        self,
        raw_frame: pd.DataFrame,
        clean_frame: pd.DataFrame,
        output_filename: str,
    ) -> Path:
        def _prepare_prices(frame: pd.DataFrame) -> pd.Series:
            prepared = frame.dropna(subset=["Price"]).copy()
            prepared = prepared.loc[prepared["Price"] > 0].copy()
            return prepared["Price"]

        def _summary(prices: pd.Series) -> dict[str, float]:
            return {
                "p10": float(prices.quantile(0.10)),
                "q1": float(prices.quantile(0.25)),
                "median": float(prices.quantile(0.50)),
                "q3": float(prices.quantile(0.75)),
                "p90": float(prices.quantile(0.90)),
            }

        raw_prices = _prepare_prices(raw_frame)
        clean_prices = _prepare_prices(clean_frame)
        summaries = [
            ("Before cleaning", _summary(raw_prices), "#8c6d5a"),
            ("After cleaning", _summary(clean_prices), "#4c78a8"),
        ]

        figure, axis = plt.subplots(figsize=(10, 7))
        x_positions = np.arange(len(summaries))

        for x, (label, stats, color) in zip(x_positions, summaries):
            axis.vlines(x, stats["p10"], stats["p90"], color="#6f6f6f", linewidth=3, zorder=1)
            axis.vlines(x, stats["q1"], stats["q3"], color=color, linewidth=16, alpha=0.9, zorder=2)
            axis.scatter(x, stats["median"], color="#c44e52", s=110, zorder=3)
            axis.text(
                x + 0.06,
                stats["median"] * 1.04,
                f"{stats['median'] / 1_000_000:.1f}M RUB",
                fontsize=14,
                color="#444444",
            )

        axis.set_title("Price range before and after cleaning")
        axis.set_xticks(x_positions)
        axis.set_xticklabels([item[0] for item in summaries])
        axis.set_ylabel("Property price")
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.3)
        axis.set_axisbelow(True)
        axis.set_xlim(-0.5, len(summaries) - 0.5)

        tick_values = [300_000, 1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000, 300_000_000]
        lower_bound = min(summary["p10"] for _, summary, _ in summaries)
        upper_bound = max(summary["p90"] for _, summary, _ in summaries)
        tick_values = [value for value in tick_values if lower_bound <= value <= upper_bound * 1.2]
        if tick_values:
            axis.set_yticks(tick_values)
            axis.set_yticklabels(
                [f"{value / 1_000_000:.1f}M RUB" if value < 1_000_000 else f"{int(value / 1_000_000)}M RUB" for value in tick_values]
            )

        legend_handles = [
            Line2D([0], [0], color="#6f6f6f", linewidth=3, label="10th–90th percentiles"),
            Line2D([0], [0], color="#4c78a8", linewidth=12, label="Interquartile range"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#c44e52", markersize=10, label="Median"),
        ]
        axis.legend(handles=legend_handles, frameon=True, loc="upper right")
        return self._save_figure(figure, output_filename)

    def plot_price_band_comparison(
        self,
        raw_frame: pd.DataFrame,
        clean_frame: pd.DataFrame,
        output_filename: str,
    ) -> Path:
        def _prepare(frame: pd.DataFrame) -> pd.Series:
            prepared = frame.dropna(subset=["Price"]).copy()
            prepared = prepared.loc[prepared["Price"] > 0].copy()
            return prepared["Price"]

        raw_prices = _prepare(raw_frame)
        clean_prices = _prepare(clean_frame)

        band_edges = [0, 3_000_000, 5_000_000, 10_000_000, 20_000_000, float("inf")]
        band_labels = [
            "Under 3M RUB",
            "3–5M RUB",
            "5–10M RUB",
            "10–20M RUB",
            "Over 20M RUB",
        ]

        raw_counts = pd.cut(raw_prices, bins=band_edges, labels=band_labels, include_lowest=True).value_counts(sort=False)
        clean_counts = pd.cut(clean_prices, bins=band_edges, labels=band_labels, include_lowest=True).value_counts(sort=False)

        plot_frame = pd.DataFrame(
            {
                "Price range": band_labels,
                "Before cleaning": raw_counts.reindex(band_labels, fill_value=0).to_numpy(),
                "After cleaning": clean_counts.reindex(band_labels, fill_value=0).to_numpy(),
            }
        )

        figure, axis = plt.subplots(figsize=(12, 7))
        x_positions = np.arange(len(band_labels))
        bar_width = 0.36

        raw_bars = axis.bar(
            x_positions - bar_width / 2,
            plot_frame["Before cleaning"],
            width=bar_width,
            color="#9c755f",
            label="Before cleaning",
        )
        clean_bars = axis.bar(
            x_positions + bar_width / 2,
            plot_frame["After cleaning"],
            width=bar_width,
            color="#4c78a8",
            label="After cleaning",
        )

        axis.set_title("Properties by price range")
        axis.set_xlabel("Price range")
        axis.set_ylabel("Number of properties")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(band_labels)
        axis.legend(frameon=True)
        axis.grid(axis="y", alpha=0.3)
        axis.set_axisbelow(True)

        for bars in (raw_bars, clean_bars):
            axis.bar_label(bars, padding=3, fontsize=11, fmt="%.0f")

        return self._save_figure(figure, output_filename)

    def plot_cleaning_rule_impact(
        self,
        report_frame: pd.DataFrame,
        output_filename: str,
    ) -> Path:
        label_mapping = {
            "invalid_kitchen": "Invalid kitchen area",
            "invalid_rooms": "Invalid room count",
            "invalid_price": "Invalid price",
            "invalid_area": "Invalid property area",
            "floor_exceeds_building": "Floor exceeds building height",
            "invalid_floors": "Invalid building height",
            "invalid_floor": "Invalid floor",
            "invalid_living_area": "Invalid living area",
        }

        plot_frame = report_frame.copy()
        plot_frame["Cleaning rule"] = plot_frame["rule"].map(label_mapping).fillna(plot_frame["rule"])
        plot_frame["Affected rows"] = plot_frame["affected_rows"].astype(int)
        plot_frame = plot_frame.sort_values(by="Affected rows", ascending=True)

        figure, axis = plt.subplots(figsize=(12, 7))
        sns.barplot(
            data=plot_frame,
            x="Affected rows",
            y="Cleaning rule",
            color="#4c78a8",
            ax=axis,
        )

        axis.set_title("Effect of cleaning rules on the input data")
        axis.set_xlabel("Affected rows")
        axis.set_ylabel("Cleaning rule")
        axis.grid(axis="x", alpha=0.3)
        axis.set_axisbelow(True)
        format_numeric_axis(axis, x=True, nbins=6)

        for container in axis.containers:
            axis.bar_label(container, fmt="%.0f", padding=4, fontsize=11)

        return self._save_figure(figure, output_filename)

    def plot_custom_boxplot(
        self,
        frame: pd.DataFrame,
        value_column: str,
        title: str,
        output_filename: str,
    ) -> Path:
        figure, axis = plt.subplots(figsize=(12, 3.5))
        sns.boxplot(x=frame[value_column].dropna(), color="#2ca02c", ax=axis)
        axis.set_title(title)
        axis.set_xlabel(column_label(value_column))
        format_numeric_axis(axis, x=True, nbins=6)
        return self._save_figure(figure, output_filename)

    def _save_figure(self, figure: plt.Figure, filename: str) -> Path:
        output_path = self.output_dir / filename
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close(figure)
        self.logger.info("Saved plot: %s", output_path.name)
        return output_path
