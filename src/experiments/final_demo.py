from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config.config import AppConfig, FinalDemoConfig
from src.experiments.decision_support import DecisionSupportEngine
from src.utils.helpers import serialize_to_json
from src.utils.logger import get_logger
from src.visualization.final_demo_visualizer import FinalDemoVisualizer


@dataclass(slots=True)
class FinalDemoSummary:
    """Metadata produced by the final end-to-end demo stage."""

    report_path: str
    payload_path: str
    best_price_model: str
    best_recommendation_method: str
    top_recommendation_listing_id: int
    tables_created: list[str]
    plots_created: list[str]


class FinalDemoBuilder:
    """Packages the final DSS outputs into a demo-ready report and payload."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.demo_config: FinalDemoConfig = config.final_demo
        self.logger = get_logger(self.__class__.__name__)
        self.visualizer = FinalDemoVisualizer(self.config.paths.plots_dir, self.demo_config)

    def execute(self) -> FinalDemoSummary:
        """Build the final demo artifacts and a concise markdown report."""

        shortlist_path = self.config.paths.tables_dir / self.demo_config.decision_support_shortlist_filename
        if not shortlist_path.exists():
            self.logger.info("Decision-support shortlist is missing; running stage 13 prerequisite.")
            DecisionSupportEngine(self.config).execute()

        shortlist_frame = pd.read_csv(shortlist_path)
        shortlist_preview = shortlist_frame.head(self.demo_config.top_k_preview).copy()

        decision_support_summary = self._load_json(
            self.config.paths.metadata_dir / self.demo_config.decision_support_summary_filename
        )
        decision_support_profile = self._load_json(
            self.config.paths.metadata_dir / self.demo_config.decision_support_profile_filename
        )
        price_model_summary = self._load_json(
            self.config.paths.metadata_dir / self.demo_config.price_model_summary_filename
        )
        recommendation_summary = self._load_json(
            self.config.paths.metadata_dir / self.demo_config.recommendation_summary_filename
        )

        system_summary_frame = self._build_system_summary(
            decision_support_summary,
            price_model_summary,
            recommendation_summary,
            shortlist_frame,
        )
        payload = self._build_demo_payload(
            decision_support_profile,
            decision_support_summary,
            price_model_summary,
            recommendation_summary,
            shortlist_preview,
        )

        report_path = self.config.paths.reports_dir / self.demo_config.report_filename
        payload_path = self.config.paths.metadata_dir / self.demo_config.payload_filename
        system_summary_path = self.config.paths.tables_dir / self.demo_config.system_summary_table_filename
        shortlist_preview_path = self.config.paths.tables_dir / self.demo_config.shortlist_preview_filename

        system_summary_frame.to_csv(system_summary_path, index=False, encoding="utf-8-sig")
        shortlist_preview.to_csv(shortlist_preview_path, index=False, encoding="utf-8-sig")
        serialize_to_json(payload, payload_path)
        report_path.write_text(
            self._build_markdown_report(
                decision_support_profile,
                decision_support_summary,
                price_model_summary,
                recommendation_summary,
                shortlist_preview,
            ),
            encoding="utf-8",
        )

        score_plot_path = self.visualizer.plot_score_overview(
            shortlist_preview,
            self.demo_config.score_plot_filename,
        )
        valuation_plot_path = self.visualizer.plot_price_vs_market(
            shortlist_preview,
            self.demo_config.valuation_plot_filename,
        )

        summary = FinalDemoSummary(
            report_path=str(report_path),
            payload_path=str(payload_path),
            best_price_model=str(price_model_summary["best_model_name"]),
            best_recommendation_method=str(recommendation_summary["best_method_name"]),
            top_recommendation_listing_id=int(shortlist_preview.iloc[0]["listing_id"]),
            tables_created=[
                str(system_summary_path),
                str(shortlist_preview_path),
            ],
            plots_created=[
                str(score_plot_path),
                str(valuation_plot_path),
            ],
        )
        serialize_to_json(summary, self.config.paths.metadata_dir / self.demo_config.summary_filename)
        self.logger.info(
            "Stage 14 completed: top recommendation listing_id=%s.",
            summary.top_recommendation_listing_id,
        )
        return summary

    def _build_system_summary(
        self,
        decision_support_summary: dict[str, Any],
        price_model_summary: dict[str, Any],
        recommendation_summary: dict[str, Any],
        shortlist_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        top_row = shortlist_frame.iloc[0]
        records = [
            {"component": "best_price_model", "value": price_model_summary["best_model_name"]},
            {"component": "best_recommendation_method", "value": recommendation_summary["best_method_name"]},
            {"component": "decision_support_candidates", "value": decision_support_summary["candidate_rows"]},
            {"component": "decision_support_shortlist_size", "value": decision_support_summary["shortlisted_rows"]},
            {"component": "top_listing_id", "value": int(top_row["listing_id"])},
            {"component": "top_listing_price", "value": float(top_row["Price"])},
            {"component": "top_listing_predicted_market_price", "value": float(top_row["predicted_market_price"])},
            {"component": "top_listing_relative_discount", "value": float(top_row["relative_discount_to_market"])},
            {"component": "top_listing_decision_score", "value": float(top_row["decision_support_score"])},
        ]
        return pd.DataFrame(records)

    def _build_demo_payload(
        self,
        decision_support_profile: dict[str, Any],
        decision_support_summary: dict[str, Any],
        price_model_summary: dict[str, Any],
        recommendation_summary: dict[str, Any],
        shortlist_preview: pd.DataFrame,
    ) -> dict[str, Any]:
        return {
            "project_name": self.config.project_name,
            "version": self.config.version,
            "profile": decision_support_profile,
            "decision_support_summary": decision_support_summary,
            "best_price_model": price_model_summary["best_model_name"],
            "best_recommendation_method": recommendation_summary["best_method_name"],
            "shortlist_preview": shortlist_preview.to_dict(orient="records"),
        }

    def _build_markdown_report(
        self,
        decision_support_profile: dict[str, Any],
        decision_support_summary: dict[str, Any],
        price_model_summary: dict[str, Any],
        recommendation_summary: dict[str, Any],
        shortlist_preview: pd.DataFrame,
        ) -> str:
        top_row = shortlist_preview.iloc[0]
        preview_columns = [
            "rank",
            "listing_id",
            "Price",
            "predicted_market_price",
            "relative_discount_to_market",
            "decision_support_score",
            "Metro station",
            "Renovation",
        ]
        preview_table = self._frame_to_markdown_table(shortlist_preview.loc[:, preview_columns])
        lines = [
            "# Final Decision Support Demo Report",
            "",
            "## Profile",
            f"- Budget: {decision_support_profile['budget']}",
            f"- Desired area: {decision_support_profile['desired_area']}",
            f"- Desired rooms: {decision_support_profile['desired_rooms']}",
            f"- Preferred metro: {decision_support_profile['preferred_metro_station']}",
            f"- Preferred renovation: {decision_support_profile['preferred_renovation']}",
            "",
            "## Best Components",
            f"- Best price model: {price_model_summary['best_model_name']}",
            f"- Best recommendation method: {recommendation_summary['best_method_name']}",
            "",
            "## Decision Support Pool",
            f"- Candidates after filtering: {decision_support_summary['candidate_rows']}",
            f"- Final shortlist size: {decision_support_summary['shortlisted_rows']}",
            "",
            "## Top Recommendation",
            f"- Listing ID: {int(top_row['listing_id'])}",
            f"- Asking price: {float(top_row['Price']):,.2f}",
            f"- Predicted market price: {float(top_row['predicted_market_price']):,.2f}",
            f"- Relative discount to market: {float(top_row['relative_discount_to_market']):.4f}",
            f"- Decision support score: {float(top_row['decision_support_score']):.4f}",
            f"- Explanation: {top_row['explanation']}",
            "",
            "## Top-5 Preview",
            preview_table,
        ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    @staticmethod
    def _frame_to_markdown_table(frame: pd.DataFrame) -> str:
        """Render a compact Markdown table without external dependencies."""

        headers = [str(column) for column in frame.columns]
        separator = ["---"] * len(headers)
        rows = [headers, separator]

        for _, row in frame.iterrows():
            rendered_row: list[str] = []
            for value in row:
                if pd.isna(value):
                    rendered_row.append("")
                elif isinstance(value, float):
                    rendered_row.append(f"{value:.4f}")
                else:
                    rendered_row.append(str(value))
            rows.append(rendered_row)

        return "\n".join("| " + " | ".join(row) + " |" for row in rows)
