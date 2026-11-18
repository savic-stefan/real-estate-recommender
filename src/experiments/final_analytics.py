from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config.config import AppConfig, FinalAnalyticsConfig
from src.experiments.final_demo import FinalDemoBuilder
from src.utils.helpers import serialize_to_json
from src.utils.logger import get_logger
from src.visualization.final_analytics_visualizer import FinalAnalyticsVisualizer


@dataclass(slots=True)
class FinalAnalyticsSummary:
    """Metadata produced by the final analytical chapter stage."""

    report_path: str
    best_price_model: str
    best_recommendation_method: str
    decision_support_top_listing_id: int
    findings_count: int
    limitations_count: int
    tables_created: list[str]
    plots_created: list[str]


class FinalAnalyticsBuilder:
    """Assembles a chapter-ready analytical summary from all prior experiment outputs."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.analytics_config: FinalAnalyticsConfig = config.final_analytics
        self.logger = get_logger(self.__class__.__name__)
        self.visualizer = FinalAnalyticsVisualizer(self.config.paths.plots_dir, self.analytics_config)

    def execute(self) -> FinalAnalyticsSummary:
        """Build final findings, limitations, stage outcomes, plots, and markdown chapter."""

        self._ensure_prerequisites()

        preprocessing_summary = self._load_json(
            self.config.paths.metadata_dir / self.analytics_config.preprocessing_summary_filename
        )
        price_summary = self._load_json(
            self.config.paths.metadata_dir / self.analytics_config.price_model_summary_filename
        )
        recommendation_summary = self._load_json(
            self.config.paths.metadata_dir / self.analytics_config.recommendation_summary_filename
        )
        feature_summary = self._load_json(
            self.config.paths.metadata_dir / self.analytics_config.feature_engineering_summary_filename
        )
        knowledge_summary = self._load_json(
            self.config.paths.metadata_dir / self.analytics_config.knowledge_summary_filename
        )
        hybrid_summary = self._load_json(
            self.config.paths.metadata_dir / self.analytics_config.hybrid_summary_filename
        )
        decision_support_summary = self._load_json(
            self.config.paths.metadata_dir / self.analytics_config.decision_support_summary_filename
        )
        final_demo_summary = self._load_json(
            self.config.paths.metadata_dir / self.analytics_config.final_demo_summary_filename
        )

        price_frame = pd.read_csv(self.config.paths.tables_dir / self.analytics_config.price_model_table_filename)
        recommendation_frame = pd.read_csv(
            self.config.paths.tables_dir / self.analytics_config.recommendation_table_filename
        )
        feature_impact_frame = pd.read_csv(
            self.config.paths.tables_dir / self.analytics_config.feature_impact_table_filename
        )
        hybrid_component_frame = pd.read_csv(
            self.config.paths.tables_dir / self.analytics_config.hybrid_component_table_filename
        )
        shortlist_frame = pd.read_csv(self.config.paths.tables_dir / self.analytics_config.shortlist_filename)

        stage_outcomes = self._build_stage_outcomes(
            preprocessing_summary=preprocessing_summary,
            feature_summary=feature_summary,
            price_frame=price_frame,
            recommendation_frame=recommendation_frame,
            feature_impact_frame=feature_impact_frame,
            hybrid_summary=hybrid_summary,
            decision_support_summary=decision_support_summary,
            final_demo_summary=final_demo_summary,
        )
        findings = self._build_key_findings(
            price_frame=price_frame,
            recommendation_frame=recommendation_frame,
            feature_impact_frame=feature_impact_frame,
            hybrid_component_frame=hybrid_component_frame,
            shortlist_frame=shortlist_frame,
            decision_support_summary=decision_support_summary,
        )
        limitations = self._build_limitations(
            recommendation_frame=recommendation_frame,
            feature_impact_frame=feature_impact_frame,
            knowledge_summary=knowledge_summary,
            shortlist_frame=shortlist_frame,
        )

        stage_outcomes_path = self.config.paths.tables_dir / self.analytics_config.stage_outcomes_table_filename
        findings_path = self.config.paths.tables_dir / self.analytics_config.findings_table_filename
        limitations_path = self.config.paths.tables_dir / self.analytics_config.limitations_table_filename
        report_path = self.config.paths.reports_dir / self.analytics_config.report_filename

        stage_outcomes.to_csv(stage_outcomes_path, index=False, encoding="utf-8-sig")
        findings.to_csv(findings_path, index=False, encoding="utf-8-sig")
        limitations.to_csv(limitations_path, index=False, encoding="utf-8-sig")

        report_path.write_text(
            self._build_markdown_report(
                stage_outcomes=stage_outcomes,
                findings=findings,
                limitations=limitations,
                price_summary=price_summary,
                recommendation_summary=recommendation_summary,
                decision_support_summary=decision_support_summary,
                shortlist_frame=shortlist_frame,
            ),
            encoding="utf-8",
        )

        price_plot_path = self.visualizer.plot_price_models(
            price_frame,
            self.analytics_config.price_model_plot_filename,
        )
        recommendation_plot_path = self.visualizer.plot_recommendation_methods(
            recommendation_frame,
            self.analytics_config.recommendation_plot_filename,
        )
        shortlist_plot_path = self.visualizer.plot_shortlist_discounts(
            shortlist_frame,
            self.analytics_config.shortlist_plot_filename,
        )

        summary = FinalAnalyticsSummary(
            report_path=str(report_path),
            best_price_model=str(price_summary["best_model_name"]),
            best_recommendation_method=str(recommendation_summary["best_method_name"]),
            decision_support_top_listing_id=int(shortlist_frame.iloc[0]["listing_id"]),
            findings_count=len(findings),
            limitations_count=len(limitations),
            tables_created=[
                str(stage_outcomes_path),
                str(findings_path),
                str(limitations_path),
            ],
            plots_created=[
                str(price_plot_path),
                str(recommendation_plot_path),
                str(shortlist_plot_path),
            ],
        )
        serialize_to_json(summary, self.config.paths.metadata_dir / self.analytics_config.summary_filename)
        self.logger.info(
            "Stage 15 completed: findings=%s, limitations=%s.",
            summary.findings_count,
            summary.limitations_count,
        )
        return summary

    def _ensure_prerequisites(self) -> None:
        required_paths = [
            self.config.paths.metadata_dir / self.analytics_config.preprocessing_summary_filename,
            self.config.paths.metadata_dir / self.analytics_config.price_model_summary_filename,
            self.config.paths.metadata_dir / self.analytics_config.recommendation_summary_filename,
            self.config.paths.metadata_dir / self.analytics_config.feature_engineering_summary_filename,
            self.config.paths.metadata_dir / self.analytics_config.knowledge_summary_filename,
            self.config.paths.metadata_dir / self.analytics_config.hybrid_summary_filename,
            self.config.paths.metadata_dir / self.analytics_config.decision_support_summary_filename,
            self.config.paths.tables_dir / self.analytics_config.price_model_table_filename,
            self.config.paths.tables_dir / self.analytics_config.recommendation_table_filename,
            self.config.paths.tables_dir / self.analytics_config.feature_impact_table_filename,
            self.config.paths.tables_dir / self.analytics_config.hybrid_component_table_filename,
            self.config.paths.tables_dir / self.analytics_config.shortlist_filename,
        ]

        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            self.logger.info("Some analytical prerequisites are missing; running stage 14 packaging.")
            FinalDemoBuilder(self.config).execute()

    def _build_stage_outcomes(
        self,
        preprocessing_summary: dict[str, Any],
        feature_summary: dict[str, Any],
        price_frame: pd.DataFrame,
        recommendation_frame: pd.DataFrame,
        feature_impact_frame: pd.DataFrame,
        hybrid_summary: dict[str, Any],
        decision_support_summary: dict[str, Any],
        final_demo_summary: dict[str, Any],
    ) -> pd.DataFrame:
        best_price_row = price_frame.sort_values(by="test_rmse", ascending=True).iloc[0]
        best_recommendation_row = recommendation_frame.sort_values(by="dcg_utility", ascending=False).iloc[0]
        feature_baseline = feature_summary["baseline_feature_count"]
        feature_enhanced = feature_summary["enhanced_feature_count"]
        enhanced_row = feature_impact_frame.loc[feature_impact_frame["feature_set"] == "enhanced"].iloc[0]
        records = [
            {
                "stage": "Stage 4 - Preprocessing",
                "headline": "Data quality pipeline",
                "key_result": f"{preprocessing_summary['cleaned_rows']} clean rows; {preprocessing_summary['modeling_rows']} modeling rows",
            },
            {
                "stage": "Stage 6 - Price Evaluation",
                "headline": "Best market valuation model",
                "key_result": f"{best_price_row['model_name']} with RMSE {best_price_row['test_rmse']:.4f} and R2 {best_price_row['test_r2']:.6f}",
            },
            {
                "stage": "Stage 7 - Feature Engineering",
                "headline": "Engineered feature impact",
                "key_result": f"{feature_baseline} baseline features vs {feature_enhanced} enhanced features; RMSE delta {enhanced_row['delta_rmse_vs_baseline']:.4f}",
            },
            {
                "stage": "Stage 12 - Recommendation Evaluation",
                "headline": "Best recommendation method",
                "key_result": f"{best_recommendation_row['method_name']} with DCG {best_recommendation_row['dcg_utility']:.6f}",
            },
            {
                "stage": "Stage 11 - Hybrid Recommendation",
                "headline": "Hybrid integration result",
                "key_result": f"{hybrid_summary['candidate_rows']} gated candidates; collaborative signal remained non-zero but small",
            },
            {
                "stage": "Stage 13 - Decision Support",
                "headline": "Integrated shortlist",
                "key_result": f"{decision_support_summary['candidate_rows']} feasible candidates; {decision_support_summary['shortlisted_rows']} final shortlist objects",
            },
            {
                "stage": "Stage 14 - Final Demo",
                "headline": "Defense-ready packaging",
                "key_result": f"Top listing {final_demo_summary['top_recommendation_listing_id']} collected into final demo payload",
            },
        ]
        return pd.DataFrame(records)

    def _build_key_findings(
        self,
        price_frame: pd.DataFrame,
        recommendation_frame: pd.DataFrame,
        feature_impact_frame: pd.DataFrame,
        hybrid_component_frame: pd.DataFrame,
        shortlist_frame: pd.DataFrame,
        decision_support_summary: dict[str, Any],
    ) -> pd.DataFrame:
        best_price_row = price_frame.sort_values(by="test_rmse", ascending=True).iloc[0]
        linear_row = price_frame.loc[price_frame["model_name"] == "Linear Regression"].iloc[0]
        rmse_improvement_ratio = 1.0 - (float(best_price_row["test_rmse"]) / float(linear_row["test_rmse"]))

        best_recommendation_row = recommendation_frame.sort_values(by="dcg_utility", ascending=False).iloc[0]
        hybrid_row = recommendation_frame.loc[recommendation_frame["method_name"] == "hybrid"].iloc[0]
        knowledge_row = recommendation_frame.loc[recommendation_frame["method_name"] == "knowledge_based"].iloc[0]

        enhanced_row = feature_impact_frame.loc[feature_impact_frame["feature_set"] == "enhanced"].iloc[0]
        top_shortlist_row = shortlist_frame.iloc[0]

        findings = [
            {
                "finding_id": "F1",
                "theme": "Price modeling",
                "statement": "Random Forest Regressor was the best market valuation model on holdout data.",
                "evidence": f"test_rmse={best_price_row['test_rmse']:.4f}; test_r2={best_price_row['test_r2']:.6f}",
            },
            {
                "finding_id": "F2",
                "theme": "Price modeling",
                "statement": "Nonlinear ensembles substantially outperformed the linear baseline.",
                "evidence": f"RMSE improvement over Linear Regression={rmse_improvement_ratio:.4%}",
            },
            {
                "finding_id": "F3",
                "theme": "Feature engineering",
                "statement": "Engineered domain features were informative but did not improve direct price regression.",
                "evidence": f"enhanced_delta_rmse={enhanced_row['delta_rmse_vs_baseline']:.4f}; enhanced_delta_r2={enhanced_row['delta_r2_vs_baseline']:.6f}",
            },
            {
                "finding_id": "F4",
                "theme": "Recommendation",
                "statement": "The weighted content-based recommender achieved the best offline utility.",
                "evidence": f"method={best_recommendation_row['method_name']}; dcg_utility={best_recommendation_row['dcg_utility']:.6f}",
            },
            {
                "finding_id": "F5",
                "theme": "Recommendation",
                "statement": "Hybrid and knowledge-based methods perfectly satisfied hard constraints, but did not surpass content_weighted on utility.",
                "evidence": f"hybrid_constraint_rate={hybrid_row['hard_constraint_rate']:.4f}; knowledge_constraint_rate={knowledge_row['hard_constraint_rate']:.4f}",
            },
            {
                "finding_id": "F6",
                "theme": "Hybrid integration",
                "statement": "Collaborative signal remained the weakest component inside the hybrid recommendation stage.",
                "evidence": f"hybrid_collaborative_component_mean={float(hybrid_component_frame.loc[hybrid_component_frame['component'] == 'Collaborative', 'mean_score'].iloc[0]):.6f}",
            },
            {
                "finding_id": "F7",
                "theme": "Decision support",
                "statement": "The integrated shortlist contained market-underpriced objects relative to the learned valuation model.",
                "evidence": f"top_listing_discount={float(top_shortlist_row['relative_discount_to_market']):.6f}; top_listing_score={float(top_shortlist_row['decision_support_score']):.6f}",
            },
            {
                "finding_id": "F8",
                "theme": "Decision support",
                "statement": "The final DSS pipeline reduced a large candidate pool to a compact shortlist suitable for human review.",
                "evidence": f"candidate_pool={decision_support_summary['candidate_rows']}; shortlist={decision_support_summary['shortlisted_rows']}",
            },
        ]
        return pd.DataFrame(findings)

    def _build_limitations(
        self,
        recommendation_frame: pd.DataFrame,
        feature_impact_frame: pd.DataFrame,
        knowledge_summary: dict[str, Any],
        shortlist_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        enhanced_row = feature_impact_frame.loc[feature_impact_frame["feature_set"] == "enhanced"].iloc[0]
        collab_item_row = recommendation_frame.loc[
            recommendation_frame["method_name"] == "collaborative_item_based"
        ].iloc[0]
        collab_user_row = recommendation_frame.loc[
            recommendation_frame["method_name"] == "collaborative_user_based"
        ].iloc[0]
        unknown_metro = int((shortlist_frame["Metro station"] == "Unknown").sum())
        unknown_renovation = int((shortlist_frame["Renovation"] == "Unknown").sum())

        limitations = [
            {
                "limitation_id": "L1",
                "area": "Data quality",
                "issue": "A large share of shortlist objects still has missing categorical semantics.",
                "evidence": f"unknown_metro={unknown_metro}/15; unknown_renovation={unknown_renovation}/15",
            },
            {
                "limitation_id": "L2",
                "area": "Collaborative filtering",
                "issue": "Collaborative recommendation quality is constrained by synthetic and sparse feedback.",
                "evidence": f"item_based_dcg={collab_item_row['dcg_utility']:.6f}; user_based_dcg={collab_user_row['dcg_utility']:.6f}",
            },
            {
                "limitation_id": "L3",
                "area": "Knowledge-based rules",
                "issue": "Exact soft preference rules were too restrictive to apply safely in the candidate pool.",
                "evidence": f"optional_rules_applied={len(knowledge_summary['optional_rules_applied'])}",
            },
            {
                "limitation_id": "L4",
                "area": "Feature engineering",
                "issue": "Rule-based engineered features did not improve direct market price prediction in the current setup.",
                "evidence": f"delta_rmse_vs_baseline={enhanced_row['delta_rmse_vs_baseline']:.4f}",
            },
        ]
        return pd.DataFrame(limitations)

    def _build_markdown_report(
        self,
        stage_outcomes: pd.DataFrame,
        findings: pd.DataFrame,
        limitations: pd.DataFrame,
        price_summary: dict[str, Any],
        recommendation_summary: dict[str, Any],
        decision_support_summary: dict[str, Any],
        shortlist_frame: pd.DataFrame,
    ) -> str:
        shortlist_preview = shortlist_frame.head(self.analytics_config.shortlist_preview_rows).loc[
            :,
            [
                "rank",
                "listing_id",
                "Price",
                "predicted_market_price",
                "relative_discount_to_market",
                "decision_support_score",
                "explanation",
            ],
        ]
        lines = [
            "# Final Analytical Chapter",
            "",
            "## Core Result",
            f"- Best price model: {price_summary['best_model_name']}",
            f"- Best recommendation method: {recommendation_summary['best_method_name']}",
            f"- Final decision-support shortlist size: {decision_support_summary['shortlisted_rows']}",
            f"- Top shortlisted listing: {int(shortlist_frame.iloc[0]['listing_id'])}",
            "",
            "## Stage Outcomes",
            self._frame_to_markdown_table(stage_outcomes),
            "",
            "## Key Findings",
            self._frame_to_markdown_table(findings),
            "",
            "## System Limitations",
            self._frame_to_markdown_table(limitations),
            "",
            "## Shortlist Preview",
            self._frame_to_markdown_table(shortlist_preview),
            "",
            "## Analytical Conclusion",
            "The project demonstrates that nonlinear market valuation and weighted content-based recommendation provide the strongest empirical foundation for the housing decision-support system. The integrated DSS shortlist is practically useful because it combines preference fit with valuation-based discount signals. At the same time, the experiments show that data sparsity, synthetic collaborative feedback, and missing categorical values still constrain the upper bound of recommendation quality.",
        ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    @staticmethod
    def _frame_to_markdown_table(frame: pd.DataFrame) -> str:
        """Render a compact Markdown table without third-party tabulation helpers."""

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
