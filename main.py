from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from config.config import AppConfig
from src.evaluation.recommendation_evaluator import RecommendationEvaluator
from src.experiments.decision_support import DecisionSupportEngine
from src.experiments.final_analytics import FinalAnalyticsBuilder
from src.experiments.final_demo import FinalDemoBuilder
from src.feature_engineering.feature_builder import RealEstateFeatureBuilder
from src.models.model_evaluator import PriceModelEvaluator
from src.models.price_model import PriceModelTrainer
from src.preprocessing.eda_analyzer import RealEstateEDAAnalyzer
from src.preprocessing.data_merger import SchemaHarmonizer
from src.preprocessing.preprocessor import RealEstatePreprocessor
from src.recommendation.collaborative import CollaborativeFilteringRecommender
from src.recommendation.content_based import ContentBasedRecommender
from src.recommendation.hybrid import HybridRecommender
from src.recommendation.knowledge_based import KnowledgeBasedRecommender
from src.utils.helpers import discover_csv_files, ensure_directories, project_tree_snapshot, serialize_to_json
from src.utils.logger import get_logger, setup_logging


@dataclass(slots=True)
class BootstrapSummary:
    """Metadata produced by the initial project bootstrap stage."""

    project_name: str
    version: str
    project_root: str
    discovered_datasets: list[str]
    created_directories: list[str]


class ProjectBootstrapper:
    """Initializes folders, logging, and metadata for the diploma project."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._logger = get_logger(self.__class__.__name__)

    def initialize(self) -> BootstrapSummary:
        ensure_directories(self.config.required_directories())
        setup_logging(self.config.paths.logs_dir, self.config.logging)
        self._logger.info("Project directories and logging are initialized.")

        created_directories = [str(path) for path in self.config.required_directories()]
        discovered = self._discover_source_datasets()
        self._logger.info("Discovered %s candidate dataset(s).", len(discovered))

        summary = BootstrapSummary(
            project_name=self.config.project_name,
            version=self.config.version,
            project_root=str(self.config.paths.project_root),
            discovered_datasets=[str(path) for path in discovered],
            created_directories=created_directories,
        )
        serialize_to_json(summary, self.config.paths.metadata_dir / "bootstrap_summary.json")
        return summary

    def _discover_source_datasets(self) -> list[Path]:
        roots = (
            self.config.paths.raw_data_dir,
            self.config.paths.project_root,
        )
        discovered = discover_csv_files(roots, self.config.datasets.source_patterns)
        return [
            path for path in discovered
            if path.suffix.lower() in self.config.datasets.accepted_extensions
        ]


def print_summary(summary: BootstrapSummary, project_root: Path) -> None:
    """Emit a concise human-readable architecture summary."""

    logger = get_logger("bootstrap_summary")
    logger.info("Project: %s v%s", summary.project_name, summary.version)
    logger.info("Root: %s", summary.project_root)
    logger.info("Datasets discovered: %s", len(summary.discovered_datasets))
    for dataset in summary.discovered_datasets:
        logger.info("Dataset candidate: %s", dataset)

    logger.info("Project structure snapshot:")
    for line in project_tree_snapshot(project_root, max_depth=2):
        logger.info(line)


def main() -> None:
    """Entry point for the diploma project's orchestration pipeline."""

    parser = argparse.ArgumentParser(description="Housing decision support system pipeline.")
    parser.add_argument("--run-merge", action="store_true", help="Run stage 2 dataset harmonization.")
    parser.add_argument("--run-eda", action="store_true", help="Run stage 3 exploratory data analysis.")
    parser.add_argument("--run-preprocessing", action="store_true", help="Run stage 4 data preprocessing.")
    parser.add_argument("--run-price-models", action="store_true", help="Run stage 5 housing price model training.")
    parser.add_argument("--run-model-evaluation", action="store_true", help="Run stage 6 price model evaluation.")
    parser.add_argument("--run-feature-engineering", action="store_true", help="Run stage 7 feature engineering.")
    parser.add_argument("--run-content-based", action="store_true", help="Run stage 8 content-based recommendation.")
    parser.add_argument("--run-collaborative", action="store_true", help="Run stage 9 collaborative filtering.")
    parser.add_argument("--run-knowledge-based", action="store_true", help="Run stage 10 knowledge-based recommendation.")
    parser.add_argument("--run-hybrid", action="store_true", help="Run stage 11 hybrid recommendation.")
    parser.add_argument("--run-recommendation-evaluation", action="store_true", help="Run stage 12 recommendation evaluation.")
    parser.add_argument("--run-decision-support", action="store_true", help="Run stage 13 integrated decision-support shortlist.")
    parser.add_argument("--run-final-demo", action="store_true", help="Run stage 14 end-to-end demo packaging.")
    parser.add_argument("--run-final-analytics", action="store_true", help="Run stage 15 final analytical chapter assembly.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config = AppConfig.from_root(project_root)
    bootstrapper = ProjectBootstrapper(config)
    summary = bootstrapper.initialize()
    print_summary(summary, project_root)

    if args.run_merge:
        harmonizer = SchemaHarmonizer(config)
        harmonizer.execute()

    if args.run_eda:
        analyzer = RealEstateEDAAnalyzer(config)
        analyzer.execute()

    if args.run_preprocessing:
        preprocessor = RealEstatePreprocessor(config)
        preprocessor.execute()

    if args.run_price_models:
        trainer = PriceModelTrainer(config)
        trainer.execute()

    if args.run_model_evaluation:
        evaluator = PriceModelEvaluator(config)
        evaluator.execute()

    if args.run_feature_engineering:
        feature_builder = RealEstateFeatureBuilder(config)
        feature_builder.execute()

    if args.run_content_based:
        recommender = ContentBasedRecommender(config)
        recommender.execute()

    if args.run_collaborative:
        collaborative = CollaborativeFilteringRecommender(config)
        collaborative.execute()

    if args.run_knowledge_based:
        recommender = KnowledgeBasedRecommender(config)
        recommender.execute()

    if args.run_hybrid:
        recommender = HybridRecommender(config)
        recommender.execute()

    if args.run_recommendation_evaluation:
        evaluator = RecommendationEvaluator(config)
        evaluator.execute()

    if args.run_decision_support:
        engine = DecisionSupportEngine(config)
        engine.execute()

    if args.run_final_demo:
        builder = FinalDemoBuilder(config)
        builder.execute()

    if args.run_final_analytics:
        builder = FinalAnalyticsBuilder(config)
        builder.execute()

if __name__ == "__main__":
    main()
