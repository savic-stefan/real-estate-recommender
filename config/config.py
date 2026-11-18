from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class PathConfig:
    """Centralized project paths for data, reports, artifacts, and source code."""

    project_root: Path
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    tables_dir: Path = field(init=False)
    notebooks_dir: Path = field(init=False)
    src_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.project_root = self.project_root.resolve()
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.reports_dir = self.project_root / "reports"
        self.plots_dir = self.reports_dir / "plots"
        self.tables_dir = self.reports_dir / "tables"
        self.notebooks_dir = self.project_root / "notebooks"
        self.src_dir = self.project_root / "src"
        self.artifacts_dir = self.project_root / "artifacts"
        self.models_dir = self.artifacts_dir / "models"
        self.logs_dir = self.artifacts_dir / "logs"
        self.metadata_dir = self.artifacts_dir / "metadata"

    def required_directories(self) -> tuple[Path, ...]:
        return (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.reports_dir,
            self.plots_dir,
            self.tables_dir,
            self.notebooks_dir,
            self.src_dir,
            self.artifacts_dir,
            self.models_dir,
            self.logs_dir,
            self.metadata_dir,
        )


@dataclass(slots=True)
class LoggingConfig:
    """Controls console and file-based logging behavior."""

    level: str = "INFO"
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    log_filename: str = "system.log"
    enable_console: bool = True
    enable_file: bool = True


@dataclass(slots=True)
class DatasetConfig:
    """Declares source datasets and semantic aliases for schema harmonization."""

    accepted_extensions: tuple[str, ...] = (".csv",)
    source_patterns: tuple[str, ...] = ("*.csv",)
    preferred_filenames: tuple[str, ...] = (
        "dataset_1.csv",
        "dataset_2.csv",
        "dataset_3.csv",
        "data1.csv",
        "data2.csv",
        "data3.csv",
    )
    sniff_bytes: int = 8192
    profile_sample_rows: int = 3
    merge_chunk_size: int = 100_000
    merged_output_filename: str = "merged_data.csv"
    schema_report_filename: str = "dataset_schema_comparison.csv"
    mapping_report_filename: str = "dataset_column_mappings.csv"
    merge_summary_filename: str = "merge_summary.json"
    canonical_columns: tuple[str, ...] = (
        "Price",
        "Apartment type",
        "Metro station",
        "Minutes to metro",
        "Region",
        "Region id",
        "Number of rooms",
        "Area",
        "Living area",
        "Kitchen area",
        "Floor",
        "Number of floors",
        "Renovation",
        "Building type",
        "Object type",
        "Geo latitude",
        "Geo longitude",
        "Postal code",
        "Street id",
        "House id",
        "Date",
        "Time",
    )
    required_columns: tuple[str, ...] = (
        "Price",
        "Area",
        "Number of rooms",
    )
    metadata_columns: tuple[str, ...] = (
        "Source dataset",
        "Source file",
        "Source row number",
    )
    encoding_candidates: tuple[str, ...] = (
        "utf-8-sig",
        "utf-8",
        "cp1251",
        "latin-1",
    )
    column_aliases: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "Price": ("price", "price_rub", "price_usd", "cost", "sale_price"),
            "Apartment type": ("apartment_type", "flat_type", "housing_type"),
            "Metro station": ("metro_station", "metro", "subway_station", "station"),
            "Minutes to metro": (
                "minutes_to_metro",
                "metro_minutes",
                "time_to_metro",
                "walk_to_metro_minutes",
            ),
            "Region": ("region", "district", "area_name"),
            "Region id": ("id_region", "region_id"),
            "Number of rooms": ("rooms", "room_count", "number_of_rooms"),
            "Area": ("area", "total_area", "full_area"),
            "Living area": ("living_area", "living_space", "residential_area"),
            "Kitchen area": ("kitchen_area", "kitchen_space"),
            "Floor": ("floor", "level", "current_floor"),
            "Number of floors": ("levels", "num_floors", "total_floors", "number_of_floors"),
            "Renovation": ("renovation", "condition", "finish"),
            "Building type": ("building_type", "building_class"),
            "Object type": ("object_type", "property_type"),
            "Geo latitude": ("geo_lat", "latitude", "lat"),
            "Geo longitude": ("geo_lon", "longitude", "lon", "lng"),
            "Postal code": ("postal_code", "postcode", "zip_code"),
            "Street id": ("street_id",),
            "House id": ("house_id",),
            "Date": ("date", "publication_date", "created_at"),
            "Time": ("time", "publication_time"),
        }
    )


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level research settings reused across ML and recommender experiments."""

    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2
    n_splits_cv: int = 5
    top_k_recommendations: int = 10
    target_column: str = "Price"
    currency_label: str = "RUB"


@dataclass(slots=True)
class EDAConfig:
    """Controls reproducible exploratory analysis and plot generation."""

    input_filename: str = "merged_data.csv"
    sample_size: int = 250_000
    chunk_size: int = 200_000
    random_state: int = 42
    top_metro_count: int = 12
    top_region_count: int = 15
    histogram_bins: int = 60
    plot_dpi: int = 140
    numeric_columns: tuple[str, ...] = (
        "Price",
        "Minutes to metro",
        "Number of rooms",
        "Area",
        "Living area",
        "Kitchen area",
        "Floor",
        "Number of floors",
        "Geo latitude",
        "Geo longitude",
    )
    categorical_columns: tuple[str, ...] = (
        "Apartment type",
        "Metro station",
        "Region",
        "Renovation",
        "Building type",
        "Object type",
        "Source dataset",
    )
    duplicate_subset_columns: tuple[str, ...] = (
        "Price",
        "Area",
        "Number of rooms",
        "Floor",
        "Number of floors",
        "Region",
        "Metro station",
        "Source dataset",
    )
    overview_table_filename: str = "eda_overview.csv"
    missing_table_filename: str = "eda_missing_values.csv"
    numeric_summary_filename: str = "eda_numeric_summary.csv"
    categorical_summary_filename: str = "eda_categorical_summary.csv"
    correlation_table_filename: str = "eda_correlation_matrix.csv"
    source_price_table_filename: str = "eda_source_price_summary.csv"
    region_distribution_table_filename: str = "eda_region_distribution.csv"
    metro_distribution_table_filename: str = "eda_metro_distribution.csv"
    duplicates_table_filename: str = "eda_duplicates_sample.csv"
    report_metadata_filename: str = "eda_summary.json"


@dataclass(slots=True)
class PreprocessingConfig:
    """Controls data cleaning, encoding, scaling, and split generation."""

    input_filename: str = "merged_data.csv"
    clean_output_filename: str = "clean_data.csv"
    modeling_sample_filename: str = "modeling_dataset.csv"
    train_output_filename: str = "train_data.csv"
    test_output_filename: str = "test_data.csv"
    summary_filename: str = "preprocessing_summary.json"
    missing_report_filename: str = "preprocessing_missing_report.csv"
    outlier_report_filename: str = "preprocessing_outlier_report.csv"
    encoding_report_filename: str = "preprocessing_encoding_report.csv"
    scaler_report_filename: str = "preprocessing_scaler_report.csv"
    chunk_size: int = 150_000
    modeling_sample_size: int = 300_000
    random_state: int = 42
    test_size: float = 0.2
    price_min_valid: float = 300_000.0
    price_max_valid: float = 300_000_000.0
    area_min_valid: float = 10.0
    area_max_valid: float = 300.0
    living_area_min_valid: float = 8.0
    living_area_max_valid: float = 250.0
    kitchen_area_min_valid: float = 2.0
    kitchen_area_max_valid: float = 80.0
    minutes_to_metro_min_valid: float = 0.0
    minutes_to_metro_max_valid: float = 180.0
    floor_min_valid: float = 1.0
    floor_max_valid: float = 60.0
    num_floors_min_valid: float = 1.0
    num_floors_max_valid: float = 80.0
    rooms_min_valid: float = 0.0
    rooms_max_valid: float = 8.0
    plot_dpi: int = 140
    pre_clean_histogram_filename: str = "price_distribution_before_cleaning.png"
    post_clean_histogram_filename: str = "price_distribution_after_cleaning.png"
    pre_clean_boxplot_filename: str = "price_boxplot_before_cleaning.png"
    post_clean_boxplot_filename: str = "price_boxplot_after_cleaning.png"
    numeric_columns: tuple[str, ...] = (
        "Price",
        "Minutes to metro",
        "Number of rooms",
        "Area",
        "Living area",
        "Kitchen area",
        "Floor",
        "Number of floors",
        "Region id",
        "Geo latitude",
        "Geo longitude",
    )
    categorical_columns: tuple[str, ...] = (
        "Apartment type",
        "Metro station",
        "Region",
        "Renovation",
        "Building type",
        "Object type",
        "Date",
        "Time",
        "Source dataset",
    )
    columns_to_keep: tuple[str, ...] = (
        "Price",
        "Apartment type",
        "Metro station",
        "Minutes to metro",
        "Region",
        "Region id",
        "Number of rooms",
        "Area",
        "Living area",
        "Kitchen area",
        "Floor",
        "Number of floors",
        "Renovation",
        "Building type",
        "Object type",
        "Geo latitude",
        "Geo longitude",
        "Date",
        "Time",
        "Source dataset",
    )


@dataclass(slots=True)
class PriceModelConfig:
    """Controls supervised learning for housing price estimation."""

    train_input_filename: str = "train_data.csv"
    test_input_filename: str = "test_data.csv"
    target_column: str = "Price"
    summary_filename: str = "price_model_training_summary.json"
    feature_manifest_filename: str = "price_model_feature_manifest.csv"
    registry_table_filename: str = "price_model_training_registry.csv"
    prediction_preview_filename: str = "price_model_prediction_preview.csv"
    prediction_preview_rows: int = 150
    feature_strategy: str = "scaled_numeric_plus_frequency_encoded_categorical"
    required_feature_suffixes: tuple[str, ...] = ("_scaled", "_encoded")
    leakage_columns: tuple[str, ...] = (
        "Price",
        "Price_scaled",
        "Price_normalized",
    )
    linear_regression_params: dict[str, Any] = field(
        default_factory=lambda: {
            "fit_intercept": True,
            "positive": False,
        }
    )
    random_forest_params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 180,
            "max_depth": 18,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "bootstrap": True,
            "n_jobs": 1,
            "random_state": 42,
        }
    )
    gradient_boosting_params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 220,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.85,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "random_state": 42,
        }
    )


@dataclass(slots=True)
class ModelEvaluationConfig:
    """Controls holdout evaluation, cross-validation, and model-comparison reporting."""

    modeling_input_filename: str = "modeling_dataset.csv"
    test_input_filename: str = "test_data.csv"
    training_summary_filename: str = "price_model_training_summary.json"
    summary_filename: str = "price_model_evaluation_summary.json"
    comparison_table_filename: str = "price_model_comparison_metrics.csv"
    top_feature_count: int = 15
    cv_sample_size: int = 60_000
    plot_sample_size: int = 8_000
    plot_dpi: int = 140
    comparison_plot_filename: str = "price_model_comparison_bar_chart.png"
    predicted_vs_actual_prefix: str = "predicted_vs_actual"
    residual_plot_prefix: str = "residual_plot"
    feature_importance_prefix: str = "feature_importance"
    feature_importance_table_prefix: str = "feature_importance"
    best_model_selection_metric: str = "test_rmse"


@dataclass(slots=True)
class FeatureEngineeringConfig:
    """Controls domain-specific feature construction and feature-impact experiments."""

    input_filename: str = "clean_data.csv"
    output_filename: str = "engineered_data.csv"
    summary_filename: str = "feature_engineering_summary.json"
    district_profile_filename: str = "district_market_profile.csv"
    feature_summary_filename: str = "engineered_feature_summary.csv"
    model_impact_filename: str = "feature_engineering_model_impact.csv"
    importance_filename: str = "feature_engineering_feature_importance.csv"
    comparison_plot_filename: str = "feature_engineering_model_comparison.png"
    importance_plot_filename: str = "feature_engineering_feature_importance.png"
    chunk_size: int = 150_000
    comparison_sample_size: int = 120_000
    test_size: float = 0.2
    plot_dpi: int = 140
    engineered_numeric_columns: tuple[str, ...] = (
        "price_per_m2",
        "floor_ratio",
        "floor_comfort_score",
        "transport_score",
        "renovation_score",
        "investment_score",
        "liquidity_score",
        "district_popularity_score",
    )
    leakage_sensitive_engineered_columns: tuple[str, ...] = (
        "price_per_m2",
    )


@dataclass(slots=True)
class ContentBasedConfig:
    """Controls content-based recommendation experiments and ranking analysis."""

    input_filename: str = "engineered_data.csv"
    summary_filename: str = "content_based_summary.json"
    top_recommendations_filename: str = "content_based_top_recommendations.csv"
    overlap_filename: str = "content_based_overlap_matrix.csv"
    ranking_plot_filename: str = "content_based_ranking_comparison.png"
    sample_size: int = 100_000
    top_k: int = 15
    chunk_size: int = 150_000
    plot_dpi: int = 140
    default_budget: float = 12_000_000.0
    default_area: float = 45.0
    default_rooms: float = 2.0
    default_metro_station: str = "Котельники"
    default_renovation: str = "European-style renovation"
    default_investment_score: float = 0.65
    default_liquidity_score: float = 0.70
    weighted_similarity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "Price": 0.23,
            "Area": 0.17,
            "Number of rooms": 0.14,
            "Metro station": 0.13,
            "Renovation": 0.10,
            "investment_score": 0.12,
            "liquidity_score": 0.11,
        }
    )


@dataclass(slots=True)
class CollaborativeFilteringConfig:
    """Controls mock interaction generation and collaborative filtering experiments."""

    input_filename: str = "engineered_data.csv"
    summary_filename: str = "collaborative_summary.json"
    mock_interactions_filename: str = "mock_user_interactions.csv"
    recommendations_filename: str = "collaborative_top_recommendations.csv"
    comparison_filename: str = "collaborative_vs_content_based.csv"
    ranking_plot_filename: str = "collaborative_vs_content_based.png"
    sample_size: int = 12_000
    n_mock_users: int = 320
    interactions_per_user: int = 24
    top_k: int = 15
    chunk_size: int = 150_000
    plot_dpi: int = 140
    default_target_user_id: str = "user_0001"


@dataclass(slots=True)
class KnowledgeBasedConfig:
    """Controls rule-based recommendation experiments and explainable ranking."""

    input_filename: str = "engineered_data.csv"
    summary_filename: str = "knowledge_based_summary.json"
    recommendations_filename: str = "knowledge_based_top_recommendations.csv"
    diagnostics_filename: str = "knowledge_based_filter_diagnostics.csv"
    comparison_filename: str = "knowledge_based_vs_content_based.csv"
    funnel_plot_filename: str = "knowledge_based_filter_funnel.png"
    comparison_plot_filename: str = "knowledge_based_vs_content_based.png"
    sample_size: int = 120_000
    top_k: int = 15
    chunk_size: int = 150_000
    plot_dpi: int = 140
    default_max_budget: float = 12_000_000.0
    default_target_area: float = 45.0
    default_area_tolerance: float = 12.0
    default_target_rooms: float = 2.0
    default_room_tolerance: float = 1.0
    default_preferred_metro_station: str = "Котельники"
    default_preferred_renovation: str = "European-style renovation"
    min_price_per_m2: float = 35_000.0
    min_investment_score: float = 0.55
    min_liquidity_score: float = 0.65
    min_transport_score: float = 0.80
    min_candidates_after_hard_filters: int = 150
    min_candidates_after_optional_filters: int = 25


@dataclass(slots=True)
class HybridRecommendationConfig:
    """Controls hybrid recommendation experiments over a shared candidate universe."""

    input_filename: str = "engineered_data.csv"
    summary_filename: str = "hybrid_summary.json"
    recommendations_filename: str = "hybrid_top_recommendations.csv"
    interactions_filename: str = "hybrid_mock_interactions.csv"
    comparison_filename: str = "hybrid_method_comparison.csv"
    overlap_filename: str = "hybrid_overlap_matrix.csv"
    component_table_filename: str = "hybrid_component_contributions.csv"
    component_summary_filename: str = "hybrid_component_summary.csv"
    ranking_plot_filename: str = "hybrid_ranking_comparison.png"
    contribution_plot_filename: str = "hybrid_component_contributions.png"
    sample_size: int = 15_000
    top_k: int = 15
    chunk_size: int = 150_000
    plot_dpi: int = 140
    n_mock_users: int = 320
    interactions_per_user: int = 24
    default_budget: float = 12_000_000.0
    default_area: float = 45.0
    default_area_tolerance: float = 12.0
    default_rooms: float = 2.0
    default_room_tolerance: float = 1.0
    default_metro_station: str = "Котельники"
    default_renovation: str = "European-style renovation"
    default_investment_score: float = 0.65
    default_liquidity_score: float = 0.70
    min_price_per_m2: float = 35_000.0
    min_investment_score: float = 0.55
    min_liquidity_score: float = 0.65
    min_transport_score: float = 0.80


@dataclass(slots=True)
class RecommendationEvaluationConfig:
    """Controls offline evaluation and comparison of recommendation algorithms."""

    content_input_filename: str = "content_based_top_recommendations.csv"
    collaborative_input_filename: str = "collaborative_top_recommendations.csv"
    knowledge_input_filename: str = "knowledge_based_top_recommendations.csv"
    hybrid_input_filename: str = "hybrid_top_recommendations.csv"
    summary_filename: str = "recommendation_evaluation_summary.json"
    comparison_table_filename: str = "recommendation_algorithm_comparison.csv"
    item_level_table_filename: str = "recommendation_item_level_evaluation.csv"
    metric_heatmap_filename: str = "recommendation_metric_heatmap.png"
    overall_score_plot_filename: str = "recommendation_overall_score_comparison.png"
    top_k: int = 15
    plot_dpi: int = 140
    best_method_selection_metric: str = "dcg_utility"


@dataclass(slots=True)
class DecisionSupportConfig:
    """Controls the integrated decision-support shortlist generation stage."""

    input_filename: str = "engineered_data.csv"
    summary_filename: str = "decision_support_summary.json"
    profile_filename: str = "decision_support_profile.json"
    shortlist_filename: str = "decision_support_shortlist.csv"
    component_filename: str = "decision_support_component_breakdown.csv"
    diagnostics_filename: str = "decision_support_candidate_diagnostics.csv"
    value_gap_plot_filename: str = "decision_support_value_gap.png"
    component_plot_filename: str = "decision_support_component_scores.png"
    sample_size: int = 20_000
    top_k: int = 15
    chunk_size: int = 150_000
    plot_dpi: int = 140
    budget: float = 12_000_000.0
    desired_area: float = 45.0
    area_tolerance: float = 12.0
    desired_rooms: float = 2.0
    room_tolerance: float = 1.0
    preferred_metro_station: str = "Котельники"
    preferred_renovation: str = "European-style renovation"
    target_investment_score: float = 0.65
    target_liquidity_score: float = 0.70
    min_price_per_m2: float = 35_000.0
    min_investment_score: float = 0.55
    min_liquidity_score: float = 0.65
    min_transport_score: float = 0.80
    max_budget_slack_ratio: float = 0.05


@dataclass(slots=True)
class FinalDemoConfig:
    """Controls the final end-to-end demo packaging stage."""

    decision_support_summary_filename: str = "decision_support_summary.json"
    decision_support_profile_filename: str = "decision_support_profile.json"
    decision_support_shortlist_filename: str = "decision_support_shortlist.csv"
    price_model_summary_filename: str = "price_model_evaluation_summary.json"
    recommendation_summary_filename: str = "recommendation_evaluation_summary.json"
    summary_filename: str = "final_demo_summary.json"
    payload_filename: str = "final_demo_payload.json"
    report_filename: str = "final_decision_support_report.md"
    system_summary_table_filename: str = "final_demo_system_summary.csv"
    shortlist_preview_filename: str = "final_demo_shortlist_top5.csv"
    score_plot_filename: str = "final_demo_score_overview.png"
    valuation_plot_filename: str = "final_demo_price_vs_market.png"
    top_k_preview: int = 5
    plot_dpi: int = 140


@dataclass(slots=True)
class FinalAnalyticsConfig:
    """Controls automated assembly of the final analytical diploma chapter."""

    preprocessing_summary_filename: str = "preprocessing_summary.json"
    price_model_summary_filename: str = "price_model_evaluation_summary.json"
    recommendation_summary_filename: str = "recommendation_evaluation_summary.json"
    feature_engineering_summary_filename: str = "feature_engineering_summary.json"
    knowledge_summary_filename: str = "knowledge_based_summary.json"
    hybrid_summary_filename: str = "hybrid_summary.json"
    decision_support_summary_filename: str = "decision_support_summary.json"
    final_demo_summary_filename: str = "final_demo_summary.json"
    price_model_table_filename: str = "price_model_comparison_metrics.csv"
    recommendation_table_filename: str = "recommendation_algorithm_comparison.csv"
    feature_impact_table_filename: str = "feature_engineering_model_impact.csv"
    hybrid_component_table_filename: str = "hybrid_component_summary.csv"
    shortlist_filename: str = "decision_support_shortlist.csv"
    summary_filename: str = "final_analytics_summary.json"
    report_filename: str = "final_analytical_chapter.md"
    findings_table_filename: str = "final_key_findings.csv"
    limitations_table_filename: str = "final_system_limitations.csv"
    stage_outcomes_table_filename: str = "final_stage_outcomes.csv"
    price_model_plot_filename: str = "final_price_model_overview.png"
    recommendation_plot_filename: str = "final_recommendation_overview.png"
    shortlist_plot_filename: str = "final_shortlist_discount_overview.png"
    shortlist_preview_rows: int = 5
    plot_dpi: int = 140


@dataclass(slots=True)
class RecommendationWeights:
    """Default hybrid recommendation weights used as a baseline in experiments."""

    price: float = 0.20
    transport: float = 0.12
    renovation: float = 0.10
    investment: float = 0.16
    floor: float = 0.08
    liquidity: float = 0.12
    similarity: float = 0.14
    collaborative: float = 0.08

    def as_dict(self) -> dict[str, float]:
        return {
            "price": self.price,
            "transport": self.transport,
            "renovation": self.renovation,
            "investment": self.investment,
            "floor": self.floor,
            "liquidity": self.liquidity,
            "similarity": self.similarity,
            "collaborative": self.collaborative,
        }


@dataclass(slots=True)
class AppConfig:
    """Aggregate configuration object for the entire diploma project."""

    paths: PathConfig
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)
    eda: EDAConfig = field(default_factory=EDAConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    price_modeling: PriceModelConfig = field(default_factory=PriceModelConfig)
    model_evaluation: ModelEvaluationConfig = field(default_factory=ModelEvaluationConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    content_based: ContentBasedConfig = field(default_factory=ContentBasedConfig)
    collaborative: CollaborativeFilteringConfig = field(default_factory=CollaborativeFilteringConfig)
    knowledge_based: KnowledgeBasedConfig = field(default_factory=KnowledgeBasedConfig)
    hybrid: HybridRecommendationConfig = field(default_factory=HybridRecommendationConfig)
    recommendation_evaluation: RecommendationEvaluationConfig = field(default_factory=RecommendationEvaluationConfig)
    decision_support: DecisionSupportConfig = field(default_factory=DecisionSupportConfig)
    final_demo: FinalDemoConfig = field(default_factory=FinalDemoConfig)
    final_analytics: FinalAnalyticsConfig = field(default_factory=FinalAnalyticsConfig)
    recommendation: RecommendationWeights = field(default_factory=RecommendationWeights)
    project_name: str = "Housing Decision Support System"
    version: str = "0.16.0"
    author: str = "Codex + Stefan"

    @classmethod
    def from_root(cls, project_root: Path) -> "AppConfig":
        return cls(paths=PathConfig(project_root=project_root))

    def required_directories(self) -> Iterable[Path]:
        return self.paths.required_directories()
