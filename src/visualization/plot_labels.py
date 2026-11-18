from __future__ import annotations

from textwrap import fill
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import ticker


MODEL_LABELS = {
    "Linear Regression": "Linear Regression",
    "Random Forest Regressor": "Random Forest",
    "Gradient Boosting Regressor": "Gradient Boosting",
}

METHOD_LABELS = {
    "cosine_similarity": "Cosine similarity",
    "euclidean_similarity": "Euclidean similarity",
    "manhattan_similarity": "Manhattan similarity",
    "weighted_similarity": "Weighted similarity",
    "content_cosine": "Content (cosine)",
    "content_euclidean": "Content (Euclidean)",
    "content_manhattan": "Content (Manhattan)",
    "content_weighted": "Content (weighted)",
    "content_based_reference": "Content-based",
    "user_based_cf": "User-based CF",
    "item_based_cf": "Item-based CF",
    "collaborative_user_based": "User-based CF",
    "collaborative_item_based": "Item-based CF",
    "knowledge_based": "Knowledge-based",
    "knowledge_based_reference": "Knowledge-based",
    "collaborative_reference": "Collaborative filtering",
    "hybrid": "Hybrid",
    "hybrid_recommendation": "Hybrid recommendation",
}

FEATURE_LABELS = {
    "Price": "Price",
    "Minutes to metro": "Minutes to metro",
    "Number of rooms": "Rooms",
    "Area": "Total area",
    "Living area": "Living area",
    "Kitchen area": "Kitchen area",
    "Floor": "Floor",
    "Number of floors": "Building floors",
    "Region id": "Region ID",
    "Geo latitude": "Latitude",
    "Geo longitude": "Longitude",
    "Apartment type": "Apartment type",
    "Metro station": "Metro station",
    "Region": "Region",
    "Renovation": "Renovation",
    "Building type": "Building type",
    "Object type": "Property type",
    "Date": "Listing date",
    "Time": "Listing time",
    "Source dataset": "Source dataset",
    "price_per_m2": "Price per m²",
    "floor_ratio": "Floor ratio",
    "floor_comfort_score": "Floor comfort score",
    "transport_score": "Transport score",
    "renovation_score": "Renovation score",
    "investment_score": "Investment score",
    "liquidity_score": "Liquidity score",
    "district_popularity_score": "District popularity score",
    "actual_price": "Actual price",
    "predicted_price": "Predicted price",
    "residual": "Residual",
    "relative_discount_to_market": "Discount to market",
    "decision_support_score": "Decision support score",
    "preference_score": "Preference score",
    "valuation_score": "Valuation score",
    "investment_score_component": "Investment score",
    "liquidity_score_component": "Liquidity score",
    "rank": "Rank",
}

COMPONENT_LABELS = {
    "Price": "Price",
    "Transport": "Transport",
    "Renovation": "Renovation",
    "Investment": "Investment",
    "Floor": "Floor",
    "Liquidity": "Liquidity",
    "Content similarity": "Content similarity",
    "Collaborative": "Collaborative signal",
    "price_component": "Price",
    "transport_component": "Transport",
    "renovation_component": "Renovation",
    "investment_component": "Investment",
    "floor_component": "Floor",
    "liquidity_component": "Liquidity",
    "similarity_component": "Content similarity",
    "collaborative_component": "Collaborative signal",
    "preference_score": "Preference score",
    "valuation_score": "Valuation score",
    "investment_score_component": "Investment score",
    "liquidity_score_component": "Liquidity score",
}

RULE_STAGE_LABELS = {
    "initial_pool": "Initial pool",
    "max_budget": "Budget limit",
    "target_area_band": "Area range",
    "target_rooms_band": "Room range",
    "minimum_price_per_m2": "Minimum price per m²",
    "minimum_investment_score": "Minimum investment score",
    "minimum_liquidity_score": "Minimum liquidity score",
    "minimum_transport_score": "Minimum transport score",
    "preferred_metro_station": "Preferred metro station",
    "preferred_renovation": "Preferred renovation",
    "known_metro_value": "Known metro station",
    "known_renovation_value": "Known renovation",
    "optional_rules_skipped": "Optional rules skipped",
}

METRIC_LABELS = {
    "avg_item_utility": "Average item utility",
    "top1_utility": "Top-1 utility",
    "dcg_utility": "DCG utility",
    "hard_constraint_rate": "Hard-constraint rate",
    "metro_match_rate": "Metro match rate",
    "renovation_match_rate": "Renovation match rate",
    "known_metro_rate": "Known metro rate",
    "known_renovation_rate": "Known renovation rate",
    "test_mae": "Test MAE",
    "test_rmse": "Test RMSE",
    "test_mse": "Test MSE",
    "test_r2": "Test R²",
}

SECTION_LABELS = {
    "intro": "Introduction",
    "core": "Main section",
    "results": "Results",
    "qa": "Questions",
    "Intro": "Introduction",
    "Data": "Data",
    "Modeling": "Modeling",
    "Recommendation": "Recommendations",
    "Decision Support": "Decision support",
    "Limitations": "Limitations",
    "Conclusion": "Conclusion",
}

FEATURE_SET_LABELS = {"baseline": "Baseline features", "enhanced": "Engineered features"}
BOOLEAN_LABELS = {True: "Yes", False: "No"}


def setup_plot_style() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def _label(name: Any, labels: dict[Any, str]) -> str:
    return labels.get(name, str(name))


def model_label(name: Any) -> str:
    return _label(str(name), MODEL_LABELS)


def method_label(name: Any) -> str:
    return _label(str(name), METHOD_LABELS)


def feature_label(name: Any) -> str:
    raw_name = str(name)
    if raw_name in FEATURE_LABELS:
        return FEATURE_LABELS[raw_name]
    for suffix in ("_scaled", "_encoded"):
        if raw_name.endswith(suffix):
            base_name = raw_name.removesuffix(suffix)
            return FEATURE_LABELS.get(base_name, base_name.replace("_", " ").title())
    return raw_name.replace("_", " ").title()


def component_label(name: Any) -> str:
    return _label(str(name), COMPONENT_LABELS)


def rule_stage_label(name: Any) -> str:
    return _label(str(name), RULE_STAGE_LABELS)


def metric_label(name: Any) -> str:
    return _label(str(name), METRIC_LABELS)


def column_label(name: Any) -> str:
    return feature_label(name)


def price_type_label(name: Any) -> str:
    return {
        "Price": "Asking price",
        "predicted_market_price": "Predicted market price",
    }.get(str(name), str(name))


def section_label(name: Any) -> str:
    return _label(str(name), SECTION_LABELS)


def feature_set_label(name: Any) -> str:
    return _label(str(name), FEATURE_SET_LABELS)


def boolean_label(value: Any) -> str:
    return _label(bool(value), BOOLEAN_LABELS)


def format_numeric_axis(axis: Any, *, x: bool = False, y: bool = False, nbins: int = 6) -> None:
    def apply(target_axis: Any) -> None:
        formatter = ticker.ScalarFormatter(useMathText=False)
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        target_axis.set_major_locator(ticker.MaxNLocator(nbins=nbins))
        target_axis.set_major_formatter(formatter)

    if x:
        apply(axis.xaxis)
    if y:
        apply(axis.yaxis)


def wrap_ticklabels(
    axis: Any,
    *,
    axis_name: str = "x",
    width: int = 16,
    rotation: float = 0,
    ha: str = "center",
    fontsize: float | None = None,
) -> None:
    figure = axis.get_figure()
    figure.canvas.draw()
    labels = axis.get_xticklabels() if axis_name == "x" else axis.get_yticklabels()
    wrapped = [
        fill(str(label.get_text()), width=width, break_long_words=False, break_on_hyphens=False)
        for label in labels
    ]
    ticks = axis.get_xticks() if axis_name == "x" else axis.get_yticks()
    kwargs: dict[str, Any] = {"rotation": rotation, "ha": ha}
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    if axis_name == "x":
        axis.set_xticks(ticks)
        axis.set_xticklabels(wrapped, **kwargs)
    else:
        axis.set_yticks(ticks)
        axis.set_yticklabels(wrapped, **kwargs)
