from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

from config.config import AppConfig, ModelEvaluationConfig
from src.models.price_model import (
    BasePriceRegressor,
    GradientBoostingPriceModel,
    LinearRegressionPriceModel,
    RandomForestPriceModel,
)
from src.utils.helpers import serialize_to_json
from src.utils.logger import get_logger
from src.visualization.model_visualizer import ModelEvaluationVisualizer


@dataclass(slots=True)
class EvaluationModelSpec:
    """Describes how a trained model should be loaded and re-evaluated."""

    model_name: str
    model_class: type[BasePriceRegressor]
    artifact_path: Path
    hyperparameters: dict[str, Any]


@dataclass(slots=True)
class ModelEvaluationRecord:
    """Evaluation metrics and diagnostics for a single regression model."""

    model_name: str
    test_mae: float
    test_rmse: float
    test_mse: float
    test_r2: float
    cv_mae_mean: float
    cv_mae_std: float
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_mse_mean: float
    cv_mse_std: float
    cv_r2_mean: float
    cv_r2_std: float
    artifact_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "test_mae": self.test_mae,
            "test_rmse": self.test_rmse,
            "test_mse": self.test_mse,
            "test_r2": self.test_r2,
            "cv_mae_mean": self.cv_mae_mean,
            "cv_mae_std": self.cv_mae_std,
            "cv_rmse_mean": self.cv_rmse_mean,
            "cv_rmse_std": self.cv_rmse_std,
            "cv_mse_mean": self.cv_mse_mean,
            "cv_mse_std": self.cv_mse_std,
            "cv_r2_mean": self.cv_r2_mean,
            "cv_r2_std": self.cv_r2_std,
            "artifact_path": self.artifact_path,
        }


@dataclass(slots=True)
class PriceModelEvaluationSummary:
    """Global output metadata for the evaluation and comparison stage."""

    test_input_path: str
    modeling_input_path: str
    best_model_name: str
    best_model_selection_metric: str
    feature_count: int
    feature_columns: list[str]
    comparison_table_path: str
    feature_importance_table_paths: list[str]
    plots_created: list[str]


class PriceModelEvaluator:
    """Evaluates trained price models on holdout data and via cross-validation."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.evaluation_config: ModelEvaluationConfig = config.model_evaluation
        self.price_model_config = config.price_modeling
        self.logger = get_logger(self.__class__.__name__)
        self.modeling_input_path = self.config.paths.processed_data_dir / self.evaluation_config.modeling_input_filename
        self.test_input_path = self.config.paths.processed_data_dir / self.evaluation_config.test_input_filename
        self.training_summary_path = self.config.paths.metadata_dir / self.evaluation_config.training_summary_filename
        self.visualizer = ModelEvaluationVisualizer(self.config.paths.plots_dir, self.evaluation_config)

    def execute(self) -> PriceModelEvaluationSummary:
        """Run holdout evaluation, cross-validation, feature analysis, and visualization."""

        self._assert_required_inputs()
        training_summary = self._load_training_summary()
        feature_columns = list(training_summary["feature_columns"])

        test_frame = pd.read_csv(self.test_input_path, low_memory=False)
        modeling_frame = pd.read_csv(self.modeling_input_path, low_memory=False)
        test_features, test_target = self._build_feature_matrix(test_frame, feature_columns)
        cv_frame = self._prepare_cv_frame(modeling_frame)
        cv_features, cv_target = self._build_feature_matrix(cv_frame, feature_columns)

        model_specs = self._build_model_specs(training_summary)
        records: list[ModelEvaluationRecord] = []
        plots_created: list[str] = []
        feature_importance_table_paths: list[str] = []

        for model_spec in model_specs:
            record, feature_importance_table_path, model_plots = self._evaluate_single_model(
                model_spec=model_spec,
                feature_columns=feature_columns,
                test_features=test_features,
                test_target=test_target,
                cv_features=cv_features,
                cv_target=cv_target,
            )
            records.append(record)
            feature_importance_table_paths.append(feature_importance_table_path)
            plots_created.extend(model_plots)

        comparison_frame = pd.DataFrame([record.as_dict() for record in records])
        comparison_frame = comparison_frame.sort_values(
            by=self.evaluation_config.best_model_selection_metric,
            ascending=True,
        ).reset_index(drop=True)
        comparison_table_path = self.config.paths.tables_dir / self.evaluation_config.comparison_table_filename
        comparison_frame.to_csv(comparison_table_path, index=False, encoding="utf-8-sig")

        comparison_plot_path = self.visualizer.plot_model_comparison(
            comparison_frame=comparison_frame,
            output_filename=self.evaluation_config.comparison_plot_filename,
        )
        plots_created.append(str(comparison_plot_path))

        best_model_name = str(comparison_frame.iloc[0]["model_name"])
        summary = PriceModelEvaluationSummary(
            test_input_path=str(self.test_input_path),
            modeling_input_path=str(self.modeling_input_path),
            best_model_name=best_model_name,
            best_model_selection_metric=self.evaluation_config.best_model_selection_metric,
            feature_count=len(feature_columns),
            feature_columns=feature_columns,
            comparison_table_path=str(comparison_table_path),
            feature_importance_table_paths=feature_importance_table_paths,
            plots_created=plots_created,
        )
        serialize_to_json(
            summary,
            self.config.paths.metadata_dir / self.evaluation_config.summary_filename,
        )
        self.logger.info(
            "Stage 6 completed. Best model by %s: %s.",
            self.evaluation_config.best_model_selection_metric,
            best_model_name,
        )
        return summary

    def _assert_required_inputs(self) -> None:
        missing_paths = [
            path
            for path in (self.modeling_input_path, self.test_input_path, self.training_summary_path)
            if not path.exists()
        ]
        if missing_paths:
            missing_labels = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(
                f"Required evaluation inputs are missing: {missing_labels}. Run stages 4-5 first."
            )

    def _load_training_summary(self) -> dict[str, Any]:
        with self.training_summary_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        return dict(payload)

    def _prepare_cv_frame(self, modeling_frame: pd.DataFrame) -> pd.DataFrame:
        sample_size = min(self.evaluation_config.cv_sample_size, len(modeling_frame))
        sampled_frame = modeling_frame.sample(
            n=sample_size,
            random_state=self.config.experiments.random_state,
        ).reset_index(drop=True)
        self.logger.info("Prepared cross-validation sample with %s rows.", len(sampled_frame))
        return sampled_frame

    def _build_model_specs(self, training_summary: dict[str, Any]) -> list[EvaluationModelSpec]:
        artifact_paths = {
            Path(path_string).name: Path(path_string)
            for path_string in training_summary["artifact_paths"]
        }
        return [
            EvaluationModelSpec(
                model_name="Linear Regression",
                model_class=LinearRegressionPriceModel,
                artifact_path=artifact_paths["linear_regression_price_model.joblib"],
                hyperparameters=self.price_model_config.linear_regression_params,
            ),
            EvaluationModelSpec(
                model_name="Random Forest Regressor",
                model_class=RandomForestPriceModel,
                artifact_path=artifact_paths["random_forest_regressor_price_model.joblib"],
                hyperparameters=self.price_model_config.random_forest_params,
            ),
            EvaluationModelSpec(
                model_name="Gradient Boosting Regressor",
                model_class=GradientBoostingPriceModel,
                artifact_path=artifact_paths["gradient_boosting_regressor_price_model.joblib"],
                hyperparameters=self.price_model_config.gradient_boosting_params,
            ),
        ]

    def _build_feature_matrix(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
    ) -> tuple[pd.DataFrame, pd.Series]:
        features = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
        target = pd.to_numeric(frame[self.price_model_config.target_column], errors="coerce")

        if features.isna().any().any():
            missing_total = int(features.isna().sum().sum())
            raise ValueError(f"Evaluation feature matrix contains {missing_total} missing values.")
        if target.isna().any():
            missing_target = int(target.isna().sum())
            raise ValueError(f"Evaluation target vector contains {missing_target} missing values.")
        return features, target

    def _evaluate_single_model(
        self,
        model_spec: EvaluationModelSpec,
        feature_columns: list[str],
        test_features: pd.DataFrame,
        test_target: pd.Series,
        cv_features: pd.DataFrame,
        cv_target: pd.Series,
    ) -> tuple[ModelEvaluationRecord, str, list[str]]:
        loaded_model = model_spec.model_class.load(model_spec.artifact_path)
        predictions = loaded_model.predict(test_features)
        holdout_metrics = self._compute_holdout_metrics(test_target, predictions)
        cv_metrics = self._run_cross_validation(model_spec, cv_features, cv_target)

        importance_frame = self._extract_feature_importance(
            loaded_model=loaded_model,
            feature_columns=feature_columns,
        )
        importance_table_path = self._save_feature_importance_table(
            model_name=model_spec.model_name,
            importance_frame=importance_frame,
        )

        plot_frame = self._build_plot_frame(
            actual=test_target,
            predicted=predictions,
        )
        model_plots = self._create_model_plots(
            model_name=model_spec.model_name,
            plot_frame=plot_frame,
            importance_frame=importance_frame,
        )

        record = ModelEvaluationRecord(
            model_name=model_spec.model_name,
            test_mae=holdout_metrics["mae"],
            test_rmse=holdout_metrics["rmse"],
            test_mse=holdout_metrics["mse"],
            test_r2=holdout_metrics["r2"],
            cv_mae_mean=cv_metrics["mae_mean"],
            cv_mae_std=cv_metrics["mae_std"],
            cv_rmse_mean=cv_metrics["rmse_mean"],
            cv_rmse_std=cv_metrics["rmse_std"],
            cv_mse_mean=cv_metrics["mse_mean"],
            cv_mse_std=cv_metrics["mse_std"],
            cv_r2_mean=cv_metrics["r2_mean"],
            cv_r2_std=cv_metrics["r2_std"],
            artifact_path=str(model_spec.artifact_path),
        )
        return record, str(importance_table_path), model_plots

    def _compute_holdout_metrics(self, actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
        mse = float(mean_squared_error(actual, predicted))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(actual, predicted))
        r2 = float(r2_score(actual, predicted))
        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mse": round(mse, 4),
            "r2": round(r2, 6),
        }

    def _run_cross_validation(
        self,
        model_spec: EvaluationModelSpec,
        cv_features: pd.DataFrame,
        cv_target: pd.Series,
    ) -> dict[str, float]:
        model_wrapper = model_spec.model_class(
            target_column=self.price_model_config.target_column,
            hyperparameters=model_spec.hyperparameters,
        )
        estimator = model_wrapper.build_estimator()
        splitter = KFold(
            n_splits=self.config.experiments.n_splits_cv,
            shuffle=True,
            random_state=self.config.experiments.random_state,
        )
        scoring = {
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "mse": make_scorer(mean_squared_error, greater_is_better=False),
            "rmse": make_scorer(self._rmse_metric, greater_is_better=False),
            "r2": make_scorer(r2_score),
        }
        results = cross_validate(
            estimator=estimator,
            X=cv_features,
            y=cv_target,
            scoring=scoring,
            cv=splitter,
            n_jobs=1,
            return_train_score=False,
        )
        metrics = {
            "mae_mean": round(float(-np.mean(results["test_mae"])), 4),
            "mae_std": round(float(np.std(-results["test_mae"])), 4),
            "mse_mean": round(float(-np.mean(results["test_mse"])), 4),
            "mse_std": round(float(np.std(-results["test_mse"])), 4),
            "rmse_mean": round(float(-np.mean(results["test_rmse"])), 4),
            "rmse_std": round(float(np.std(-results["test_rmse"])), 4),
            "r2_mean": round(float(np.mean(results["test_r2"])), 6),
            "r2_std": round(float(np.std(results["test_r2"])), 6),
        }
        self.logger.info(
            "Cross-validation completed for %s: RMSE mean=%s.",
            model_spec.model_name,
            metrics["rmse_mean"],
        )
        return metrics

    def _extract_feature_importance(
        self,
        loaded_model: BasePriceRegressor,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        estimator = loaded_model.estimator
        if estimator is None:
            raise RuntimeError(f"Loaded model '{loaded_model.model_name}' has no estimator.")

        if hasattr(estimator, "feature_importances_"):
            raw_importance = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            raw_importance = np.abs(np.ravel(estimator.coef_)).astype(float)
        else:
            raise ValueError(f"Estimator '{loaded_model.model_name}' does not expose feature importance.")

        relative_importance = raw_importance / raw_importance.sum() if raw_importance.sum() else raw_importance
        importance_frame = pd.DataFrame(
            {
                "feature_name": feature_columns,
                "raw_importance": raw_importance,
                "relative_importance": relative_importance,
            }
        )
        importance_frame = importance_frame.sort_values(by="relative_importance", ascending=False).reset_index(drop=True)
        return importance_frame.head(self.evaluation_config.top_feature_count)

    def _save_feature_importance_table(
        self,
        model_name: str,
        importance_frame: pd.DataFrame,
    ) -> Path:
        sanitized = self._sanitize_model_name(model_name)
        output_path = (
            self.config.paths.tables_dir
            / f"{self.evaluation_config.feature_importance_table_prefix}_{sanitized}.csv"
        )
        importance_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path

    def _build_plot_frame(
        self,
        actual: pd.Series,
        predicted: np.ndarray,
    ) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "actual_price": actual.to_numpy(dtype=float),
                "predicted_price": predicted,
            }
        )
        frame["residual"] = frame["actual_price"] - frame["predicted_price"]
        sample_size = min(self.evaluation_config.plot_sample_size, len(frame))
        return frame.sample(
            n=sample_size,
            random_state=self.config.experiments.random_state,
        ).reset_index(drop=True)

    def _create_model_plots(
        self,
        model_name: str,
        plot_frame: pd.DataFrame,
        importance_frame: pd.DataFrame,
    ) -> list[str]:
        sanitized = self._sanitize_model_name(model_name)
        predicted_vs_actual_path = self.visualizer.plot_predicted_vs_actual(
            model_name=model_name,
            plot_frame=plot_frame,
            output_filename=f"{self.evaluation_config.predicted_vs_actual_prefix}_{sanitized}.png",
        )
        residual_plot_path = self.visualizer.plot_residuals(
            model_name=model_name,
            plot_frame=plot_frame,
            output_filename=f"{self.evaluation_config.residual_plot_prefix}_{sanitized}.png",
        )
        feature_importance_path = self.visualizer.plot_feature_importance(
            model_name=model_name,
            importance_frame=importance_frame,
            output_filename=f"{self.evaluation_config.feature_importance_prefix}_{sanitized}.png",
        )
        return [
            str(predicted_vs_actual_path),
            str(residual_plot_path),
            str(feature_importance_path),
        ]

    @staticmethod
    def _sanitize_model_name(model_name: str) -> str:
        return model_name.lower().replace(" ", "_")

    @staticmethod
    def _rmse_metric(actual: np.ndarray, predicted: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(actual, predicted)))
