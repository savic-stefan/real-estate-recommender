from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Self

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from config.config import AppConfig, PriceModelConfig
from src.utils.helpers import serialize_to_json
from src.utils.logger import get_logger


@dataclass(slots=True)
class ModelTrainingRecord:
    """Training metadata for a single fitted regressor."""

    model_name: str
    estimator_class: str
    train_rows: int
    test_rows: int
    feature_count: int
    training_seconds: float
    prediction_mean: float
    prediction_std: float
    prediction_min: float
    prediction_max: float
    reload_verification_max_abs_diff: float
    artifact_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "estimator_class": self.estimator_class,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "feature_count": self.feature_count,
            "training_seconds": self.training_seconds,
            "prediction_mean": self.prediction_mean,
            "prediction_std": self.prediction_std,
            "prediction_min": self.prediction_min,
            "prediction_max": self.prediction_max,
            "reload_verification_max_abs_diff": self.reload_verification_max_abs_diff,
            "artifact_path": self.artifact_path,
        }


@dataclass(slots=True)
class PriceModelTrainingSummary:
    """Global output metadata for the supervised price modeling stage."""

    train_input_path: str
    test_input_path: str
    target_column: str
    feature_strategy: str
    feature_count: int
    feature_columns: list[str]
    trained_models: list[str]
    artifact_paths: list[str]
    tables_created: list[str]


class BasePriceRegressor(ABC):
    """Abstract wrapper that standardizes training, inference, and persistence."""

    def __init__(
        self,
        model_name: str,
        target_column: str,
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.target_column = target_column
        self.hyperparameters = hyperparameters or {}
        self.estimator: Any | None = None
        self.feature_columns: list[str] = []
        self.training_context: dict[str, Any] = {}
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def _build_estimator(self) -> Any:
        """Instantiate the underlying scikit-learn regressor."""

    def build_estimator(self) -> Any:
        """Expose a fresh unfitted estimator for evaluation workflows."""

        return self._build_estimator()

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "BasePriceRegressor":
        """Fit the underlying estimator on a validated feature matrix."""

        validated_features = self._validate_feature_frame(features)
        validated_target = pd.to_numeric(target, errors="raise")
        self.estimator = self.build_estimator()
        self.estimator.fit(validated_features, validated_target)
        self.feature_columns = validated_features.columns.tolist()
        self.training_context.update(
            {
                "feature_count": len(self.feature_columns),
                "training_rows": len(validated_features),
            }
        )
        self.logger.info(
            "Fitted %s on %s rows and %s features.",
            self.model_name,
            len(validated_features),
            len(self.feature_columns),
        )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions on a feature frame aligned to the training schema."""

        if self.estimator is None:
            raise RuntimeError(f"Model '{self.model_name}' is not trained or loaded.")
        validated_features = self._validate_feature_frame(features)
        return np.asarray(self.estimator.predict(validated_features), dtype=float)

    def save(self, artifact_path: Path) -> Path:
        """Persist estimator, feature schema, and metadata as a joblib artifact."""

        if self.estimator is None:
            raise RuntimeError(f"Cannot save '{self.model_name}' before training.")

        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "target_column": self.target_column,
            "hyperparameters": self.hyperparameters,
            "feature_columns": self.feature_columns,
            "training_context": self.training_context,
            "estimator": self.estimator,
        }
        joblib.dump(payload, artifact_path)
        self.logger.info("Saved model artifact: %s", artifact_path.name)
        return artifact_path

    @classmethod
    def load(cls, artifact_path: Path) -> Self:
        """Restore a trained wrapper instance from a saved artifact."""

        payload = joblib.load(artifact_path)
        instance = cls(
            target_column=str(payload["target_column"]),
            hyperparameters=dict(payload["hyperparameters"]),
        )
        instance.feature_columns = list(payload["feature_columns"])
        instance.training_context = dict(payload["training_context"])
        instance.estimator = payload["estimator"]
        instance.logger.info("Loaded model artifact: %s", artifact_path.name)
        return instance

    def _validate_feature_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        """Ensure the feature frame is numeric, aligned, and complete."""

        frame = features.copy()
        if self.feature_columns:
            missing_columns = [column for column in self.feature_columns if column not in frame.columns]
            if missing_columns:
                raise ValueError(
                    f"Feature frame for '{self.model_name}' is missing columns: {missing_columns}"
                )
            frame = frame.loc[:, self.feature_columns]

        frame = frame.apply(pd.to_numeric, errors="coerce")
        if frame.isna().any().any():
            missing_total = int(frame.isna().sum().sum())
            raise ValueError(
                f"Feature frame for '{self.model_name}' contains {missing_total} missing values."
            )
        return frame


class LinearRegressionPriceModel(BasePriceRegressor):
    """Linear baseline model for interpretable market price estimation."""

    def __init__(
        self,
        target_column: str,
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        default_params = {"fit_intercept": True, "positive": False}
        merged = {**default_params, **(hyperparameters or {})}
        super().__init__("Linear Regression", target_column, merged)

    def _build_estimator(self) -> LinearRegression:
        return LinearRegression(**self.hyperparameters)


class RandomForestPriceModel(BasePriceRegressor):
    """Nonlinear ensemble model for robust tabular price estimation."""

    def __init__(
        self,
        target_column: str,
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        default_params = {
            "n_estimators": 180,
            "max_depth": 18,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "bootstrap": True,
            "n_jobs": 1,
            "random_state": 42,
        }
        merged = {**default_params, **(hyperparameters or {})}
        super().__init__("Random Forest Regressor", target_column, merged)

    def _build_estimator(self) -> RandomForestRegressor:
        return RandomForestRegressor(**self.hyperparameters)


class GradientBoostingPriceModel(BasePriceRegressor):
    """Boosted tree model capturing complex nonlinear price drivers."""

    def __init__(
        self,
        target_column: str,
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        default_params = {
            "n_estimators": 220,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.85,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "random_state": 42,
        }
        merged = {**default_params, **(hyperparameters or {})}
        super().__init__("Gradient Boosting Regressor", target_column, merged)

    def _build_estimator(self) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(**self.hyperparameters)


class PriceModelTrainer:
    """Orchestrates supervised training of multiple housing price models."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model_config: PriceModelConfig = config.price_modeling
        self.logger = get_logger(self.__class__.__name__)
        self.train_input_path = self.config.paths.processed_data_dir / self.model_config.train_input_filename
        self.test_input_path = self.config.paths.processed_data_dir / self.model_config.test_input_filename

    def execute(self) -> PriceModelTrainingSummary:
        """Train, persist, reload, and register all configured regressors."""

        self._assert_required_inputs()
        train_frame = pd.read_csv(self.train_input_path, low_memory=False)
        test_frame = pd.read_csv(self.test_input_path, low_memory=False)
        feature_columns = self._resolve_feature_columns(train_frame, test_frame)
        train_features, train_target = self._build_feature_matrix(train_frame, feature_columns)
        test_features, test_target = self._build_feature_matrix(test_frame, feature_columns)

        feature_manifest = self._build_feature_manifest(feature_columns)
        feature_manifest_path = self.config.paths.tables_dir / self.model_config.feature_manifest_filename
        feature_manifest.to_csv(feature_manifest_path, index=False, encoding="utf-8-sig")

        model_specs = self._initialize_models()
        training_records: list[ModelTrainingRecord] = []
        prediction_previews: list[pd.DataFrame] = []
        artifact_paths: list[str] = []

        for model in model_specs:
            record, preview = self._train_and_persist_model(
                model=model,
                train_features=train_features,
                train_target=train_target,
                test_features=test_features,
                test_target=test_target,
            )
            training_records.append(record)
            prediction_previews.append(preview)
            artifact_paths.append(record.artifact_path)

        registry_frame = pd.DataFrame([record.as_dict() for record in training_records])
        registry_path = self.config.paths.tables_dir / self.model_config.registry_table_filename
        registry_frame.to_csv(registry_path, index=False, encoding="utf-8-sig")

        prediction_preview_frame = pd.concat(prediction_previews, ignore_index=True)
        prediction_preview_path = self.config.paths.tables_dir / self.model_config.prediction_preview_filename
        prediction_preview_frame.to_csv(prediction_preview_path, index=False, encoding="utf-8-sig")

        summary = PriceModelTrainingSummary(
            train_input_path=str(self.train_input_path),
            test_input_path=str(self.test_input_path),
            target_column=self.model_config.target_column,
            feature_strategy=self.model_config.feature_strategy,
            feature_count=len(feature_columns),
            feature_columns=feature_columns,
            trained_models=[record.model_name for record in training_records],
            artifact_paths=artifact_paths,
            tables_created=[
                str(feature_manifest_path),
                str(registry_path),
                str(prediction_preview_path),
            ],
        )
        serialize_to_json(
            summary,
            self.config.paths.metadata_dir / self.model_config.summary_filename,
        )
        self.logger.info(
            "Stage 5 completed: trained %s models on %s features.",
            len(summary.trained_models),
            summary.feature_count,
        )
        return summary

    def _assert_required_inputs(self) -> None:
        if not self.train_input_path.exists():
            raise FileNotFoundError(
                f"Training dataset was not found at {self.train_input_path}. Run stage 4 first."
            )
        if not self.test_input_path.exists():
            raise FileNotFoundError(
                f"Test dataset was not found at {self.test_input_path}. Run stage 4 first."
            )

    def _initialize_models(self) -> list[BasePriceRegressor]:
        return [
            LinearRegressionPriceModel(
                target_column=self.model_config.target_column,
                hyperparameters=self.model_config.linear_regression_params,
            ),
            RandomForestPriceModel(
                target_column=self.model_config.target_column,
                hyperparameters=self.model_config.random_forest_params,
            ),
            GradientBoostingPriceModel(
                target_column=self.model_config.target_column,
                hyperparameters=self.model_config.gradient_boosting_params,
            ),
        ]

    def _resolve_feature_columns(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
    ) -> list[str]:
        """Select the model feature space while excluding target leakage columns."""

        shared_columns = [column for column in train_frame.columns if column in test_frame.columns]
        selected_columns = [
            column for column in shared_columns
            if self._is_model_feature(column)
        ]
        if not selected_columns:
            raise ValueError("No eligible feature columns were found for supervised modeling.")

        self.logger.info(
            "Resolved %s model feature columns using strategy '%s'.",
            len(selected_columns),
            self.model_config.feature_strategy,
        )
        return selected_columns

    def _is_model_feature(self, column_name: str) -> bool:
        if column_name in self.model_config.leakage_columns:
            return False
        return column_name.endswith(self.model_config.required_feature_suffixes)

    def _build_feature_matrix(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Extract aligned feature matrix and numeric target vector."""

        features = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
        target = pd.to_numeric(frame[self.model_config.target_column], errors="coerce")

        if features.isna().any().any():
            missing_total = int(features.isna().sum().sum())
            raise ValueError(f"Feature matrix contains {missing_total} missing values.")
        if target.isna().any():
            missing_target = int(target.isna().sum())
            raise ValueError(f"Target vector contains {missing_target} missing values.")

        return features, target

    def _build_feature_manifest(self, feature_columns: list[str]) -> pd.DataFrame:
        """Describe the selected feature space for reporting and reproducibility."""

        records: list[dict[str, Any]] = []
        for feature_name in feature_columns:
            if feature_name.endswith("_scaled"):
                feature_group = "standardized_numeric"
            elif feature_name.endswith("_encoded"):
                feature_group = "frequency_encoded_categorical"
            else:
                feature_group = "other"
            records.append(
                {
                    "feature_name": feature_name,
                    "feature_group": feature_group,
                    "feature_strategy": self.model_config.feature_strategy,
                }
            )
        return pd.DataFrame(records)

    def _train_and_persist_model(
        self,
        model: BasePriceRegressor,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        test_features: pd.DataFrame,
        test_target: pd.Series,
    ) -> tuple[ModelTrainingRecord, pd.DataFrame]:
        """Train a single model, save it, reload it, and produce a preview table."""

        started_at = perf_counter()
        model.fit(train_features, train_target)
        predictions = model.predict(test_features)
        training_seconds = round(perf_counter() - started_at, 4)

        artifact_filename = self._artifact_filename_for(model)
        artifact_path = self.config.paths.models_dir / artifact_filename
        model.training_context.update(
            {
                "test_rows": len(test_features),
                "training_seconds": training_seconds,
                "feature_strategy": self.model_config.feature_strategy,
            }
        )
        model.save(artifact_path)

        reload_verification_diff = self._verify_reloaded_model(model, artifact_path, test_features)
        preview_frame = self._build_prediction_preview(
            model_name=model.model_name,
            actual=test_target,
            predicted=predictions,
        )

        record = ModelTrainingRecord(
            model_name=model.model_name,
            estimator_class=model.estimator.__class__.__name__,
            train_rows=len(train_features),
            test_rows=len(test_features),
            feature_count=len(model.feature_columns),
            training_seconds=training_seconds,
            prediction_mean=round(float(np.mean(predictions)), 4),
            prediction_std=round(float(np.std(predictions)), 4),
            prediction_min=round(float(np.min(predictions)), 4),
            prediction_max=round(float(np.max(predictions)), 4),
            reload_verification_max_abs_diff=reload_verification_diff,
            artifact_path=str(artifact_path),
        )
        self.logger.info(
            "Trained %s in %.2f seconds; artifact saved to %s.",
            model.model_name,
            training_seconds,
            artifact_path.name,
        )
        return record, preview_frame

    def _verify_reloaded_model(
        self,
        model: BasePriceRegressor,
        artifact_path: Path,
        test_features: pd.DataFrame,
    ) -> float:
        """Confirm that a reloaded artifact reproduces the original predictions."""

        preview_rows = min(self.model_config.prediction_preview_rows, len(test_features))
        validation_features = test_features.head(preview_rows).copy()
        original_predictions = model.predict(validation_features)
        reloaded_model = model.__class__.load(artifact_path)
        restored_predictions = reloaded_model.predict(validation_features)
        difference = float(np.max(np.abs(original_predictions - restored_predictions))) if preview_rows else 0.0
        return round(difference, 10)

    def _build_prediction_preview(
        self,
        model_name: str,
        actual: pd.Series,
        predicted: np.ndarray,
    ) -> pd.DataFrame:
        """Create a compact table of first-K predictions for appendix usage."""

        preview_rows = min(self.model_config.prediction_preview_rows, len(predicted))
        preview = pd.DataFrame(
            {
                "model_name": model_name,
                "actual_price": actual.iloc[:preview_rows].to_numpy(dtype=float),
                "predicted_price": predicted[:preview_rows],
            }
        )
        preview["absolute_error"] = (preview["actual_price"] - preview["predicted_price"]).abs()
        preview["relative_error"] = np.where(
            preview["actual_price"] != 0,
            preview["absolute_error"] / preview["actual_price"],
            0.0,
        )
        return preview

    @staticmethod
    def _artifact_filename_for(model: BasePriceRegressor) -> str:
        sanitized = model.model_name.lower().replace(" ", "_")
        return f"{sanitized}_price_model.joblib"
