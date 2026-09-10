"""Truth-blind chronological fit/predict assembly (no evaluation imports)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..contracts import FitContext, PredictContext
from ..folds import PITFold
from .adapters import (
    OBSERVED_TARGET,
    ProbabilisticAdapter,
    adapter_binding_sha256_by_id,
    create_adapter,
    verify_candidate_runtime_environment,
)
from .authorization import ExecutionAuthorization
from .contracts import (
    DENSITY_OUTPUT_COLUMNS,
    EVALUATION_ONLY_COLUMN,
    IDENTITY_COLUMNS,
    MODEL_OUTPUT_COLUMNS,
    QuantilePredictionBatch,
    ProbabilisticContractError,
    require_no_evaluation_truth,
)
from .spec import FEATURE_COLUMNS, feature_metadata
from .custody import (
    VerifiedPITFeatureArtifact,
    VerifiedTrainingLabelArtifact,
    materialize_authorized_fold,
)
from .resources import CandidateResourceGuard


@dataclass(frozen=True)
class FoldPredictionResult:
    output: pd.DataFrame
    raw_diagnostics: pd.DataFrame
    batch: QuantilePredictionBatch
    adapter: ProbabilisticAdapter

    def __post_init__(self) -> None:
        require_no_evaluation_truth(self.output.columns, context="fold prediction output")
        require_no_evaluation_truth(
            self.raw_diagnostics.columns, context="fold raw quantile diagnostics"
        )
        object.__setattr__(self, "output", self.output.copy(deep=True))
        object.__setattr__(self, "raw_diagnostics", self.raw_diagnostics.copy(deep=True))


def fit_predict_fold(
    *,
    model_id: str,
    features: VerifiedPITFeatureArtifact,
    labels: VerifiedTrainingLabelArtifact,
    authorization: ExecutionAuthorization,
    fold: PITFold,
    experiment_id: str,
    seed: int,
    entity_id: str,
    resource_guard: CandidateResourceGuard,
) -> FoldPredictionResult:
    """Fit only the authorized factory adapter on custody-owned chronological rows."""

    if not isinstance(authorization, ExecutionAuthorization):
        raise ProbabilisticContractError("runner requires factory execution authorization")
    if not isinstance(resource_guard, CandidateResourceGuard):
        raise ProbabilisticContractError("runner requires a live factory resource guard")
    authorization.verify_integrity()
    current_binding = adapter_binding_sha256_by_id().get(model_id)
    expected_binding = authorization.bindings["adapter_binding_sha256_by_id"].get(model_id)
    if current_binding != expected_binding:
        raise ProbabilisticContractError("adapter class/source/config/environment binding differs")
    current_environment = verify_candidate_runtime_environment(model_id)
    expected_environment = authorization.bindings["environment_manifest_sha256_by_id"].get(model_id)
    if current_environment != expected_environment:
        raise ProbabilisticContractError("candidate-specific runtime environment binding differs")
    resource_guard.before_fold(
        authorization=authorization,
        model_id=model_id,
        seed=seed,
        entity_id=entity_id,
        fold_id=fold.fold_id,
    )
    train_frame, test_frame = materialize_authorized_fold(
        features=features,
        labels=labels,
        authorization=authorization,
        fold=fold,
        seed=seed,
        entity_id=entity_id,
    )

    require_no_evaluation_truth(train_frame.columns, context="prediction-stage training frame")
    require_no_evaluation_truth(test_frame.columns, context="prediction-stage test frame")
    if OBSERVED_TARGET not in train_frame:
        raise ProbabilisticContractError("training frame is missing observed_pe")
    if OBSERVED_TARGET in test_frame:
        raise ProbabilisticContractError("test decision frame must not expose same-row observed_pe")
    for context_name, frame in (("training", train_frame), ("test", test_frame)):
        missing = [column for column in FEATURE_COLUMNS if column not in frame]
        if missing:
            raise ProbabilisticContractError(f"{context_name} frame missing features: {missing}")
    train_dates = pd.to_datetime(train_frame["date"], errors="coerce")
    test_dates = pd.to_datetime(test_frame["date"], errors="coerce")
    if train_dates.isna().any() or test_dates.isna().any():
        raise ProbabilisticContractError("chronological frames contain invalid dates")
    if (
        train_dates.max() > fold.train_end
        or test_dates.min() != fold.test_start
        or test_dates.max() != fold.test_end
        or train_dates.max() >= test_dates.min()
    ):
        raise ProbabilisticContractError("frame dates differ from the chronological fold")
    identity = _identity_output(test_frame, fold=fold, seed=seed)
    model = create_adapter(model_id)
    if model.definition.model_id != model_id:
        raise ProbabilisticContractError("adapter identity differs from requested candidate")
    features = feature_metadata()
    fit_context = FitContext(
        experiment_id=experiment_id,
        fold_id=fold.fold_id,
        seed=seed,
        train_end=pd.Timestamp(fold.train_end),
        target_name=OBSERVED_TARGET,
        feature_metadata=features,
    )
    predict_context = PredictContext(
        experiment_id=experiment_id,
        fold_id=fold.fold_id,
        seed=seed,
        prediction_start=pd.Timestamp(fold.test_start),
        prediction_end=pd.Timestamp(fold.test_end),
        feature_metadata=features,
    )
    model.fit(
        train_frame.loc[:, list(FEATURE_COLUMNS)],
        train_frame[OBSERVED_TARGET].rename(OBSERVED_TARGET),
        context=fit_context,
    )
    batch = model.predict(
        test_frame.loc[:, list(FEATURE_COLUMNS)],
        context=predict_context,
    )
    output = pd.concat(
        [
            identity,
            pd.Series(model_id, index=identity.index, name="model_id"),
            batch.output_frame(),
        ],
        axis=1,
    )
    allowed_schemas = {
        MODEL_OUTPUT_COLUMNS,
        (*MODEL_OUTPUT_COLUMNS, *DENSITY_OUTPUT_COLUMNS),
    }
    if tuple(output.columns) not in allowed_schemas:
        raise ProbabilisticContractError("assembled output schema differs from locked contract")
    resource_guard.after_fold(batch=batch, output=output, fold_id=fold.fold_id)
    adjacent = np.diff(batch.raw_log_quantiles, axis=1)
    raw_diagnostics = identity.copy()
    raw_diagnostics["model_id"] = model_id
    for index, label in enumerate(("p10", "p25", "p50", "p75", "p90")):
        raw_diagnostics[f"raw_predicted_log_pe_{label}"] = batch.raw_log_quantiles[:, index]
    raw_diagnostics["raw_crossing_pair_count"] = (adjacent < 0.0).sum(axis=1)
    raw_diagnostics["raw_total_negative_adjacent_gap"] = np.maximum(-adjacent, 0.0).sum(axis=1)
    raw_diagnostics["post_repair_crossing"] = (
        np.diff(batch.repaired_log_quantiles, axis=1) < 0.0
    ).any(axis=1)
    raw_diagnostics["interval_collapse"] = (
        batch.repaired_log_quantiles[:, 4] == batch.repaired_log_quantiles[:, 0]
    )
    return FoldPredictionResult(
        output=output,
        raw_diagnostics=raw_diagnostics,
        batch=batch,
        adapter=model,
    )


def _identity_output(test_frame: pd.DataFrame, *, fold: PITFold, seed: int) -> pd.DataFrame:
    missing = [
        column for column in ("entity_id", "date", "ordered_position") if column not in test_frame
    ]
    if missing:
        raise ProbabilisticContractError(f"test identity is incomplete: {missing}")
    if "seed" in test_frame and not (test_frame["seed"] == seed).all():
        raise ProbabilisticContractError("test seed differs from execution seed")
    if "fold_id" in test_frame and not (test_frame["fold_id"] == fold.fold_id).all():
        raise ProbabilisticContractError("test fold_id differs from scheduler fold")
    output = pd.DataFrame(
        {
            "seed": seed,
            "entity_id": test_frame["entity_id"].to_numpy(copy=True),
            "date": pd.to_datetime(test_frame["date"], errors="coerce").to_numpy(copy=True),
            "ordered_position": test_frame["ordered_position"].to_numpy(copy=True),
            "fold_id": fold.fold_id,
        }
    )
    if output.isna().any().any() or output.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ProbabilisticContractError("test identity is invalid or duplicated")
    return output.loc[:, list(IDENTITY_COLUMNS)]


def assert_prediction_module_truth_free() -> None:
    """Auditable invariant: this module never imports or names evaluator internals."""

    if EVALUATION_ONLY_COLUMN != "true_fair_pe":
        raise ProbabilisticContractError("evaluation-only truth sentinel changed")
