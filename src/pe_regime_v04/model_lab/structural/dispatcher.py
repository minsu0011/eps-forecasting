"""Single authorization boundary for Structural Wave candidate operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .authorization import EXPECTED_ENABLED, StructuralExecutionAuthorization
from .contracts import KernelBinding, StructuralContractError
from .decomposition import (
    DecompositionFit,
    StructuralTrack,
    fit_decomposition_kernel,
    predict_decomposition_kernel,
)
from .ensemble import (
    SimplexLogWeightFit,
    apply_two_base_simplex_log_weights,
    equal_geometric_blend,
    fit_two_base_simplex_log_weights,
)
from .meta import GeneratedMetaFeatureArtifact
from .residuals import (
    ResidualAR1Fit,
    fit_residual_ar1_kernel,
    fit_residual_huber_kernel,
    predict_residual_ar1_kernel,
    predict_residual_huber_kernel,
)


_DISPATCHER_TOKEN = object()


@dataclass(frozen=True, init=False)
class StructuralDispatcher:
    """Factory-only dispatcher carrying one still-live sealed authorization."""

    authorization: StructuralExecutionAuthorization

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise StructuralContractError("StructuralDispatcher is factory-only")

    @classmethod
    def from_authorization(
        cls, authorization: StructuralExecutionAuthorization
    ) -> "StructuralDispatcher":
        if not isinstance(authorization, StructuralExecutionAuthorization):
            raise StructuralContractError("dispatcher requires sealed authorization")
        authorization.verify()
        output = object.__new__(cls)
        object.__setattr__(output, "authorization", authorization)
        return output

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """The dispatcher surface is exactly the five sealed enabled candidates."""

        self.authorization.verify()
        if self.authorization.enabled_candidates != EXPECTED_ENABLED:
            raise StructuralContractError("dispatcher candidate universe changed")
        return EXPECTED_ENABLED

    def decomposition_binding(
        self,
        *,
        candidate_id: str,
        fold_sha256: str,
    ) -> KernelBinding:
        self.authorization.require_candidate(candidate_id)
        return KernelBinding.from_authorization(
            self.authorization,
            candidate_id=candidate_id,
            fold_sha256=fold_sha256,
        )

    def fit_decomposition(
        self,
        candidate_id: str,
        train_features: pd.DataFrame,
        observed_pe: pd.Series,
        train_dates: pd.Series,
        *,
        binding: KernelBinding,
    ) -> DecompositionFit:
        expected_track: StructuralTrack
        if candidate_id == "decomp_block_ridge_ar1_lag1":
            expected_track = "A"
        elif candidate_id == "decomp_block_ridge_ar1_current":
            expected_track = "C"
        else:
            raise StructuralContractError("candidate is not an authorized decomposition")
        return fit_decomposition_kernel(
            train_features,
            observed_pe,
            train_dates,
            candidate_id=candidate_id,
            track=expected_track,
            binding=binding,
            authorization=self.authorization,
        )

    def predict_decomposition(
        self,
        candidate_id: str,
        fit: DecompositionFit,
        test_features: pd.DataFrame,
        test_dates: pd.Series,
    ) -> pd.DataFrame:
        self.authorization.require_candidate(candidate_id)
        if fit.candidate_id != candidate_id:
            raise StructuralContractError("decomposition fit candidate substitution")
        return predict_decomposition_kernel(
            fit,
            test_features,
            test_dates,
            authorization=self.authorization,
        )

    def fit_residual_ar1(
        self, inner: GeneratedMetaFeatureArtifact, observed: pd.DataFrame
    ) -> ResidualAR1Fit:
        self.authorization.require_candidate("residual_ar1_nested_oof")
        return fit_residual_ar1_kernel(inner, observed, authorization=self.authorization)

    def predict_residual_ar1(
        self, fit: ResidualAR1Fit, outer: GeneratedMetaFeatureArtifact
    ) -> pd.DataFrame:
        self.authorization.require_candidate("residual_ar1_nested_oof")
        return predict_residual_ar1_kernel(fit, outer, authorization=self.authorization)

    def fit_huber(self, *args: Any, **kwargs: Any) -> None:
        return fit_residual_huber_kernel(*args, authorization=self.authorization, **kwargs)

    def predict_huber(self, *args: Any, **kwargs: Any) -> None:
        return predict_residual_huber_kernel(*args, authorization=self.authorization, **kwargs)

    def equal_blend(
        self,
        base0: GeneratedMetaFeatureArtifact,
        base1: GeneratedMetaFeatureArtifact,
    ) -> pd.DataFrame:
        self.authorization.require_candidate("stack_geometric_equal_pair")
        return equal_geometric_blend(base0, base1, authorization=self.authorization)

    def fit_simplex(
        self,
        base0: GeneratedMetaFeatureArtifact,
        base1: GeneratedMetaFeatureArtifact,
        observed: pd.DataFrame,
    ) -> SimplexLogWeightFit:
        self.authorization.require_candidate("stack_simplex_pair_frozen")
        return fit_two_base_simplex_log_weights(
            base0, base1, observed, authorization=self.authorization
        )

    def apply_simplex(
        self,
        fit: SimplexLogWeightFit,
        base0: GeneratedMetaFeatureArtifact,
        base1: GeneratedMetaFeatureArtifact,
    ) -> pd.DataFrame:
        self.authorization.require_candidate("stack_simplex_pair_frozen")
        return apply_two_base_simplex_log_weights(
            fit, base0, base1, authorization=self.authorization
        )
