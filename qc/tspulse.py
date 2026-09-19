"""Phase 11: TSPulse research adapter.

TSPulse (`ibm-granite/granite-timeseries-tspulse-r1`) is an ultra-light
(1M parameter) model for embeddings, similarity search, classification and
anomaly detection. This adapter keeps it strictly research-only:

- embeddings need >= 512 points; weekly retail histories (~320 points) require
  an explicit linear resample and are never production-eligible;
- anomaly mode needs daily frequency and >= 2048 points and is deliberately not
  wired into weekly QC;
- the embedding revision is pinned and the output dimension is validated, so a
  silent model change fails loudly.

The gates mirror the adapter proven in the predecessor ``verity`` project.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

MODEL_ID = "ibm-granite/granite-timeseries-tspulse-r1"
EMBEDDING_REVISION = "tspulse-hybrid-dualhead-512-p8-r1"
ANOMALY_REVISION = "main"
MIN_EMBEDDING_LENGTH = 512
MIN_ANOMALY_LENGTH = 2048
EXPECTED_EMBEDDING_DIMENSION = 240
EMBEDDING_WEIGHT = 0.5


def _finite(values: Sequence[float], name: str = "series") -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains non-finite values")
    return result


@dataclass
class EmbeddingResult:
    status: str
    embedding: list[float] | None = None
    input_length: int = 0
    transformation: str | None = None
    production_eligible: bool = False
    model: str = MODEL_ID
    revision: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "embedding": self.embedding,
            "input_length": self.input_length,
            "transformation": self.transformation,
            "production_eligible": self.production_eligible,
            "model": self.model,
            "revision": self.revision,
            "detail": self.detail,
        }


@dataclass
class AnomalyResult:
    status: str
    scores: list[float] | None = None
    input_length: int = 0
    frequency: str = "D"
    production_eligible: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "scores": self.scores,
            "input_length": self.input_length,
            "frequency": self.frequency,
            "production_eligible": self.production_eligible,
            "detail": self.detail,
        }


class TSPulseResearch:
    def __init__(
        self,
        device: str = "cpu",
        expected_dimension: int | None = EXPECTED_EMBEDDING_DIMENSION,
    ) -> None:
        self.device = device
        self.expected_dimension = expected_dimension
        self._embedding_model: Any = None
        self._anomaly_pipeline: Any = None

    def embed(
        self, values: Sequence[float], allow_resample: bool = False
    ) -> EmbeddingResult:
        series = _finite(values)
        length = len(series)
        if length < MIN_EMBEDDING_LENGTH and not allow_resample:
            return EmbeddingResult(
                status="INSUFFICIENT_LENGTH",
                input_length=length,
                detail=(
                    f"TSPulse embeddings require at least {MIN_EMBEDDING_LENGTH} "
                    "points; pass allow_resample=True to use the research "
                    "linear-resample path."
                ),
            )

        import torch  # type: ignore
        from tsfm_public.models.tspulse import TSPulseForReconstruction  # type: ignore
        from tsfm_public.models.tspulse.utils.helpers import (  # type: ignore
            get_embeddings,
        )

        if self._embedding_model is None:
            self._embedding_model = (
                TSPulseForReconstruction.from_pretrained(
                    MODEL_ID,
                    revision=EMBEDDING_REVISION,
                    num_input_channels=1,
                    mask_type="user",
                )
                .to(self.device)
                .eval()
            )

        tensor = torch.tensor(series, dtype=torch.float32, device=self.device)
        resampled = length < MIN_EMBEDDING_LENGTH
        if resampled:
            tensor = torch.nn.functional.interpolate(
                tensor.reshape(1, 1, -1),
                size=MIN_EMBEDDING_LENGTH,
                mode="linear",
                align_corners=False,
            ).flatten()
        else:
            tensor = tensor[-MIN_EMBEDDING_LENGTH:]

        with torch.inference_mode():
            embedding = (
                get_embeddings(
                    self._embedding_model,
                    tensor.reshape(1, MIN_EMBEDDING_LENGTH, 1),
                    mode="register",
                )
                .flatten()
                .cpu()
                .tolist()
            )

        if not all(math.isfinite(value) for value in embedding):
            raise ValueError("non-finite TSPulse embedding")
        if self.expected_dimension is not None and len(embedding) != self.expected_dimension:
            raise ValueError(
                f"TSPulse embedding dimension {len(embedding)} does not match the "
                f"expected {self.expected_dimension}; model revision may have changed"
            )
        return EmbeddingResult(
            status="EXPERIMENTAL",
            embedding=embedding,
            input_length=length,
            transformation=(
                "linear_resample_to_512" if resampled else "last_512"
            ),
            revision=getattr(self._embedding_model.config, "_commit_hash", None),
            detail=(
                "Research embedding. A resampled input changes the temporal "
                "resolution and is not valid for production similarity."
                if resampled
                else "Research embedding of the latest 512 points."
            ),
        )

    def anomaly(self, values: Sequence[float], frequency: str = "D") -> AnomalyResult:
        series = _finite(values)
        if frequency != "D" or len(series) < MIN_ANOMALY_LENGTH:
            return AnomalyResult(
                status="RESEARCH_GATE",
                input_length=len(series),
                frequency=frequency,
                detail=(
                    f"TSPulse anomaly mode requires daily frequency and at least "
                    f"{MIN_ANOMALY_LENGTH} points; got {frequency!r} with "
                    f"{len(series)} points. Weekly QC must not use it."
                ),
            )

        import pandas as pd  # type: ignore
        from tsfm_public.models.tspulse import TSPulseForReconstruction  # type: ignore
        from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (  # type: ignore
            TimeSeriesAnomalyDetectionPipeline,
        )

        if self._anomaly_pipeline is None:
            model = (
                TSPulseForReconstruction.from_pretrained(
                    MODEL_ID,
                    revision=ANOMALY_REVISION,
                    num_input_channels=1,
                    mask_type="user",
                )
                .to(self.device)
                .eval()
            )
            self._anomaly_pipeline = TimeSeriesAnomalyDetectionPipeline(
                model,
                timestamp_column="timestamp",
                target_columns=["x"],
                prediction_mode=["time", "fft"],
                aggregation_length=64,
                aggr_function="max",
                smoothing_length=8,
                least_significant_scale=0.01,
                least_significant_score=0.1,
            )
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2000-01-01", periods=len(series), freq="D"),
                "x": series,
            }
        )
        result = self._anomaly_pipeline(
            frame, batch_size=16, predictive_score_smoothing=False
        )
        scores = [float(value) for value in result["anomaly_score"]]
        if not all(math.isfinite(value) for value in scores):
            raise ValueError("non-finite TSPulse anomaly score")
        return AnomalyResult(
            status="EXPERIMENTAL",
            scores=scores,
            input_length=len(series),
            detail=(
                "Offline reconstruction scores, not causal forecasts or "
                "calibrated probabilities."
            ),
        )


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding dimensions differ")
    a = _finite(left, "left")
    b = _finite(right, "right")
    denominator = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else 0.0
