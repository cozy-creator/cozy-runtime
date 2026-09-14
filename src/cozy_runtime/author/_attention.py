"""Read-only facts for a model's optional attention selection policy."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AttentionContext:
    """One attention site's eligible choices, supplied before model warm-up.

    Eligibility covers installed dependencies, hardware, site dtype/head size and
    parallelism support. It is not a model-quality approval. Returning an explicit
    backend still requires Runtime's kernel validation; unavailable choices refuse.
    Weight storage precision is deliberately separate from attention activation dtype.
    """

    component: str
    module_path: str
    device_kind: str
    device_name: str
    sm: int
    activation_dtype: str
    head_dim: int
    degree: int
    eligible: tuple[str, ...]
    recommended: str
    recommendation_reason: str
