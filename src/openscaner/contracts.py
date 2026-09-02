from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


Status = Literal["ok", "not_detected", "unavailable", "error"]


@dataclass(frozen=True)
class CandidateResult:
    name: str
    status: Status
    corners: np.ndarray | None
    confidence: float
    backend: str | None
    elapsed_ms: float
    peak_rss_mb: float
    error: str | None = None
    diagnostics: dict[str, object] | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.corners is not None:
            result["corners"] = np.asarray(self.corners, dtype=float).tolist()
        return result
