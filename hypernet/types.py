"""Shared interface. Images are RGB uint8 arrays, points are (x, y) pixels."""
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol
import numpy as np

@dataclass
class MatchResult:
    score: float
    evidence: dict[str, Any] = field(default_factory=dict)
    points_reference: list[list[float]] = field(default_factory=list)
    points_query: list[list[float]] = field(default_factory=list)
    inliers: list[bool] = field(default_factory=list)
    homography: list[list[float]] | None = None
    bbox_xyxy: list[float] | None = None

    def to_dict(self):
        return asdict(self)

class Matcher(Protocol):
    name: str
    def predict(self, reference: np.ndarray, query: np.ndarray) -> MatchResult: ...
    def provenance(self) -> dict[str, Any]: ...

def rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("Expected H x W x 3 RGB uint8 image")
    if min(array.shape[:2]) < 8:
        raise ValueError("Image must be at least 8 pixels on each axis")
    return np.ascontiguousarray(array)
