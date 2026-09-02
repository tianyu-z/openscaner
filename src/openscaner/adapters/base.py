from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CPU_BACKENDS = frozenset({"CPUExecutionProvider", "torch:cpu", "opencv:cpu"})


class AdapterContractError(ValueError):
    """Raised when a candidate does not expose the common adapter contract."""


class AdapterUnavailable(RuntimeError):
    """Raised when a candidate cannot run with its installed assets or dependencies."""


@dataclass(frozen=True)
class AdapterOutput:
    """A candidate's document quadrilateral, or no detection when corners is None."""

    corners: np.ndarray | None
    confidence: float
    backend: str
    diagnostics: dict[str, object] | None = None

    def __post_init__(self) -> None:
        confidence = float(self.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in the range [0, 1]")
        object.__setattr__(self, "confidence", confidence)

        if self.backend not in CPU_BACKENDS:
            raise AdapterContractError(
                "adapter must declare a supported CPU backend: "
                + ", ".join(sorted(CPU_BACKENDS))
            )

        if self.diagnostics is not None:
            if not isinstance(self.diagnostics, dict):
                raise TypeError("diagnostics must be a JSON-safe object or None")
            try:
                diagnostics = json.loads(json.dumps(self.diagnostics, allow_nan=False))
            except (TypeError, ValueError) as error:
                raise ValueError("diagnostics must be a JSON-safe object") from error
            object.__setattr__(self, "diagnostics", diagnostics)

        if self.corners is None:
            return
        corners = np.array(self.corners, dtype=np.float32, copy=True)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError("corners must contain exactly four finite 2D points")
        corners.setflags(write=False)
        object.__setattr__(self, "corners", corners)


AdapterEntrypoint = Callable[[np.ndarray, Path, int], AdapterOutput]
_ENTRYPOINT_PARAMETERS = ("image", "model_dir", "cpu_threads")


def validate_entrypoint(entrypoint: Callable[..., object]) -> AdapterEntrypoint:
    """Validate and return a candidate's exact three-argument entrypoint."""
    if not callable(entrypoint):
        raise AdapterContractError("adapter entrypoint must be callable")

    parameters = tuple(inspect.signature(entrypoint).parameters.values())
    names = tuple(parameter.name for parameter in parameters)
    positional_kinds = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    if names != _ENTRYPOINT_PARAMETERS or any(
        parameter.kind not in positional_kinds for parameter in parameters
    ):
        raise AdapterContractError(
            "adapter entrypoint must accept only image, model_dir, cpu_threads"
        )
    return entrypoint  # type: ignore[return-value]


def load_adapter(module_name: str) -> AdapterEntrypoint:
    """Import one adapter module and validate its ``run`` function."""
    module = importlib.import_module(module_name)
    try:
        entrypoint = module.run
    except AttributeError as exc:
        raise AdapterContractError(f"{module_name} does not define run") from exc
    return validate_entrypoint(entrypoint)


def discover_adapters(package_name: str = "openscaner.adapters") -> dict[str, str]:
    """Discover adapter module names without importing candidate modules."""
    package = importlib.import_module(package_name)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        raise ValueError(f"{package_name} is not a package")

    discovered: dict[str, str] = {}
    for module in pkgutil.iter_modules(package_paths):
        if module.ispkg or module.name == "base" or module.name.startswith("_"):
            continue
        discovered[module.name] = f"{package_name}.{module.name}"
    return dict(sorted(discovered.items()))
