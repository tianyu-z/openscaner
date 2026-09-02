"""Frozen model artifact identities required by production fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ModelArtifactIdentity:
    """Identity of exact immutable model bytes used to build a runtime."""

    adapter: str
    filename: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, str) or not self.adapter:
            raise ValueError("adapter must be a non-empty string")
        if (
            not isinstance(self.filename, str)
            or not self.filename
            or Path(self.filename).name != self.filename
        ):
            raise ValueError("filename must be a safe basename")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("sha256 must be lowercase hexadecimal")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 1
        ):
            raise ValueError("size_bytes must be a positive integer")

    def as_document(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class PolicyArtifactIdentity:
    """Identity and schema of the immutable production fusion policy."""

    adapter: str
    filename: str
    schema_version: int
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        ModelArtifactIdentity(
            adapter=self.adapter,
            filename=self.filename,
            sha256=self.sha256,
            size_bytes=self.size_bytes,
        )
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")

    def as_document(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


FUSION_DEPENDENCY_ARTIFACTS: Mapping[str, ModelArtifactIdentity] = MappingProxyType(
    {
        "docaligner": ModelArtifactIdentity(
            adapter="docaligner",
            filename="lcnet100_h_e_bifpn_256_fp32.onnx",
            sha256="f4117b786e3a18470f3865c93f3c2bd69d9b998edd60f385574a5c665e79594e",
            size_bytes=4_767_987,
        ),
        "pp_lcnet_050_corner": ModelArtifactIdentity(
            adapter="pp_lcnet_050_corner",
            filename="pp_lcnet_050_corner.pth",
            sha256="e14d68142d15959664287a98eea0d1e5c5470398e6d8fac0e9a32736d6518718",
            size_bytes=1_300_171,
        ),
    }
)

FUSION_POLICY_ARTIFACT = PolicyArtifactIdentity(
    adapter="docaligner_pp_lcnet_fusion",
    filename="docaligner_pp_lcnet_fusion_policy.json",
    schema_version=1,
    sha256="186e6bc56211228eaaf128794eab5f758f678e2fbfee90cc28ae5e06c01df82b",
    size_bytes=279,
)


__all__ = [
    "FUSION_DEPENDENCY_ARTIFACTS",
    "FUSION_POLICY_ARTIFACT",
    "ModelArtifactIdentity",
    "PolicyArtifactIdentity",
]
