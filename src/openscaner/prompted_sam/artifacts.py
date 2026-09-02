"""Frozen artifact identities for DocAligner-prompted MobileSAM."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from openscaner.fusion.artifacts import (
    ModelArtifactIdentity,
    PolicyArtifactIdentity,
)


PROMPTED_SAM_DEPENDENCY_ARTIFACTS: Mapping[str, ModelArtifactIdentity] = (
    MappingProxyType(
        {
            "docaligner": ModelArtifactIdentity(
                adapter="docaligner",
                filename="lcnet100_h_e_bifpn_256_fp32.onnx",
                sha256="f4117b786e3a18470f3865c93f3c2bd69d9b998edd60f385574a5c665e79594e",
                size_bytes=4_767_987,
            ),
            "mobile_sam": ModelArtifactIdentity(
                adapter="mobile_sam",
                filename="mobile_sam.pt",
                sha256="6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
                size_bytes=40_728_226,
            ),
        }
    )
)

PROMPTED_SAM_POLICY_ARTIFACT = PolicyArtifactIdentity(
    adapter="docaligner_prompted_mobile_sam",
    filename="docaligner_prompted_mobile_sam_policy.json",
    schema_version=1,
    sha256="c3ce7727def194b1f06bf7cc8df057a03fd148a88aa4b4a84b82602e275858f2",
    size_bytes=332,
)

PROMPTED_SAM_CALIBRATION_REPORT_ARTIFACT = ModelArtifactIdentity(
    adapter="docaligner_prompted_mobile_sam",
    filename="docaligner_prompted_mobile_sam.json",
    sha256="6a20f096a1ef5302e1bb3a74aeefce92512e1029822361d7e6f76669e13081aa",
    size_bytes=3_206,
)


__all__ = [
    "PROMPTED_SAM_CALIBRATION_REPORT_ARTIFACT",
    "PROMPTED_SAM_DEPENDENCY_ARTIFACTS",
    "PROMPTED_SAM_POLICY_ARTIFACT",
]
