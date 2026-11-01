"""
Conversion pipeline: Cholesky-whitened SVD decomposition and gauge calibration
for converting LiquidAI/LFM3-1.7B into parameter-shared recurrent-depth LFM3-1.7B-RD.
"""

from .calibration import build_calibration_domains, compute_cholesky_factors
from .decomposition import align_conv_bx_gauge, balance_ba, decompose_5_motifs_rd

__all__ = [
    "build_calibration_domains",
    "compute_cholesky_factors",
    "align_conv_bx_gauge",
    "balance_ba",
    "decompose_5_motifs_rd",
]
