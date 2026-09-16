"""Pytorch implementation.

This module contains the pytorch implementation of the various soft distances.
"""

from cade.torch._soft_dtw_torch import (
    SoftDTWLoss,
    soft_dtw_alignment_matrix,
    soft_dtw_grad_x,
)
from cade.torch._cade_torch import (
    CADELoss,
    cade_alignment_matrix,
    cade_grad_x,
)

__all__ = [
    "SoftDTWLoss",
    "soft_dtw_alignment_matrix",
    "soft_dtw_grad_x",
    "CADELoss",
    "cade_grad_x",
    "cade_alignment_matrix",
]
