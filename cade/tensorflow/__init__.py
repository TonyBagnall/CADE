"""TensorFlow implementation of soft distances.

Provides classes and functions for computing Soft-DTW and CADE distances,
alignment matrices, and gradients. Compatible with tf.GradientTape.
"""

from cade.tensorflow._soft_dtw_tf import (
    SoftDTWLoss,
    soft_dtw_alignment_matrix,
    soft_dtw_grad_x,
)
from cade.tensorflow._cade_tf import (
    CADELoss,
    cade_alignment_matrix,
    cade_grad_x,
)

__all__ = [
    "SoftDTWLoss",
    "soft_dtw_alignment_matrix",
    "soft_dtw_grad_x",
    "CADELoss",
    "cade_alignment_matrix",
    "cade_grad_x",
]
