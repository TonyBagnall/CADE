"""JAX implementation of soft distances.

Provides pure functions for computing Soft-DTW and CADE distances,
alignment matrices, and gradients. Compatible with jax.grad and jax.jit.
"""

from cade.jax._soft_dtw_jax import (
    soft_dtw_alignment_matrix,
    soft_dtw_grad_x,
    soft_dtw_loss,
)
from cade.jax._cade_jax import (
    cade_alignment_matrix,
    cade_grad_x,
    cade_loss,
)

__all__ = [
    "soft_dtw_loss",
    "soft_dtw_alignment_matrix",
    "soft_dtw_grad_x",
    "cade_loss",
    "cade_alignment_matrix",
    "cade_grad_x",
]
