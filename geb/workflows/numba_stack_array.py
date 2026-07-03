"""Stack-allocated array utilities for Numba JIT-compiled code.

Stack allocation avoids heap overhead for small, short-lived arrays inside
hot loops. Arrays allocated here must not be returned from the enclosing
function since their memory is reclaimed when the stack frame exits.
"""

from typing import Any, Literal

import numpy as np
from numba import carray, farray, njit, typeof, types
from numba.core import cgutils, errors, typing as numba_typing
from numba.extending import intrinsic


@intrinsic
def _stack_empty_alloc(
    typingctx: numba_typing.Context, shape: int | tuple[int, ...], dtype: np.dtype
) -> tuple[types.CPointer, Any]:
    """Intrinsic that emits a stack allocation of the given total element count.

    The shape must be composed of integer literals so the total size is known
    at compile time — dynamic shapes cannot be stack-allocated.

    Args:
        typingctx: Numba typing context (automatically passed by numba).
        shape: An integer literal or a tuple of integer literals giving the
            array dimensions.
        dtype: A Numba dtype instance (e.g. ``numba.float64``).

    Returns:
        A raw pointer to the stack-allocated memory region.

    Raises:
        errors.TypingError: If shape contains non-literal dimensions.
    """
    size = 1
    if isinstance(shape, types.scalars.IntegerLiteral):
        size = shape.literal_value
        sig = types.CPointer(dtype.dtype)(types.int64, dtype)  # ty:ignore[unresolved-attribute]
    elif isinstance(shape, (types.containers.Tuple, types.containers.UniTuple)):
        for i in range(len(shape)):
            size *= shape[i].literal_value
        sig = types.CPointer(dtype.dtype)(typeof(shape).instance_type, dtype)  # ty:ignore[unresolved-attribute]
    else:
        raise errors.TypingError(
            "shape must be an IntegerLiteral or a tuple of IntegerLiterals"
        )

    def impl(
        context: numba_typing.Context, builder: Any, signature: Any, args: Any
    ) -> Any:
        ty = context.get_value_type(dtype.dtype)  # ty:ignore[unresolved-attribute]
        ptr = cgutils.alloca_once(builder, ty, size=size)
        return ptr

    return sig, impl


@njit(inline="always")
def stack_empty(
    shape: int | tuple[int, ...], dtype: np.dtype, order: Literal["C", "F"] = "C"
) -> np.ndarray:
    """Allocate an uninitialised array on the call stack.

    Avoids heap allocation overhead for small, temporary arrays inside tight
    loops. The returned array must not be returned from the enclosing function
    and the shape must be composed of compile-time integer constants.

    Must be inlined: stack memory is freed when the stack frame exits, so the
    array view must not outlive the caller's frame.

    Notes:
        Stack size is limited; do not use for large arrays.

    Args:
        shape: An integer literal or a tuple of integer literals.
        dtype: NumPy dtype (e.g. ``np.float64``).
        order: Memory layout, ``'C'`` (row-major) or ``'F'`` (column-major).

    Returns:
        An uninitialised NumPy array backed by stack memory.

    Raises:
        ValueError: If order is not ``'C'`` or ``'F'``.
    """
    arr_ptr = _stack_empty_alloc(shape, dtype)  # ty:ignore[invalid-argument-type, too-many-positional-arguments]
    if order == "C":
        return carray(arr_ptr, shape)
    elif order == "F":
        return farray(arr_ptr, shape)
    else:
        raise ValueError("order must be one of 'C', 'F'")
