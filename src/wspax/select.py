# Contains convenience definitions for selectors.
# Primarily intended to increase readability in client code.

from collections.abc import Callable
from typing import Any
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
import numpy as np

Selector = Callable[[Any], bool]


def all_of(*selectors: Selector) -> Selector:
    """combines all selectors using logical and"""

    def _internal(leaf):
        return all(selector(leaf) for selector in selectors)

    return _internal


def any_of(*selectors: Selector) -> Selector:
    """combines all selectors using logical or"""

    def _internal(leaf):
        return any(selector(leaf) for selector in selectors)

    return _internal


def negate(selector: Selector) -> Selector:
    """negates a selector"""
    return lambda leaf: not selector(leaf)


def of_dtype(dtype: DTypeLike) -> Selector:
    """
    Creates a selector for leaves that are arrays with specified dtype.
    Supports abstract dtype categories.
    """
    return lambda leaf: ((isinstance(leaf, np.ndarray) or isinstance(leaf, jax.Array))
                         and jnp.issubdtype(leaf.dtype, dtype))


def of_shape(shape: tuple[int, ...]) -> Selector:
    """Creates a selector for leaves that are arrays with specified shape."""
    return lambda leaf: ((isinstance(leaf, np.ndarray) or isinstance(leaf, jax.Array))
                         and leaf.shape == shape)


def of_ndim(ndim: int) -> Selector:
    """Creates a selector for leaves that are arrays with specified shape."""
    return lambda leaf: ((isinstance(leaf, np.ndarray) or isinstance(leaf, jax.Array))
                         and leaf.ndim == ndim)


# Select every leaf
everything: Selector = lambda leaf: True
# Select no leaf
nothing: Selector = lambda leaf: False

# These aliases exist so users don't have to use equinox directly
# Believe it or not, this is what real software engineering looks like.
is_array = eqx.is_array
is_array_like = eqx.is_array_like
is_inexact_array = eqx.is_inexact_array
is_inexact_array_like = eqx.is_inexact_array_like
