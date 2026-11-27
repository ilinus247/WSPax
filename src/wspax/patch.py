"""
Patches applied to PyTrees.

Patches are created by mapping functions over PyTrees just like jax.tree.map. wspax.patch.create_patch maps a
function over a tree and outputs both the modified PyTree and a patch representing the changes applied.
"""
from typing import Callable, Any

import jax
import jax.numpy as jnp
import equinox as eqx
from jax import Array


class PatchLeaf(eqx.Module):
    """
    Represents a modification to a PyTree leaf.
    The intended use for PatchLeaf is to track a node being changed in mapping.
    The PatchLeaf stores the value it had before and after.
    PatchLeaf also stores a helper flag to mark if the value was actually changed.
    """
    before: Any
    after: Any
    changed: jax.Array


def create_patch(fn: Callable[[Any], Any],
                 tree: Any,
                 *,
                 equality_fn: Callable[[Any, Any], Any] = None) -> tuple[Any, Any]:
    """
    Map fn over a tree and create a patch representing the changes applied.
    equality_fn is used to judge if the mapping provided resulted in a change of this leaf.
    A default equality_fn is used if not provided.
    The default function checks equality, array equality, dtypes when possible, and raw types.

    Returns the mapped tree and the patch tracking the changes made
    """
    values, structure = jax.tree.flatten(tree)
    f_values = [fn(v) for v in values]
    patch = []
    if equality_fn is None:
        equality_fn = _equality
    for v, f_v in zip(values, f_values):
        equality = equality_fn(f_v, v)
        patch.append(PatchLeaf(v, f_v, jnp.logical_not(equality)))
    return jax.tree.unflatten(structure, f_values), jax.tree.unflatten(structure, patch)


def apply_patch(patch: Any, tree: Any, *, fn: Callable[[PatchLeaf, Any], Any] | None = None) -> Any:
    """
    Apply a patch over a tree and return the modified PyTree.

    fn is an optional function that takes a PatchLeaf and its
    corresponding tree leaf and returns the updated leaf.
    It is called regardless of the PatchLeaf's changed status.

    If fn is None, the patch is applied as a replacement patch,
    and changed leaves are replaced with their recorded after values
    and preserving unchanged leaves.

    Returns the tree created by applying the patch
    """
    if fn is None:
        fn = lambda p, t: jnp.where(p.changed, p.after, t)
    return jax.tree.map(fn, patch, tree, is_leaf=_is_patch_leaf)


def compose_patches(patch1: Any, patch2: Any) -> Any:
    """
    Compose two patches. Intermediate values are ignored completely.
    Creates a patch that represents the effects of applying patch1 and then patch2 sequentially.
    Verifying the patches are compatible is left to the user, with validate_composition provided as a helper.
    i.e., for compatible patches applied to the original tree using default equality and replacement semantics,
    apply_patch(patch2, apply_patch(patch1, tree)) == apply_patch(compose_patches(patch1, patch2), tree)
    """

    def _combine_patch_leaves(p1, p2):
        return PatchLeaf(p1.before, p2.after, jnp.logical_not(_equality(p1.before, p2.after)))

    return jax.tree.map(_combine_patch_leaves,
                        patch1,
                        patch2,
                        is_leaf=_is_patch_leaf)


def invert_patch(patch: Any) -> Any:
    """
    Invert a patch so applying it over a tree reverses its original effect. The inverted patch has the property that
    applying it after applying the original patch once results in an unchanged tree.
    i.e. tree == apply_patch(invert_patch(patch), apply_patch(patch, tree))

    Returns the inverted patch
    """
    return jax.tree.map(lambda p: PatchLeaf(p.after, p.before, p.changed),
                        patch,
                        is_leaf=_is_patch_leaf)


def validate_application(patch: Any, tree: Any, *, check_values: bool = True) -> jax.Array:
    """
    Check structural and original-value compatibility between a patch and a tree.
    If check_values is False, only their structures are checked to ensure they map correctly.

    Returns a scalar boolean array representing the validity of the application.
    """
    structure_integrity = (jax.tree.structure(patch, is_leaf=_is_patch_leaf) ==
                           jax.tree.structure(tree))
    if not (structure_integrity and check_values):
        return jnp.asarray(structure_integrity)

    compatible = jnp.all(
        jnp.asarray(
            jax.tree.leaves(
                jax.tree.map(lambda p, t: _equality(p.before, t),
                             patch,
                             tree,
                             is_leaf=_is_patch_leaf)
            )
        )
    )
    return compatible


def validate_composition(patch1: Any, patch2: Any, *, check_values: bool = True) -> jax.Array:
    """
    Check structural and intermediate-value compatibility between two patches.
    If check_values is False, only the structure of both patches are checked to ensure they map correctly.

    Returns a boolean representing the validity of a composition between the two patches.
    """
    structure_integrity = (jax.tree.structure(patch1, is_leaf=_is_patch_leaf) ==
                           jax.tree.structure(patch2, is_leaf=_is_patch_leaf))
    if not (structure_integrity and check_values):
        return jnp.asarray(structure_integrity)
    compatible = jnp.all(
        jnp.asarray(
            jax.tree.leaves(
                jax.tree.map(lambda p1, p2: _equality(p1.after, p2.before),
                             patch1,
                             patch2,
                             is_leaf=_is_patch_leaf)
            )
        )
    )
    return compatible


def _is_patch_leaf(x) -> bool:
    return isinstance(x, PatchLeaf)


def _equality(f_v, v) -> Any | Array:
    if eqx.is_array(v) and eqx.is_array(f_v):
        equality = (v.dtype == f_v.dtype) and jnp.array_equal(v, f_v, equal_nan=True)
    elif eqx.is_array(v) or eqx.is_array(f_v):
        equality = False
    else:
        equality = type(f_v) is type(v) and f_v == v
    return equality
