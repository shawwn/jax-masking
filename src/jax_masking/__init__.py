# read version from installed package
from importlib.metadata import version
__version__ = version(__name__)
del version

import inspect
from functools import partial
import functools
import numpy as np

from jax._src import api
from jax._src import core
from jax._src.api import Callable
from jax.tree_util import tree_flatten, tree_unflatten
from jax import linear_util as lu
from jax.interpreters import partial_eval as pe

from jax._src.api_util import flatten_fun_nokwargs
from jax._src.util import curry
from jax._src.util import safe_map, safe_zip, unzip2, prod, wrap_name

from . import masking

map = safe_map
zip = safe_zip


def _check_callable(fun):
  # In Python 3.10+, the only thing stopping us from supporting staticmethods
  # is that we can't take weak references to them, which the C++ JIT requires.
  if isinstance(fun, staticmethod):
    raise TypeError(f"staticmethod arguments are not supported, got {fun}")
  if not callable(fun):
    raise TypeError(f"Expected a callable value, got {fun}")
  if _isgeneratorfunction(fun):
    raise TypeError(f"Expected a function, got a generator function: {fun}")

def _isgeneratorfunction(fun):
  # TODO 3.9+: remove
  # re-implemented here because of https://bugs.python.org/issue33261
  while inspect.ismethod(fun):
    fun = fun.__func__
  while isinstance(fun, functools.partial):
    fun = fun.func
  return inspect.isfunction(fun) and bool(fun.__code__.co_flags & inspect.CO_GENERATOR)

@curry
def shapecheck(in_shapes, out_shape, fun: Callable):
  _check_callable(fun)
  in_shapes, in_tree = tree_flatten(in_shapes)
  in_shapes = map(masking.parse_spec, in_shapes)
  out_specs, out_spec_tree = tree_flatten(out_shape)
  out_specs = map(masking.parse_spec, out_specs)
  flat_fun, out_tree_thunk = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  avals = map(partial(core.ShapedArray, dtype=np.float32), in_shapes)
  out_shapes = [o.shape for o in pe.abstract_eval_fun(flat_fun.call_wrapped, *avals)]
  masking.check_shapes(map(tuple, out_specs), out_spec_tree,
                       map(tuple, out_shapes), out_tree_thunk())
  return fun

def mask(fun: Callable, in_shapes, out_shape=None) -> Callable:
  _check_callable(fun)
  unique_ids = masking.UniqueIds()

  in_specs, in_shapes_tree = tree_flatten(in_shapes)
  in_specs = map(masking.parse_spec, in_specs)
  in_specs = map(partial(masking.remap_ids, unique_ids), in_specs)

  if out_shape is not None:
    out_specs, out_spec_tree = tree_flatten(out_shape)
    out_specs = map(masking.parse_spec, out_specs)
    out_specs = map(partial(masking.remap_ids, unique_ids), out_specs)

  def wrapped_fun(args, logical_env):
    args_flat, in_tree = tree_flatten(args)
    if in_tree != in_shapes_tree:
      raise TypeError(f"Tree mismatch: Input {in_tree} and shape spec {in_shapes_tree}.")
    logical_env = {unique_ids[name] : val for name, val in logical_env.items()}
    in_shapes = map(masking.finalize_spec, in_specs, map(np.shape, args_flat))
    padded_env = masking.bind_shapes(in_shapes, [x.shape for x in args_flat])
    f = lu.wrap_init(fun)
    flat_fun, out_tree_thunk = flatten_fun_nokwargs(f, in_tree)
    outs, out_shapes = masking.mask_fun(
      flat_fun, logical_env, padded_env, args_flat, in_shapes)
    out_tree = out_tree_thunk()

    if out_shape is None:
      def logical_shape(poly_shape, padded_val):
        shape = masking.eval_poly_shape(poly_shape, logical_env)
        return api.ShapeDtypeStruct(shape, core.get_aval(padded_val).dtype)
      out_logicals = map(logical_shape, out_shapes, outs)
      return tree_unflatten(out_tree, outs), tree_unflatten(out_tree, out_logicals)
    else:
      masking.check_shapes(out_specs, out_spec_tree, list(out_shapes), out_tree)
      def padded_spec(shape_spec):
        return tuple(dim if dim is masking._monomorphic_dim else
                     masking.eval_poly(dim, padded_env) for dim in shape_spec)
      masking.check_shapes(map(padded_spec, out_specs), out_spec_tree,
                           map(np.shape, outs), out_tree, "Padded output")
      return tree_unflatten(out_tree, outs)
  return wrapped_fun

