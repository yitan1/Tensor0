"""Update storage and operand validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._ffi._calls import execute_update
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._jax import update_p
from tensor0._stride._layout import AffineRecord


@pytest.mark.parametrize("operand", ["source", "base"])
@pytest.mark.parametrize("shape,error", [((), "at least one dimension"), ((2, 4), "batch shapes")])
def test_update_storage_shape_validation(operand, shape, error):
    arguments = [jnp.ones(4), jnp.ones(4), jnp.float32(1), jnp.float32(0)]
    arguments[0 if operand == "source" else 1] = jnp.ones(shape)
    layout = encode_layout((AffineRecord((4,), (1,), 0, (1,), 0),), source_size=4, output_size=4)
    for operation in (execute_update, jax.jit(lambda *values: execute_update(*values, layout=layout))):
        with pytest.raises(ValueError, match=error):
            if operation is execute_update:
                operation(*arguments, layout=layout)
            else:
                operation(*arguments)


@pytest.mark.parametrize("operand", range(4))
def test_update_rejects_non_array_operands(operand):
    arguments = [jnp.ones(4), jnp.ones(4), jnp.float32(1), jnp.float32(0)]
    arguments[operand] = np.asarray(arguments[operand])
    layout = encode_layout((), source_size=4, output_size=4)
    with pytest.raises(TypeError, match="JAX Array or Tracer"):
        execute_update(*arguments, layout=layout)


@pytest.mark.parametrize("operand", ["source", "base"])
def test_update_rejects_out_of_bounds_layout(operand):
    source = jnp.ones(3 if operand == "source" else 4)
    base = jnp.ones(3 if operand == "base" else 4)
    records = (AffineRecord((4,), (1,), 0, (1,), 0),)
    with pytest.raises(Exception, match="address exceeds storage"):
        update_p.bind(source, base, jnp.float32(1), jnp.float32(0), records=records).block_until_ready()
    np.testing.assert_array_equal(base, np.ones(base.shape))


def test_update_rejects_unsupported_storage_pair():
    with pytest.raises(Exception, match="unsupported update source/storage dtype pair"):
        update_p.bind(jnp.ones(4, jnp.complex64), jnp.ones(4, jnp.int32),
                      jnp.int32(1), jnp.int32(0), records=()).block_until_ready()
