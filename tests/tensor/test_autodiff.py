import jax
import jax.numpy as jnp

import tensor0
from tensor0 import SU2Irrep, TensorMap, U1Irrep, hom, space
from tests.cases import assert_allclose


def _su2_tensor(dtype=jnp.float32):
    factor = space(SU2Irrep, {0: 1, 1: 1})
    target = hom((factor,), (factor,))
    first = jnp.asarray([[2.0]], dtype=dtype)
    second = jnp.asarray([[3.0]], dtype=dtype)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        first = first + jnp.asarray([[1.0j]], dtype=dtype)
        second = second + jnp.asarray([[2.0j]], dtype=dtype)
    return tensor0.from_blocks(target, {0: first, 1: second})


def _squared_norm(value):
    return jnp.real(tensor0.inner(value, value))


def test_grad_returns_su2_riesz_gradient_for_real_tensormap():
    tensor = _su2_tensor()

    coordinate = jax.grad(_squared_norm)(tensor)
    gradient = tensor0.grad(_squared_norm)(tensor)

    for coupled, block in tensor.blocks():
        weight = tensor.space.sector_spec.quantum_dim(coupled)
        assert_allclose(coordinate.block(coupled), 2 * weight * block)
        assert_allclose(gradient.block(coupled), 2 * block)


def test_grad_conjugates_complex_jax_cotangent_before_inverse_metric():
    tensor = _su2_tensor(jnp.complex64)

    coordinate = jax.grad(_squared_norm)(tensor)
    gradient = tensor0.grad(_squared_norm)(tensor)

    for coupled, block in tensor.blocks():
        weight = tensor.space.sector_spec.quantum_dim(coupled)
        assert_allclose(coordinate.block(coupled), 2 * weight * jnp.conj(block))
        assert_allclose(gradient.block(coupled), 2 * block)


def test_grad_is_identity_metric_for_abelian_tensormap():
    factor = space(U1Irrep, {0: 1, 1: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(
        target,
        jnp.asarray([1.0 + 2.0j, 3.0 + 4.0j], dtype=jnp.complex64),
    )

    gradient = tensor0.grad(_squared_norm)(tensor)

    assert_allclose(gradient.storage.data, 2 * tensor.storage.data)


def test_value_and_grad_supports_jit_and_auxiliary_output():
    tensor = _su2_tensor()

    def loss(value):
        result = _squared_norm(value)
        return result, value.storage.data.shape[0]

    compiled = jax.jit(tensor0.value_and_grad(loss, has_aux=True))
    (value, aux), gradient = compiled(tensor)

    assert_allclose(value, _squared_norm(tensor))
    assert aux == tensor.storage.data.shape[0]
    assert_allclose(gradient.storage.data, 2 * tensor.storage.data)


def test_grad_converts_nested_tensormaps_but_preserves_array_gradients():
    tensor = _su2_tensor()
    array = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    parameters = {"tensor": tensor, "array": array}

    def loss(values):
        return _squared_norm(values["tensor"]) + jnp.sum(values["array"] ** 2)

    gradient = tensor0.grad(loss)(parameters)

    assert_allclose(gradient["tensor"].storage.data, 2 * tensor.storage.data)
    assert_allclose(gradient["array"], 2 * array)


def test_grad_supports_multiple_tensor_and_array_argnums():
    tensor = _su2_tensor()
    array = jnp.asarray([2.0, 3.0], dtype=jnp.float32)

    def loss(tensor_value, array_value):
        return _squared_norm(tensor_value) + jnp.sum(array_value**2)

    tensor_gradient, array_gradient = tensor0.grad(
        loss,
        argnums=(0, 1),
    )(tensor, array)

    assert_allclose(tensor_gradient.storage.data, 2 * tensor.storage.data)
    assert_allclose(array_gradient, 2 * array)


def test_array_parameter_remains_a_coordinate_gradient_when_tensormap_is_internal():
    tensor = _su2_tensor()

    def loss(data):
        return _squared_norm(TensorMap(tensor.space, data))

    expected = jax.grad(loss)(tensor.storage.data)
    actual = tensor0.grad(loss)(tensor.storage.data)

    assert_allclose(actual, expected)
