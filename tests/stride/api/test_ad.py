"""Public view-algebra differentiation and batching contracts."""

from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

from tensor0._stride import StridedView, add, dotc, dotu, materialize, reduce_sum, scale
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._jax import reduction_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.data import INTEGER_PAIRS
from tests.stride.support.oracles.coefficients import DISCRETE, check
from tests.stride.support.oracles.materialize import _view, reference_materialize
from tests.stride.support.oracles.reduction import addresses, assert_close
from tests.stride.support.oracles.scalar import assert_components
from tests.stride.support.oracles.scale import reference_scale
from tests.stride.support.oracles.update import compare, update_functions
from tests.stride.support.samples import values
from tests.stride.support.views import _dense, _pitched


@pytest.mark.parametrize(
    "source_dtype,result_dtype",
    [
        (jnp.float32, jnp.float32),
        (jnp.complex64, jnp.complex64),
        (jnp.float16, jnp.float32),
    ],
)
def test_add_batched_zero_one_coefficients_jvp_vjp_and_mixed_derivatives(
    source_dtype: DTypeLike,
    result_dtype: DTypeLike,
) -> None:
    left = jnp.arange(36, dtype=jnp.float32).reshape(3, 12).astype(result_dtype)
    right = jnp.arange(18, dtype=jnp.float32).reshape(3, 2, 3).astype(source_dtype)
    if jnp.issubdtype(result_dtype, jnp.complexfloating):
        left = left * (1 - 0.5j)
        right = right * (1 + 0.25j)
    alpha = jnp.asarray([0, 1, -0.5], dtype=jnp.float32)
    beta = jnp.asarray([1, 0, 2], dtype=jnp.float32)
    indices = jnp.asarray([2, 6, 10, 3, 7, 11])

    def operation(old, new, lhs, rhs):
        return add(_pitched(old), _dense(new), alpha=lhs, beta=rhs).data

    def oracle(old, new, lhs, rhs):
        mapped = new.reshape(3, 6)
        return old.at[:, indices].set(
            (rhs[:, None] * mapped + lhs[:, None] * old[:, indices]).astype(old.dtype)
        )

    primals = (left, right, alpha, beta)
    tangents = tuple(jnp.ones_like(value) for value in primals)
    np.testing.assert_allclose(jax.vmap(operation)(*primals), oracle(*primals))
    for actual, expected in zip(
        jax.jvp(operation, primals, tangents),
        jax.jvp(oracle, primals, tangents),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)
    _, actual_pullback = jax.vjp(operation, *primals)
    _, expected_pullback = jax.vjp(oracle, *primals)
    cotangent = jnp.ones_like(left)
    for actual, expected in zip(
        actual_pullback(cotangent), expected_pullback(cotangent), strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)
    actual_gradient = jax.grad(
        lambda *arguments: jnp.real(jnp.sum(operation(*arguments))), argnums=2,
    )
    expected_gradient = jax.grad(
        lambda *arguments: jnp.real(jnp.sum(oracle(*arguments))), argnums=2,
    )
    for actual, expected in zip(
        jax.jvp(actual_gradient, primals, tangents),
        jax.jvp(expected_gradient, primals, tangents),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("conjugate_left", [False, True])
def test_native_complex_dot_jvp_vjp_and_higher_order(
    conjugate_left: bool,
) -> None:
    left = jnp.asarray(
        [[1 + 2j, -3 + 1j, 2 - 4j], [5 + 0j, -1 - 2j, 3 + 3j]],
        dtype=jnp.complex64,
    )
    right = jnp.asarray(
        [complex(real / 2, -real / 3) for real in range(12)],
        dtype=jnp.complex64,
    )
    left_tangent = jnp.asarray(left * (0.25 - 0.5j), dtype=jnp.complex64)
    right_tangent = jnp.asarray(right * (-0.75 + 0.125j), dtype=jnp.complex64)

    def operation(left_data: jax.Array, right_data: jax.Array) -> jax.Array:
        left_view = _dense(left_data)
        right_view = _pitched(right_data)
        return (
            dotc(left_view, right_view)
            if conjugate_left
            else dotu(left_view, right_view)
        )

    def oracle(left_data: jax.Array, right_data: jax.Array) -> jax.Array:
        left_value = jnp.conj(left_data) if conjugate_left else left_data
        return jnp.sum(left_value * materialize(_pitched(right_data)))

    actual_primal, actual_tangent = jax.jvp(
        operation,
        (left, right),
        (left_tangent, right_tangent),
    )
    expected_primal, expected_tangent = jax.jvp(
        oracle,
        (left, right),
        (left_tangent, right_tangent),
    )
    np.testing.assert_allclose(actual_primal, expected_primal)
    np.testing.assert_allclose(actual_tangent, expected_tangent, rtol=1e-6)

    cotangent = jnp.asarray(0.75 - 1.25j, dtype=jnp.complex64)
    _, actual_pullback = jax.vjp(operation, left, right)
    _, expected_pullback = jax.vjp(oracle, left, right)
    for actual, expected in zip(
        actual_pullback(cotangent),
        expected_pullback(cotangent),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def actual_first_tangent(
        left_data: jax.Array,
        right_data: jax.Array,
    ) -> jax.Array:
        return jax.jvp(
            operation,
            (left_data, right_data),
            (left_tangent, right_tangent),
        )[1]

    def expected_first_tangent(
        left_data: jax.Array,
        right_data: jax.Array,
    ) -> jax.Array:
        return jax.jvp(
            oracle,
            (left_data, right_data),
            (left_tangent, right_tangent),
        )[1]

    actual_second = jax.jvp(
        actual_first_tangent,
        (left, right),
        (left_tangent, right_tangent),
    )[1]
    expected_second = jax.jvp(
        expected_first_tangent,
        (left, right),
        (left_tangent, right_tangent),
    )[1]
    np.testing.assert_allclose(actual_second, expected_second, rtol=1e-6)


def test_native_add_jvp_and_vjp_match_dense_oracle() -> None:
    source = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) + 1
    destination = jnp.arange(12, dtype=jnp.float32)
    alpha = jnp.asarray(0.5, dtype=jnp.float32)
    beta = jnp.asarray(-2, dtype=jnp.float32)
    source_tangent = jnp.linspace(-1, 1, 6, dtype=jnp.float32).reshape(2, 3)
    destination_tangent = jnp.linspace(1, -1, 12, dtype=jnp.float32)
    alpha_tangent = jnp.asarray(0.25, dtype=jnp.float32)
    beta_tangent = jnp.asarray(-0.125, dtype=jnp.float32)

    def operation(
        coefficient: jax.Array,
        source_data: jax.Array,
        destination_coefficient: jax.Array,
        destination_data: jax.Array,
    ) -> jax.Array:
        return materialize(
            add(
                _pitched(destination_data),
                _dense(source_data),
                alpha=destination_coefficient,
                beta=coefficient,
            )
        )

    def oracle(
        coefficient: jax.Array,
        source_data: jax.Array,
        destination_coefficient: jax.Array,
        destination_data: jax.Array,
    ) -> jax.Array:
        return (
            coefficient * source_data
            + destination_coefficient * materialize(_pitched(destination_data))
        )

    primals = (alpha, source, beta, destination)
    tangents = (
        alpha_tangent,
        source_tangent,
        beta_tangent,
        destination_tangent,
    )
    actual_primal, actual_tangent = jax.jvp(operation, primals, tangents)
    expected_primal, expected_tangent = jax.jvp(oracle, primals, tangents)
    np.testing.assert_allclose(actual_primal, expected_primal)
    np.testing.assert_allclose(actual_tangent, expected_tangent, rtol=1e-6)

    cotangent = jnp.linspace(-2, 2, 6, dtype=jnp.float32).reshape(2, 3)
    _, actual_pullback = jax.vjp(operation, *primals)
    _, expected_pullback = jax.vjp(oracle, *primals)
    for actual, expected in zip(
        actual_pullback(cotangent),
        expected_pullback(cotangent),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected)

    actual_alpha_gradient = jax.grad(
        lambda *arguments: jnp.sum(operation(*arguments) ** 2),
        argnums=0,
    )
    expected_alpha_gradient = jax.grad(
        lambda *arguments: jnp.sum(oracle(*arguments) ** 2),
        argnums=0,
    )
    actual_gradient, actual_gradient_tangent = jax.jvp(
        actual_alpha_gradient,
        primals,
        tangents,
    )
    expected_gradient, expected_gradient_tangent = jax.jvp(
        expected_alpha_gradient,
        primals,
        tangents,
    )
    np.testing.assert_allclose(actual_gradient, expected_gradient)
    np.testing.assert_allclose(
        actual_gradient_tangent,
        expected_gradient_tangent,
        rtol=1e-6,
    )


@pytest.fixture(autouse=False)
def coefficient_enable_x64():
    with jax.enable_x64():
        yield


INEXACT = ('float16', 'bfloat16', 'float32', 'float64', 'complex64', 'complex128')


@pytest.mark.usefixtures("coefficient_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('operation,dtype,source_dtype', [
    (operation, dtype, source) for operation in ('copy', 'reduce', 'empty_axes')
    for dtype in DISCRETE for source in INEXACT
    if source == 'float32' or dtype in ('bool', 'int32', 'uint32')
])
def test_public_discrete_results(source_dtype, dtype, operation):
    source = jnp.asarray([1, 2, 3], dtype=source_dtype)
    if jnp.iscomplexobj(source):
        source = source + 1j

    def run(data):
        view = StridedView(data, (2, 2), (1, -1), 1)
        if operation == 'copy':
            return materialize(view, dtype=dtype)
        return reduce_sum(view, axes=() if operation == 'empty_axes' else None, dtype=dtype)
    check(run, (source,))


@pytest.mark.usefixtures("coefficient_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('dtype', ['bool', 'int32'])
@pytest.mark.parametrize('operation', ['copy', 'reduce', 'empty_axes'])
def test_discrete_conversion_composed_derivatives(dtype, operation):

    def loss(data):
        view = StridedView(data, (3,), (1,), 0)
        result = materialize(view, dtype=dtype) if operation == 'copy' else reduce_sum(view, axes=() if operation == 'empty_axes' else None, dtype=dtype)
        return jnp.sum(result.astype(jnp.float32))
    data = jnp.asarray([1, 2, 3], dtype=jnp.float32)
    gradient = jax.grad(loss)
    np.testing.assert_array_equal(jax.jit(gradient)(data), jnp.zeros_like(data))
    np.testing.assert_array_equal(jax.jit(jax.jacfwd(gradient))(data), jnp.zeros((3, 3)))
    for count in (0, 2):
        batch = jnp.broadcast_to(data, (count, 3))
        np.testing.assert_array_equal(jax.jit(jax.vmap(gradient))(batch), jnp.zeros_like(batch))
    assert 'tensor0_stride_accumulation_' not in jax.jit(gradient).lower(data).as_text()


@pytest.mark.usefixtures("coefficient_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('dtype', DISCRETE)
def test_integer_inputs_keep_float0_tangents(dtype):
    source = jnp.asarray([0, 1, 1], dtype=dtype)
    direction = np.zeros(source.shape, dtype=jax.dtypes.float0)
    for operation in (materialize, reduce_sum):
        run = lambda data: operation(StridedView(data, (3,), (1,), 0))
        result, tangent = jax.jvp(run, (source,), (direction,))
        assert tangent.dtype == jax.dtypes.float0 and tangent.shape == result.shape
        gradient = jax.vjp(run, source)[1](np.zeros(result.shape, dtype=jax.dtypes.float0))[0]
        assert gradient.dtype == jax.dtypes.float0 and gradient.shape == source.shape


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32,
                                    jnp.float64, jnp.complex64, jnp.complex128])
@pytest.mark.parametrize("factor", [0, 1])
def test_scale_zero_one_coefficient_vjp_across_inexact_dtypes(dtype, factor):
    with jax.enable_x64():
        values = jnp.asarray([1, 2, 3], dtype=dtype)
        coefficient = jnp.asarray(factor, dtype=dtype)
        function = lambda value: scale(StridedView(values, (2,), (1,), 0), value).data
        transpose = jax.jit(lambda value: jax.vjp(function, value)[1](
            jnp.ones_like(values))[0])
        np.testing.assert_array_equal(transpose(coefficient), jnp.asarray(3, dtype=dtype))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32,
                                    jnp.float64, jnp.complex64, jnp.complex128])
def test_complete_contiguous_scale_tangents_use_each_batch_coefficient(dtype):
    with jax.enable_x64():
        values = jnp.arange(12, dtype=jnp.float32).reshape(3, 4).astype(dtype)
        factors = jnp.asarray([0, 1, 2], dtype=dtype)
        factor_tangents = jnp.asarray([1, 0, -1], dtype=dtype)
        function = lambda data, coefficient: scale(StridedView(data, (4,), (1,), 0), coefficient).data
        primal, tangent = jax.jit(lambda data, coefficient: jax.jvp(
            function, (data, coefficient), (jnp.ones_like(data), factor_tangents),
        ))(values, factors)
        np.testing.assert_array_equal(primal, values * factors[:, None])
        np.testing.assert_array_equal(tangent, factors[:, None]
                                      + values * factor_tangents[:, None])
        coefficient_vjp = jax.jit(lambda data, coefficient: jax.vjp(
            function, data, coefficient)[1](jnp.ones_like(data))[1])(values, factors)
        np.testing.assert_array_equal(coefficient_vjp, values.sum(-1))


def test_vmap_preserves_existing_batch_axes_for_scalar_coefficients():
    values = jnp.arange(24, dtype=jnp.float32).reshape(3, 2, 4)
    factors = jnp.asarray([0, 1, 2], dtype=jnp.float32)
    function = lambda data, factor: scale(StridedView(data, (4,), (1,), 0), factor).data
    mapped = jax.jit(jax.vmap(function))
    nested = jax.jit(jax.vmap(jax.vmap(function, in_axes=(0, None))))
    expected = values * factors[:, None, None]
    np.testing.assert_array_equal(mapped(values, factors), expected)
    np.testing.assert_array_equal(nested(values, factors), expected)
    shared = jax.jit(jax.vmap(lambda data: function(data, 2)))
    np.testing.assert_array_equal(shared(values), values * 2)
    for operation in (mapped, nested):
        tangent = jax.jvp(lambda coefficient: operation(values, coefficient),
                          (factors,), (jnp.ones_like(factors),))[1]
        np.testing.assert_array_equal(tangent, values)
        gradient = jax.grad(lambda coefficient: operation(values, coefficient).sum())(factors)
        np.testing.assert_array_equal(gradient, values.sum((1, 2)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
@pytest.mark.parametrize("factor", [0, 1])
@pytest.mark.parametrize("explicit", [False, True])
def test_scale_derivative_zero_one_finite_values(
    dtype, factor, explicit,
):
    data = jnp.asarray([2, 3, 4, 5], dtype=dtype)
    direction = jnp.asarray([complex(1.25, -.5), complex(-3.5, .25),
                             complex(-0.0, -0.0), complex(2, 0)]
                            if dtype == jnp.complex64 else
                            [1.25, -3.5, -0.0, 2], dtype=dtype)
    coefficient = jnp.asarray(factor, dtype=dtype)
    def operation(values, value):
        return scale(StridedView(values, (2,), (2,), 0), value).data

    def derivative(values, tangent, value):
        if explicit:
            return jax.jvp(operation, (values, value),
                           (tangent, jnp.zeros_like(value)))[1]
        return jax.jvp(lambda current: operation(current, value),
                       (values,), (tangent,))[1]

    compiled = jax.jit(derivative).lower(data, direction, coefficient).compile()
    actual = compiled(data, direction, coefficient).block_until_ready()
    selected = coefficient * direction[::2]
    if explicit:
        selected = selected + jnp.zeros_like(coefficient) * data[::2]
    expected = direction.at[::2].set(selected)
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imag, expected.imag)


@pytest.mark.parametrize("factor", [0, 1])
def test_explicit_zero_coefficient_tangent_preserves_finite_derivative(factor):
    data = jnp.asarray([2, 7, -3, 9], dtype=jnp.float32)
    direction = jnp.ones_like(data)
    coefficient = jnp.asarray(factor, dtype=data.dtype)
    def operation(values, value):
        return scale(StridedView(values, (2,), (2,), 0), value).data

    single_product = lambda values, value: jax.jvp(
        lambda current: operation(current, value), (values,), (direction,),
    )[1]
    two_products = lambda values, value: jax.jvp(
        operation, (values, value), (direction, jnp.zeros_like(value)),
    )[1]
    single = jax.jit(single_product)(data, coefficient)
    double = jax.jit(two_products)(data, coefficient)
    np.testing.assert_array_equal(single, direction.at[::2].set(factor))
    np.testing.assert_array_equal(double, single)


@pytest.mark.parametrize("factor", [0, 1])
def test_closed_python_scale_factor_keeps_finite_complex_derivative(factor):
    data = jnp.ones(4, dtype=jnp.complex64)
    tangent = jnp.full(4, complex(2, -3), dtype=data.dtype)
    operation = lambda values: scale(StridedView(values, (4,), (1,), 0), factor).data
    actual = jax.jit(lambda values, direction:
                     jax.jvp(operation, (values,), (direction,))[1])(data, tangent)
    expected = jnp.asarray(factor, dtype=data.dtype) * tangent
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imag, expected.imag)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32,
                                    jnp.float64, jnp.complex64, jnp.complex128])
def test_single_scale_tangent_batching_and_nested_ad(dtype):
    with jax.enable_x64():
        data = jnp.arange(24, dtype=jnp.float32).reshape(3, 2, 4).astype(dtype)
        factors = jnp.asarray([0, 1, 2], dtype=dtype)
        operation = lambda values, factor: scale(StridedView(values, (2,), (2,), 0), factor).data
        mapped = jax.jit(jax.vmap(operation))
        actual = mapped(data, factors)
        expected = data.at[..., ::2].set(data[..., ::2] * factors[:, None, None])
        np.testing.assert_array_equal(actual, expected)
        direction = jnp.ones_like(data)
        first = lambda values, coefficient: jax.jvp(
            mapped, (values, coefficient), (direction, jnp.ones_like(coefficient)),
        )[1]
        second = jax.jit(lambda values, coefficient: jax.jvp(
            first, (values, coefficient), (direction, jnp.ones_like(coefficient)),
        )[1])(data, factors)
        np.testing.assert_array_equal(second, jnp.zeros_like(data).at[..., ::2].set(2))


@pytest.mark.parametrize("factor", [0, 1])
def test_selected_scale_data_transpose_preserves_unselected_cotangent(factor):
    data = jnp.ones(4, dtype=jnp.complex64)
    cotangent = jnp.asarray([complex(2, -3)] * 4, dtype=data.dtype)
    coefficient = jnp.asarray(factor, dtype=data.dtype)
    transpose = jax.linear_transpose(
        lambda values: scale(StridedView(values, (2,), (2,), 0), coefficient).data, data,
    )
    actual = jax.jit(transpose)(cotangent)[0]
    expected = cotangent.at[::2].set(coefficient * cotangent[::2])
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imag, expected.imag)


def test_materialize_jit_jvp_vjp_linear_transpose_and_vmap_match_reference() -> None:
    source = jnp.linspace(-1, 1, 20, dtype=jnp.float32)
    tangent = jnp.linspace(1, 2, 20, dtype=jnp.float32)
    cotangent = jnp.linspace(-2, 1, 6, dtype=jnp.float32).reshape(2, 3)

    def migrated(value):
        return materialize(
            _view(value, (2, 3), (1, 4), 2),
        )

    def reference(value):
        return reference_materialize(value, (2, 3), (1, 4), 2)

    migrated_primal, migrated_tangent = jax.jvp(
        migrated,
        (source,),
        (tangent,),
    )
    reference_primal, reference_tangent = jax.jvp(
        reference,
        (source,),
        (tangent,),
    )
    migrated_vjp = jax.vjp(migrated, source)[1](cotangent)[0]
    reference_vjp = jax.vjp(reference, source)[1](cotangent)[0]
    migrated_transpose = jax.linear_transpose(
        migrated,
        jnp.zeros_like(source),
    )(cotangent)[0]
    reference_transpose = jax.linear_transpose(
        reference,
        jnp.zeros_like(source),
    )(cotangent)[0]
    batch = jnp.stack((source, 2 * source, -source))

    np.testing.assert_array_equal(migrated_primal, reference_primal)
    np.testing.assert_array_equal(migrated_tangent, reference_tangent)
    np.testing.assert_array_equal(migrated_vjp, reference_vjp)
    np.testing.assert_array_equal(migrated_transpose, reference_transpose)
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(migrated))(batch),
        jax.jit(jax.vmap(reference))(batch),
    )


def test_mixed_materialize_jit_jvp_vjp_and_vmap_match_explicit_cast() -> None:
    source = jnp.linspace(-1, 1, 6, dtype=jnp.float32)
    tangent = jnp.linspace(1, 2, 6, dtype=jnp.float32)
    cotangent = (
        jnp.linspace(-2, 1, 6, dtype=jnp.float32)
        + 1j * jnp.linspace(3, -1, 6, dtype=jnp.float32)
    ).reshape(2, 3)

    def migrated(value):
        return materialize(
            _view(value, (2, 3), (3, 1), 0),
            dtype=jnp.complex64,
        )

    def explicit(value):
        return reference_materialize(value, (2, 3), (3, 1), 0).astype(
            jnp.complex64
        )

    actual_primal, actual_tangent = jax.jvp(migrated, (source,), (tangent,))
    expected_primal, expected_tangent = jax.jvp(explicit, (source,), (tangent,))
    actual_vjp = jax.vjp(migrated, source)[1](cotangent)[0]
    expected_vjp = jax.vjp(explicit, source)[1](cotangent)[0]
    batch = jnp.stack((source, 2 * source, -source))

    np.testing.assert_array_equal(actual_primal, expected_primal)
    np.testing.assert_array_equal(actual_tangent, expected_tangent)
    np.testing.assert_array_equal(actual_vjp, expected_vjp)
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(migrated))(batch),
        jax.jit(jax.vmap(explicit))(batch),
    )


@pytest.mark.parametrize("strides", [(0, 1), (1, 1)])
def test_materialize_repeated_reads_accumulate_source_gradient(strides) -> None:
    source = jnp.arange(8, dtype=jnp.float32)
    cotangent = jnp.array([[1., 2.], [3., 4.]])

    def operation(data):
        return materialize(StridedView(data, (2, 2), strides, 1))

    def reference(data):
        return reference_materialize(data, (2, 2), strides, 1)

    actual = jax.jit(lambda data: jax.vjp(operation, data)[1](cotangent)[0])(source)
    expected = jax.vjp(reference, source)[1](cotangent)[0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("strides,offset,source_size", [((3, 1), 0, 9), ((1, 1), 0, 5),
                                                       ((1, 0), 0, 3), ((-1, 0), 2, 3)])
@pytest.mark.parametrize("axes", [(0,), (1,), (0, 1)])
def test_public_reduction_repeated_reads_forward_jvp_vjp(strides, offset, source_size, axes):
    source = jnp.arange(source_size, dtype=jnp.float32) / 4
    indices = addresses((3, 3), strides, offset)
    native = lambda value: reduce_sum(StridedView(value, (3, 3), strides, offset), axes)
    reference = lambda value: jnp.sum(value[indices], axis=axes)
    actual = jax.jit(native)(source)
    assert_close(actual, reference(source))
    cotangent = jnp.ones_like(actual) * .5
    direction = jnp.ones_like(source) * .25
    for result, wanted in zip(jax.jvp(native, (source,), (direction,)), jax.jvp(reference, (source,), (direction,)), strict=True):
        assert_close(result, wanted)
    assert_close(jax.jit(lambda cot: jax.vjp(native, source)[1](cot)[0])(cotangent),
                 jax.vjp(reference, source)[1](cotangent)[0])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_public_sum_low_precision_complex_result_ad():
    primal, tangent = jax.jvp(
        lambda source: reduce_sum(StridedView.from_dense(source, (3,)), dtype="complex64"),
        (jnp.ones(3, dtype=jnp.float16),), (jnp.ones(3, dtype=jnp.float16),))
    assert primal.dtype == tangent.dtype == jnp.complex64
    np.testing.assert_array_equal(primal, 3)
    np.testing.assert_array_equal(tangent, 3)


REDUCTION_FLOATS = ("float16", "bfloat16", "float32", "float64")


REDUCTION_MIXED_PAIRS = [(source, result) for source in REDUCTION_FLOATS for result in REDUCTION_FLOATS if source != result] + [
    ("complex64", "complex128"), ("complex128", "complex64")]


# Keep every conversion on negative strides; selected pairs anchor other layouts.
# Original row indices preserve survivor IDs across the reduced geometry crosses.
REDUCTION_BATCH_PAIRS = [("float16", "float32"), ("float64", "bfloat16"),
                         ("complex64", "complex128"), ("complex128", "complex64")]


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("shape,strides,offset,axes,source_dtype,result_dtype,batch_shape", [
    pytest.param(*layout, source, result, batch,
                 id=f"shape{li}-strides{li}-{layout[2]}-"
                    f"{'None' if layout[3] is None else f'axes{li}'}-"
                    f"{source}-{result}-batch_shape{ti}")
    for li, layout in enumerate([((2, 3), (3, -1), 2, (1,)),
        ((2, 3), (0, 1), 1, None), ((2, 2), (1, 1), 0, (0,)), ((2, 0), (1, 1), 0, (1,))])
    for ti, (source, result, batch) in enumerate(
        [(*pair, ()) for pair in REDUCTION_MIXED_PAIRS]
        + [(*pair, batch) for pair in REDUCTION_BATCH_PAIRS
           for batch in ((2, 3), (0,), (2, 0))])
    if li == 0 or (not batch and (source, result) in
        [*REDUCTION_BATCH_PAIRS, ("bfloat16", "float32")])
    or (source, result) == ("float16", "float32")
])
def test_mixed_reduction_reduce_sum_mixed_source_ad(source_dtype, result_dtype, batch_shape, shape, strides, offset, axes):
    with jax.enable_x64():
        source = (jnp.arange(prod(batch_shape) * 6, dtype=jnp.float32) % 11 / 8).astype(source_dtype).reshape((*batch_shape, 6))
        if source_dtype.startswith("complex"):
            source = source * (1 + .5j)
        indices = np.asarray([offset + sum(index * stride for index, stride in zip(coordinate, strides))
                              for coordinate in np.ndindex(shape)], dtype=np.int32).reshape(shape)
        logical_axes = tuple(range(len(shape))) if axes is None else axes
        reference_axes = tuple(len(batch_shape) + axis for axis in logical_axes)
        run = lambda values: reduce_sum(StridedView(values, shape, strides, offset), axes, dtype=result_dtype)
        oracle = lambda values: jnp.sum(values[..., indices], axis=reference_axes, dtype=result_dtype)
        tolerance = 8 * max(float(jnp.finfo(source_dtype).eps), float(jnp.finfo(result_dtype).eps))
        tangent = jnp.ones_like(source)
        primal, direction = jax.jit(lambda values: jax.jvp(run, (values,), (tangent,)))(source)
        expected_primal, expected_direction = jax.jvp(oracle, (source,), (tangent,))
        for actual, expected in ((primal, expected_primal), (direction, expected_direction)):
            assert actual.dtype == expected.dtype
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        cotangent = jnp.full(primal.shape, 1 + .5j if result_dtype.startswith("complex") else 1, dtype=result_dtype)
        expected = jax.vjp(oracle, source)[1](cotangent)[0]
        for reverse in (jax.vjp(run, source)[1], jax.linear_transpose(run, source)):
            actual = jax.jit(reverse)(cotangent)[0]
            assert actual.dtype == source.dtype
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_mixed_reduction_empty_input_with_nonempty_output_and_zero_tangent():
    source = jnp.zeros(0, dtype=jnp.float16)
    run = lambda values: reduce_sum(StridedView(values, (2, 0), (1, 1), 0), (1,), dtype=jnp.float32)
    np.testing.assert_array_equal(run(source), [0, 0])
    gradient = jax.vjp(run, source)[1](jnp.asarray([jnp.nan, jnp.inf]))[0]
    assert gradient.shape == (0,) and gradient.dtype == source.dtype
    _, tangent = jax.jvp(lambda values: run(jax.lax.stop_gradient(values)), (source,), (source,))
    np.testing.assert_array_equal(tangent, [0, 0])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_mixed_reduction_outer_vmap_and_nested_derivatives():
    source = (jnp.arange(10, dtype=jnp.float32) / 4).astype(jnp.float16).reshape(2, 5)
    run = jax.vmap(lambda values: reduce_sum(StridedView(values, (2, 3), (1, 1), 0), (1,), dtype=jnp.float32))
    oracle = lambda values: jnp.sum(values[..., jnp.asarray([[0, 1, 2], [1, 2, 3]])], axis=-1, dtype=jnp.float32)
    cotangent = jnp.ones((2, 2), dtype=jnp.float32)
    reverse = lambda values: jax.linear_transpose(run, source)(values)[0]
    np.testing.assert_allclose(jax.jit(reverse)(cotangent), jax.vjp(oracle, source)[1](cotangent)[0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(jax.jit(jax.linear_transpose(reverse, cotangent))(jnp.ones_like(source))[0],
                               oracle(jnp.ones_like(source)), rtol=1e-3, atol=1e-3)
    actual = jax.jacfwd(jax.grad(lambda values: jnp.sum(run(values)**2)))(source)
    expected = jax.jacfwd(jax.grad(lambda values: jnp.sum(oracle(values)**2)))(source)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("source_dtype,result_dtype", [("float32", "int32")])
def test_mixed_reduction_discrete_result_has_zero_tangent(source_dtype, result_dtype):
    source = jnp.ones(2, dtype=source_dtype)
    result, tangent = jax.jvp(
        lambda values: reduce_sum(StridedView(values, (2,), (1,), 0), dtype=result_dtype),
        (source,), (source,))
    np.testing.assert_array_equal(result, 2)
    assert tangent.dtype == jax.dtypes.float0 and tangent.shape == result.shape


@pytest.fixture(autouse=False)
def reduction_ad_enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("reduction_ad_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("source_dtype,dtype", [("float32", "complex128"), ("complex64", "float64")])
@pytest.mark.parametrize("shape,strides,offset", [((2, 2), (1, 1), 0), ((2, 2), (0, -1), 2), ((2, 0), (1, 1), 3)])
def test_public_sum_and_empty_reduction(source_dtype, dtype, shape, strides, offset):
    source = values((3,), source_dtype)
    def run(data):
        return reduce_sum(StridedView(data, shape, strides, offset), axes=(1,), dtype=dtype)
    def reference(data):
        logical = jnp.stack([data[offset + outer * strides[0] + inner * strides[1]]
                             for outer in range(shape[0]) for inner in range(shape[1])]) if shape[1] else jnp.zeros(0, source_dtype)
        output = logical.reshape(shape).sum(axis=1)
        return (jnp.real(output) if dtype.startswith("float") else output).astype(dtype)
    for actual, expected in zip(jax.jit(lambda data: jax.jvp(run, (data,), (data,)))(source),
                                jax.jvp(reference, (source,), (source,)), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    cotangent = jnp.ones_like(run(source))
    actual = jax.jit(jax.vjp(run, source)[1])(cotangent)[0]
    expected = jax.vjp(reference, source)[1](cotangent)[0]
    assert actual.dtype == source.dtype
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    frozen = lambda data: run(jax.lax.stop_gradient(data))
    np.testing.assert_array_equal(jax.jvp(frozen, (source,), (source,))[1], jnp.zeros(shape[0], dtype))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_public_scale_batched_ad_and_nested_pullback():
    base = jnp.linspace(-2, 3, 24, dtype=jnp.float32).reshape(2, 12)
    factors = jnp.asarray([1.25, -.75], dtype=jnp.float32)
    directions = (jnp.linspace(1, -1, 24, dtype=jnp.float32).reshape(2, 12), jnp.asarray([-.5, .25]))
    cotangent = jnp.linspace(-3, 2, 24, dtype=jnp.float32).reshape(2, 12)
    native = lambda old, factor: scale(StridedView(old, (5,), (2,), 1), factor).data
    reference = lambda old, factor: old.at[:, 1:11:2].set(old[:, 1:11:2] * factor[:, None])

    def derivatives(function, old, factor):
        forward = jax.jvp(function, (old, factor), directions)
        reverse = jax.vjp(function, old, factor)[1](cotangent)
        pullback = lambda cot: jax.vjp(lambda value: function(value, factor), old)[1](cot)[0]
        nested = jax.jvp(pullback, (cotangent,), (jnp.ones_like(cotangent),))
        return forward, reverse, nested

    actual = jax.jit(lambda old, factor: derivatives(native, old, factor))(base, factors)
    expected = derivatives(reference, base, factors)
    for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(result, wanted, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("coefficient_dtype", [jnp.float32, jnp.float64, jnp.complex128])
def test_mixed_scale_first_and_higher_derivatives(coefficient_dtype):
    with jax.enable_x64():
        base, coefficient = jnp.asarray([1.25, 2.5, 3.75, 4.25, 5.5], dtype=jnp.float32), jnp.asarray(1.25, dtype=coefficient_dtype)
        native = lambda data, factor: scale(StridedView(data, (2,), (2,), 1), factor).data
        reference = lambda data, factor: data.at[1::2].set(jnp.real(factor * data[1::2]).astype(data.dtype))
        for result, wanted in zip(jax.vjp(native, base, coefficient)[1](jnp.ones_like(base)),
                                   jax.vjp(reference, base, coefficient)[1](jnp.ones_like(base)), strict=True):
            assert_components(result, wanted)
        for explicit in (False, True):
            directions = (jnp.ones_like(base), jnp.ones_like(coefficient) if explicit else jnp.zeros_like(coefficient))
            derivative = lambda data, factor: jax.jvp(native, (data, factor), directions)[1]
            assert_components(jax.jit(derivative)(base, coefficient), jax.jvp(reference, (base, coefficient), directions)[1])
            second = jax.jvp(lambda factor: derivative(base, factor), (coefficient,), (jnp.ones_like(coefficient),))[1]
            assert_components(second, jnp.zeros_like(base).at[1::2].set(1))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("factor", [0, 1])
def test_strong_complex_zero_one_finite_data_derivative(factor):
    with jax.enable_x64():
        source = jnp.asarray([1 + 2j, 3 + 4j, 5 + 6j], dtype=jnp.complex64)
        direction = jnp.full_like(source, 1.25 - .5j)
        coefficient = jnp.asarray(factor, dtype=jnp.complex128)
        operation = lambda value: scale(StridedView(value, (3,), (1,), 0), coefficient).data
        actual = jax.jit(lambda value, dot: jax.jvp(operation, (value,), (dot,))[1])(source, direction)
        assert_components(actual, (coefficient * direction).astype(source.dtype))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("explicit", [False, True])
def test_scale_jvp_keeps_coefficient_precision_before_product(explicit):
    with jax.enable_x64():
        base = jnp.asarray([1., 2., 3.], dtype=jnp.float32)
        direction, coefficient = jnp.full_like(base, 1e30), jnp.float64(1e-46)
        operation = lambda data, factor: scale(StridedView(data, (3,), (1,), 0), factor).data
        def derivative(data, factor):
            if explicit:
                return jax.jvp(operation, (data, factor), (direction, jnp.zeros_like(factor)))[1]
            return jax.jvp(lambda value: operation(value, factor), (data,), (direction,))[1]
        actual = jax.jit(derivative)(base, coefficient)
        expected = (coefficient * direction).astype(base.dtype)
        np.testing.assert_array_equal(actual, expected)
        assert bool(jnp.all(actual != 0))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32,
                                   jnp.float64, jnp.complex64, jnp.complex128])
def test_selected_scale_factor_jvp_uses_one_accumulation(dtype):
    with jax.enable_x64():
        base = jnp.arange(50, dtype=jnp.float32).astype(dtype) - 7
        complex_storage = jnp.issubdtype(dtype, jnp.complexfloating)
        factor = jnp.asarray(1.25 - .5j if complex_storage else -1.25, dtype=dtype)
        direction = jnp.asarray(.75 + .25j if complex_storage else .75, dtype=dtype)
        operation = lambda value: scale(StridedView(base, (16,), (3,), 2), value).data
        tangent = jax.jit(lambda value, dot: jax.jvp(operation, (value,), (dot,))[1])
        expected = jax.jvp(lambda value: reference_scale(base, value), (factor,), (direction,))[1]
        np.testing.assert_allclose(tangent(factor, direction), expected, rtol=5e-3, atol=5e-3)
        text = tangent.lower(factor, direction).as_text()
        assert text.count("custom_call") == 1
        assert operation_target("accumulation", np.dtype(dtype)) in text


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
def test_selected_scale_ad_transposes_and_higher_order_compositions(dtype):
    base = jnp.linspace(-3, 4, 50, dtype=jnp.float32).astype(dtype)
    cotangent = jnp.linspace(-4, 3, 50, dtype=jnp.float32).astype(dtype)
    factor = jnp.asarray(-1.25 if dtype == jnp.float32 else 1.25 - .75j, dtype=dtype)
    if dtype == jnp.complex64:
        base = base * (1 + .5j)
        cotangent = cotangent * (.75 - 1.25j)
    directions = (jnp.full_like(base, .5), jnp.asarray(.75, dtype=dtype))
    operation = lambda old, value: scale(StridedView(old, (16,), (3,), 2), value).data

    def derivatives(function, old, value, cot):
        primal, tangent = jax.jvp(function, (old, value), directions)
        gradients = jax.vjp(function, old, value)[1](cot)
        base_transpose = jax.linear_transpose(lambda data: function(data, value), old)(cot)[0]
        factor_transpose = jax.linear_transpose(lambda coefficient: function(old, coefficient), value)(cot)[0]
        pullback = lambda output_cot: jax.vjp(lambda data: function(data, value), old)[1](output_cot)[0]
        nested = jax.jvp(pullback, (jnp.zeros_like(cot),), (jnp.ones_like(cot),))
        factor_gradient = lambda data: jax.vjp(lambda coefficient: function(data, coefficient), value)[1](cot)[0]
        mixed = jax.jvp(factor_gradient, (old,), (directions[0],))
        return primal, tangent, gradients, base_transpose, factor_transpose, nested, mixed

    actual = jax.jit(lambda old, value, cot: derivatives(operation, old, value, cot))(base, factor, cotangent)
    expected = derivatives(reference_scale, base, factor, cotangent)
    for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(result, wanted, rtol=2e-6, atol=2e-6)


@pytest.fixture(autouse=False)
def update_enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("update_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('source_dtype,base_dtype', INTEGER_PAIRS)
def test_integer_storage_has_zero_coefficient_tangent(source_dtype, base_dtype):
    source = StridedView(jnp.ones(2, dtype=source_dtype), (2,), (1,), 0)
    base = StridedView(jnp.ones(2, dtype=base_dtype), (2,), (1,), 0)
    run = lambda factor: add(base, source, beta=factor).data
    np.testing.assert_array_equal(run(jnp.float32(2)), jnp.full(2, 3, dtype=base_dtype))
    result, tangent = jax.jvp(run, (jnp.float32(2),), (jnp.float32(1),))
    assert tangent.dtype == jax.dtypes.float0 and tangent.shape == result.shape
    np.testing.assert_array_equal(jax.grad(lambda value: jnp.sum(run(value).astype(jnp.float32)))(jnp.float32(2)), 0)


@pytest.mark.usefixtures("update_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('dtype,coefficient_dtype', [('float32', 'complex128'), ('complex64', 'float64')])
def test_public_add_coefficient_order(dtype, coefficient_dtype):
    source, base = (values((6,), dtype), values((10,), dtype))
    alpha = jnp.asarray(2 + 1j if coefficient_dtype.startswith('complex') else 2, coefficient_dtype)
    beta = jnp.asarray(0.5, dtype)
    record = AffineRecord((3,), (0,), 2, (2,), 1)
    _, reference = update_functions((record,))

    def run(data, original, source_factor, base_factor):
        left = StridedView(original, (3,), (2,), 1)
        right = StridedView(data, (3,), (0,), 2)
        return add(left, right, alpha=base_factor, beta=source_factor).data
    compare(run, reference, (source, base, alpha, beta))


@pytest.mark.usefixtures("update_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_f16_fused_reduction_avoids_intermediate_storage_rounding():
    left, right = (jnp.zeros(3, dtype=jnp.float16), jnp.ones(1, dtype=jnp.float16))
    factor = jnp.float32(1.5)
    execute = lambda inputs: add(StridedView(left, (3,), (1,), 0), StridedView(inputs, (3,), (0,), 0), alpha=0, beta=factor).data
    cotangent = jnp.asarray([2048, 1, -2048], dtype=jnp.float16)
    actual = jax.jit(jax.vjp(execute, right)[1])(cotangent)[0]
    np.testing.assert_array_equal(actual, [1.5])
    fused = reduction_p.bind(cotangent, factor, records=(AffineRecord((3,), (1,), 0, (1,), 0),), output_shapes=((1,),), reduction_axes=((True,),), coefficient_records=(0,), output_size=1, dtype=jnp.dtype(jnp.float16))
    np.testing.assert_array_equal(fused, [1.5])


def add_ad_functions(shape, left_strides, left_offset, right_strides, right_offset, alpha, beta):
    left_indices = jnp.asarray([left_offset + sum((index * stride for index, stride in zip(coordinate, left_strides, strict=True))) for coordinate in np.ndindex(shape)], dtype=jnp.int32)
    right_indices = jnp.asarray([right_offset + sum((index * stride for index, stride in zip(coordinate, right_strides, strict=True))) for coordinate in np.ndindex(shape)], dtype=jnp.int32)

    def execute(left, right):
        return add(StridedView(left, shape, left_strides, left_offset), StridedView(right, shape, right_strides, right_offset), alpha=alpha, beta=beta).data

    def reference(left, right):
        left_factor, right_factor = (jnp.asarray(alpha), jnp.asarray(beta))
        if left_factor.ndim:
            left_factor = left_factor[..., None]
        if right_factor.ndim:
            right_factor = right_factor[..., None]
        left_values, right_values = (left[..., left_indices], right[..., right_indices])
        left_term = jnp.where(left_factor == 0, jnp.zeros_like(left_values), jnp.where(left_factor == 1, left_values, left_factor * left_values))
        right_term = jnp.where(right_factor == 0, jnp.zeros_like(right_values), jnp.where(right_factor == 1, right_values, right_factor * right_values))
        return left.at[..., left_indices].set((left_term + right_term).astype(left.dtype))
    return (execute, reference)


@pytest.mark.usefixtures("update_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_nonzero_stride_overlap_and_second_derivative():
    left, right = (jnp.ones(8), jnp.ones(3))
    execute, reference = add_ad_functions((2, 2, 2), (4, 2, 1), 0, (0, 1, 1), 0, 1, 2)
    np.testing.assert_array_equal(jax.jit(jax.grad(lambda inputs: execute(left, inputs).sum()))(right), [4, 8, 4])
    np.testing.assert_array_equal(jax.jit(jax.hessian(lambda inputs: (execute(left, inputs) ** 2).sum()))(right), jax.hessian(lambda inputs: (reference(left, inputs) ** 2).sum())(right))


@pytest.mark.usefixtures("update_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('factor', [0.0, 1.0, 2.0])
@pytest.mark.parametrize('empty', [False, True])
def test_integer_storage_scale_composed_derivatives(factor, empty):
    storage = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    view = StridedView(storage, (0 if empty else 2,), (1,), 1)

    def loss(coefficient):
        return jnp.sum(scale(view, coefficient).data.astype(jnp.float32))
    coefficient = jnp.float32(factor)
    gradient = jax.grad(loss)
    np.testing.assert_array_equal(jax.jit(gradient)(coefficient), 0)
    np.testing.assert_array_equal(jax.jit(jax.grad(gradient))(coefficient), 0)
    batch = jnp.asarray([0, 1, 2], dtype=jnp.float32)
    np.testing.assert_array_equal(jax.jit(jax.vmap(gradient))(batch), jnp.zeros_like(batch))
    text = jax.jit(gradient).lower(coefficient).as_text()
    assert 'tensor0_stride_dot_' not in text
    assert 'tensor0_stride_accumulation_' not in text
