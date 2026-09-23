"""Native coefficient composition, promotion and short circuits."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._dtype import normalize_coefficient
from tensor0._stride._ffi._calls import execute_accumulation, execute_reduction, execute_update
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._jax import accumulation_p, update_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.data import PAIRS
from tests.stride.support.ffi import REDUCTION_FIBER, _encode_reduction_records
from tests.stride.support.oracles.affine import broadcast_record, native
from tests.stride.support.oracles.coefficients import SHARED_CASES, shared_coefficient_case
from tests.stride.support.oracles.effective_factor import SINGLE
from tests.stride.support.oracles.update_storage import MIXED_RECORDS, mixed_operands, reference


@pytest.fixture(autouse=False)
def accumulation_enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("accumulation_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('tiny_source', [False, True])
def test_no_conversion_before_multiply(tiny_source):
    with jax.enable_x64():
        source = jnp.asarray([1e-46 if tiny_source else 1e+30], dtype=jnp.float64)
        factor = jnp.asarray(1e+30 if tiny_source else 1e-46, dtype=jnp.float64)
        layout = encode_layout((AffineRecord((), (), 0, (), 0),), source_size=1, output_size=1)
        result = execute_accumulation(source, (factor,), coefficient_records=(0,), layout=layout, output_size=1, dtype='float32')
        np.testing.assert_allclose(result, [np.float32(1e-16)], rtol=1e-06)


@pytest.mark.usefixtures("accumulation_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_integer_product_wraps_before_output_conversion():
    layout = encode_layout((AffineRecord((2,), (1,), 0, (1,), 0),), source_size=2, output_size=2)
    result = execute_accumulation(jnp.asarray([100, -100], dtype=jnp.int8), (jnp.int8(2),), coefficient_records=(0,), layout=layout, output_size=2, dtype='int32')
    np.testing.assert_array_equal(result, [-56, 56])


@pytest.mark.usefixtures("accumulation_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
def test_low_precision_product_rounding_precedes_widening(dtype):
    source = jnp.asarray([0.33325, 0.7, 1.01], dtype=dtype)
    factor = jnp.asarray(0.33325, dtype=dtype)
    layout = encode_layout((AffineRecord((3,), (1,), 0, (1,), 0),), source_size=3, output_size=3)
    expected = (source * factor).astype('float32')
    np.testing.assert_array_equal(execute_accumulation(source, (factor,), coefficient_records=(0,), layout=layout, output_size=3, dtype='float32'), expected)


@pytest.mark.usefixtures("accumulation_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_zero_and_one_follow_accumulation_binding():
    with jax.enable_x64():
        layout = encode_layout((AffineRecord((3,), (1,), 0, (1,), 0),), source_size=3, output_size=3)
        source = jnp.asarray([-0.0, np.inf, np.nan], dtype=jnp.float32)
        result = np.asarray(execute_accumulation(source, (jnp.float32(0),), coefficient_records=(0,), layout=layout, output_size=3, dtype='float64'))
        np.testing.assert_array_equal(result, np.zeros(3))
        integers = jnp.asarray([16777217, -16777217, 3], dtype=jnp.int32)
        np.testing.assert_array_equal(execute_accumulation(integers, (jnp.float32(1),), coefficient_records=(0,), layout=layout, output_size=3, dtype='int64'), [16777217, -16777217, 3])


@pytest.mark.usefixtures("accumulation_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_real_coefficient_form_is_preserved():
    with jax.enable_x64():
        layout = encode_layout((AffineRecord((1,), (1,), 0, (1,), 0),), source_size=1, output_size=1)
        source = jnp.asarray([complex(1, np.inf)], dtype=jnp.complex64)
        real_product = np.asarray(execute_accumulation(source, (jnp.float32(2),), coefficient_records=(0,), layout=layout, output_size=1, dtype='complex128'))
        complex_product = np.asarray(execute_accumulation(source, (jnp.complex64(2),), coefficient_records=(0,), layout=layout, output_size=1, dtype='complex128'))
        assert real_product[0].real == 2 and np.isposinf(real_product[0].imag)
        assert np.isnan(complex_product[0].real) and np.isposinf(complex_product[0].imag)


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_broadcast_real_complex_multiply_uses_real_operand_form():
    record, size = broadcast_record(shape=(8, 7), broadcast_axes=(1,), source_signs=(1,))
    source = jnp.asarray([0., -0., np.inf, -np.inf, np.nan, np.finfo(np.float32).max,
                          -np.finfo(np.float32).max, np.finfo(np.float32).tiny], dtype=jnp.float32)
    assert size == source.size
    actual = jax.jit(lambda value: native(value, record, complex(np.nan, 1), "complex64"))(source)
    assert np.isnan(np.asarray(actual.real)).all()
    np.testing.assert_array_equal(actual.imag.reshape(8, 7), jnp.broadcast_to(source[:, None], (8, 7)))
    assert np.signbit(np.asarray(actual.imag).reshape(8, 7)[1]).all()


@pytest.fixture(autouse=False)
def coefficient_enable_x64():
    with jax.enable_x64():
        yield


# Repeated types plus explicit mixed product/store stages, separately per operation.
SHARED_STAGE_TRIPLES = {
    ("float16", "bfloat16", "float32"),
    ("float32", "float16", "bfloat16"),
    ("complex64", "float64", "float16"),
    ("float16", "complex128", "float64"),
    ("bfloat16", "float16", "float32"),
    ("float32", "bfloat16", "float16"),
    ("complex64", "float64", "bfloat16"),
    ("bfloat16", "complex128", "float64"),
}


@pytest.mark.usefixtures("coefficient_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('operation,source_dtype,coefficient_dtype,dtype', [
    pytest.param(operation, source, coefficient, result,
                 id=f"{operation}-{source}-{coefficient}-{result}")
    for operation, source, coefficient, result in SHARED_CASES
    if operation == "update" or len({source, coefficient, result}) <= 2
    or (source, coefficient, result) in SHARED_STAGE_TRIPLES
])
def test_shared_coefficient_rules(source_dtype, coefficient_dtype, dtype, operation):
    run, reference, arguments = shared_coefficient_case(source_dtype, coefficient_dtype, dtype, operation)
    actual, expected = jax.jit(run)(*arguments), reference(*arguments)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


PARTIAL = (AffineRecord((3,), (1,), 0, (2,), 1),)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("first,second", [(2., .5), (0., 2.), (.75, 2.)])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.complex64])
def test_effective_factor_short_circuits_after_coefficient_product(first, second, dtype):
    source = jnp.asarray([0., 40000., -3.5], dtype=dtype)
    base = jnp.full((7,), -7., dtype=dtype)
    coefficient = jnp.asarray(second, dtype=dtype)

    def operation(old, values, factor):
        return update_p.bind(values, old, jnp.asarray(first, dtype=dtype) * factor,
                             jnp.asarray(0, dtype=dtype), records=PARTIAL)

    effective = first * second
    selected = jnp.zeros_like(source) if effective == 0 else source if effective == 1 else source * effective
    expected = base.at[1::2].set(selected)
    for function in (operation, jax.jit(operation)):
        np.testing.assert_allclose(function(base, source, coefficient), expected, rtol=1e-3, atol=0)
    text = jax.jit(operation).lower(base, source, coefficient).as_text()
    assert text.count("custom_call") == 1
    assert "tensor0_stride_update_" in text


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_mapping_and_update_receive_independent_coefficients():
    source = jnp.asarray([30000], dtype=jnp.float16)
    fresh = accumulation_p.bind(source, jnp.float16(2), records=SINGLE,
                                coefficient_records=(0,), output_size=1, dtype=source.dtype)
    updated = update_p.bind(source, jnp.zeros_like(source), jnp.float16(2) * jnp.float16(.5),
                            jnp.float16(0), records=SINGLE)
    np.testing.assert_allclose(fresh, source * jnp.float16(2), rtol=1e-3, atol=0)
    np.testing.assert_array_equal(updated, source)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("strong", [False, True])
def test_coefficient_product_precedes_weak_normalization(strong):
    with jax.enable_x64():
        first = np.float32(1.0003) if strong else 1.0003
        second = np.float32(1.0003) if strong else 1.0003
        source = jnp.ones((1,), dtype=jnp.float16)

        def operation(values, factor):
            effective = normalize_coefficient(values.dtype, jnp.asarray(first) * factor)
            return update_p.bind(values, jnp.zeros_like(values), effective, jnp.float16(0), records=SINGLE)

        product = jnp.asarray(first) * jnp.asarray(second)
        coefficient = product if strong else product.astype(source.dtype)
        expected = (source * coefficient).astype(source.dtype)
        for function in (operation, jax.jit(operation)):
            actual = function(source, jnp.asarray(second))
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
            assert float(actual[0]) > 1


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32, jnp.bool_])
def test_integer_coefficient_product_precedes_source_promotion(dtype):
    records = (AffineRecord((2,), (1,), 0, (1,), 0),)
    source = jnp.asarray([1, 3], dtype=dtype)
    actual = jax.jit(lambda values, factor: update_p.bind(
        values, jnp.zeros_like(values), jnp.int8(100) * factor, jnp.int8(0), records=records))(
            source, jnp.int8(2))
    expected = jnp.asarray([True, True] if dtype == jnp.bool_ else [-56, -168], dtype=dtype)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_boolean_coefficient_product_selects_zero_without_reading_source():
    source = jnp.asarray([jnp.nan], dtype=jnp.float32)
    actual = jax.jit(lambda values, factor: update_p.bind(
        values, jnp.zeros_like(values), jnp.asarray(False) * factor, jnp.float32(0), records=SINGLE))(
            source, jnp.float32(3.5))
    np.testing.assert_array_equal(actual, jnp.zeros_like(actual))


DTYPES = ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
          "float16", "bfloat16", "float32", "float64", "complex64", "complex128")


UPDATE_DTYPES = tuple((dtype, dtype) for dtype in DTYPES) + (
    ("float16", "float32"), ("float32", "float16"),
    ("float32", "complex64"), ("complex64", "float32"),
    ("float64", "complex128"), ("complex128", "float64"),
    ("int32", "bool"), ("int32", "int8"), ("int32", "int16"), ("uint32", "uint8"),
)


def test_effective_factor_is_composed_before_source_multiplication():
    records = (AffineRecord((1,), (1,), 0, (1,), 0),)
    base = jnp.asarray([jnp.nan], dtype=jnp.float16)
    source = jnp.asarray([40000], dtype=jnp.float16)
    result = jax.jit(lambda factor: update_p.bind(
        source, base, factor * 2, jnp.int32(0), records=records,
    ))(jnp.asarray(0.5, dtype=jnp.float16))
    np.testing.assert_array_equal(result, source)
    np.testing.assert_array_equal(update_p.bind(
        source, base, jnp.int32(0), jnp.int32(0), records=records,
    ), jnp.zeros_like(base))


@pytest.mark.parametrize("source_dtype,result_dtype", UPDATE_DTYPES)
@pytest.mark.parametrize("factor", [0, 1, 2])
def test_dynamic_update_preserves_native_dtype_matrix(source_dtype, result_dtype, factor):
    with jax.enable_x64():
        records = (AffineRecord((3,), (1,), 0, (1,), 0),)
        base = jnp.asarray([1, 2, 3], dtype=result_dtype)
        source = jnp.asarray([4, 5, 6], dtype=source_dtype)
        coefficient = jnp.asarray(factor, dtype=result_dtype)
        expected = jnp.real((coefficient * 2) * source + base).astype(result_dtype)
        function = jax.jit(lambda value: update_p.bind(
            source, base, value * 2, jnp.int32(1), records=records,
        ))
        output = function(coefficient)
        output.block_until_ready()
        assert function.lower(coefficient).as_text().count("stablehlo.custom_call") == 1
        np.testing.assert_array_equal(output, expected)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_mapped_value_is_not_narrowed_before_add():
    with jax.enable_x64():
        records = tuple(dict(REDUCTION_FIBER, source_shape=(1,), source_offset=index) for index in range(2))
        layout = _encode_reduction_records(records, source_size=2, output_size=1)
        result = execute_reduction(jnp.asarray([-1, 1], dtype=jnp.float32),
                                   (jnp.float64(1 + 2**-24),), coefficient_records=(1,),
                                   layout=layout, output_size=1)
        np.testing.assert_array_equal(result, [np.float32(2**-24)])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_unit_skips_promotion_and_zero_skips_input():
    with jax.enable_x64():
        layout = _encode_reduction_records((dict(REDUCTION_FIBER, source_shape=(1,)),), source_size=1, output_size=1)
        source = jnp.asarray([16777217], dtype=jnp.int32)
        np.testing.assert_array_equal(execute_reduction(source, layout=layout, output_size=1, dtype="int64"), [16777217])
        np.testing.assert_array_equal(execute_reduction(source, (jnp.float32(1),), coefficient_records=(0,),
                                                       layout=layout, output_size=1, dtype="int64"), [16777217])
        result = execute_reduction(jnp.asarray([np.inf], dtype=jnp.float32), (jnp.float32(0),),
                                   coefficient_records=(0,), layout=layout, output_size=1)
        np.testing.assert_array_equal(result, [0])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_integer_mapping_wrap_and_final_conversion():
    layout = _encode_reduction_records((dict(REDUCTION_FIBER, source_shape=(2,)),), source_size=2, output_size=1)
    mapped = execute_reduction(jnp.asarray([100, 100], dtype=jnp.int8), (jnp.int8(2),),
                               coefficient_records=(0,), layout=layout, output_size=1, dtype="int32")
    np.testing.assert_array_equal(mapped, [-112])
    result = execute_reduction(jnp.asarray([.9, .9], dtype=jnp.float32), layout=layout,
                               output_size=1, dtype="int32")
    np.testing.assert_array_equal(result, [1])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_low_precision_mapping_rounds_before_wider_sum(dtype):
    source = jnp.asarray([.33325, .7, 1.01], dtype=dtype)
    coefficient = jnp.asarray(.33325, dtype=dtype)
    layout = _encode_reduction_records((REDUCTION_FIBER,), source_size=3, output_size=1)
    expected = jnp.sum((source * coefficient).astype(jnp.float32))
    result = execute_reduction(source, (coefficient,), coefficient_records=(0,),
                               layout=layout, output_size=1, dtype="float32")
    np.testing.assert_array_equal(result, np.asarray(expected).reshape(1))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_high_rank_and_real_coefficient():
    with jax.enable_x64():
        record = dict(source_shape=(1,) * 12, source_strides=(2**63 - 1,) * 12, source_offset=0, output_shape=(1,) * 12, output_strides=(2**63 - 1,) * 12, output_offset=0, reduction_axes=(True,) * 12)
        layout = _encode_reduction_records((record,), source_size=1, output_size=1)
        source = jnp.asarray([complex(1, np.inf)], dtype=jnp.complex64)
        actual = np.asarray(execute_reduction(source, (jnp.float32(2),), coefficient_records=(0,),
                                              layout=layout, output_size=1, dtype="complex128"))
        assert actual[0].real == 2 and np.isposinf(actual[0].imag)


@pytest.fixture(autouse=False)
def update_enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("update_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
# Preserve scalar branches; anchor batched branches and empty descriptors.
@pytest.mark.parametrize('source_dtype,base_dtype,batch_shape,alpha,beta', [
    pytest.param(source, base, batch, alpha, beta,
                 id=f"{alpha}-{beta}-batch_shape{index}-{source}-{base}")
    for alpha, beta in product((0, 1, 2), repeat=2)
    for index, batch in enumerate([(), (2, 3), (0,), (2, 0)])
    for source, base in PAIRS
    if not batch
    or (batch == (2, 3) and ((alpha, beta) == (2, 2)
        or (source, base) == ("float16", "float32")))
    or (batch in ((0,), (2, 0)) and (alpha, beta) == (2, 2)
        and (source, base) in (("float16", "float32"),
                              ("complex64", "float32"), ("uint32", "uint8")))
])
def test_mixed_storage_branches_records_and_batches(source_dtype, base_dtype, batch_shape, alpha, beta):
    with jax.enable_x64():
        source, base = mixed_operands(source_dtype, base_dtype, batch_shape)
        layout = encode_layout(MIXED_RECORDS, source_size=5, output_size=10)
        run = jax.jit(lambda values, original, first, second: execute_update(values, original, first, second, layout=layout))
        result = run(source, base, jnp.float32(alpha), jnp.full(batch_shape, beta, dtype=jnp.int32))
        assert result.dtype == base.dtype and result.shape == base.shape
        np.testing.assert_allclose(result, reference(source, base, alpha, beta), rtol=1e-06, atol=1e-06)


@pytest.mark.usefixtures("update_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_mixed_complex_term_is_converted_after_addition():
    source = jnp.asarray([1 + 2j], jnp.complex64)
    base = jnp.asarray([3], jnp.float32)
    layout = encode_layout((AffineRecord((1,), (1,), 0, (1,), 0),), source_size=1, output_size=1)
    result = execute_update(source, base, jnp.complex64(1j), jnp.float32(1), layout=layout)
    np.testing.assert_array_equal(result, [1])


@pytest.mark.usefixtures("update_enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('alpha,beta', [(0, 0), (0, 2), (2, 0)])
def test_mixed_zero_terms_skip_nonfinite_inputs(alpha, beta):
    source = jnp.asarray([np.nan if alpha == 0 else 2], jnp.complex64)
    base = jnp.asarray([np.inf if beta == 0 else 3], jnp.float32)
    layout = encode_layout((AffineRecord((1,), (1,), 0, (1,), 0),), source_size=1, output_size=1)
    np.testing.assert_array_equal(execute_update(source, base, jnp.int32(alpha), jnp.int32(beta), layout=layout), [alpha * 2 + beta * 3])
