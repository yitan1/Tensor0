import jax
import jax.numpy as jnp

from tensor0.tensor import _blocks


def test_strided_indices_values_match_manual_formula():
    indices = _blocks.strided_indices((2, 3), (1, 4), 2)

    assert indices.tolist() == [2, 6, 10, 3, 7, 11]


def test_strided_indices_rank3_values_match_manual_formula():
    _blocks._clear_strided_indices_cache_for_tests()

    indices = _blocks.strided_indices((2, 2, 3), (6, 1, 2), 5)

    assert indices.tolist() == [
        5,
        7,
        9,
        6,
        8,
        10,
        11,
        13,
        15,
        12,
        14,
        16,
    ]


def test_strided_indices_reuses_cached_array_for_same_key():
    _blocks._clear_strided_indices_cache_for_tests()

    first = _blocks.strided_indices((2, 3), (1, 4), 2)
    second = _blocks.strided_indices((2, 3), (1, 4), 2)

    assert first is second


def test_strided_indices_cache_is_bounded():
    _blocks._clear_strided_indices_cache_for_tests()
    original_maxsize = _blocks._STRIDED_INDICES_CACHE_MAXSIZE
    _blocks._STRIDED_INDICES_CACHE_MAXSIZE = 2
    try:
        first = _blocks.strided_indices((1,), (1,), 0)
        _blocks.strided_indices((1,), (1,), 1)
        _blocks.strided_indices((1,), (1,), 2)
        repeated_first = _blocks.strided_indices((1,), (1,), 0)
    finally:
        _blocks._STRIDED_INDICES_CACHE_MAXSIZE = original_maxsize
        _blocks._clear_strided_indices_cache_for_tests()

    assert repeated_first is not first
    assert repeated_first.tolist() == [0]


def test_get_and_add_to_strided_use_same_values_after_cache_clear():
    _blocks._clear_strided_indices_cache_for_tests()
    data = jnp.arange(12.0)

    value = _blocks.get_strided(data, (2, 3), (1, 4), 2)
    result = _blocks.add_to_strided(
        jnp.zeros_like(data),
        (2, 3),
        (1, 4),
        2,
        value,
    )

    assert value.tolist() == [[2.0, 6.0, 10.0], [3.0, 7.0, 11.0]]
    assert result.tolist() == [
        0.0,
        0.0,
        2.0,
        3.0,
        0.0,
        0.0,
        6.0,
        7.0,
        0.0,
        0.0,
        10.0,
        11.0,
    ]
    assert len(_blocks._STRIDED_INDICES_CACHE) == 1


def test_jitted_get_matches_expected_result():
    _blocks._clear_strided_indices_cache_for_tests()
    data = jnp.arange(12.0)

    @jax.jit
    def get_once(value):
        return _blocks.get_strided(value, (2, 3), (1, 4), 2)

    result = get_once(data)

    assert result.tolist() == [[2.0, 6.0, 10.0], [3.0, 7.0, 11.0]]


def test_jitted_strided_indices_with_dynamic_offset_does_not_cache_tracer():
    _blocks._clear_strided_indices_cache_for_tests()

    @jax.jit
    def build_indices(offset):
        return _blocks.strided_indices((2, 3), (1, 4), offset)

    result = build_indices(jnp.asarray(2))

    assert result.tolist() == [2, 6, 10, 3, 7, 11]
    assert len(_blocks._STRIDED_INDICES_CACHE) == 0


def test_contiguous_get_matches_flat_slice_without_populating_cache():
    _blocks._clear_strided_indices_cache_for_tests()
    data = jnp.arange(12.0)

    value = _blocks.get_strided(data, (2, 3), (3, 1), 4)

    assert value.tolist() == [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    assert len(_blocks._STRIDED_INDICES_CACHE) == 0


def test_contiguous_add_to_matches_flat_slice_without_populating_cache():
    _blocks._clear_strided_indices_cache_for_tests()
    storage = jnp.zeros((12,), dtype=jnp.float32)
    value = jnp.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=jnp.float32)

    result = _blocks.add_to_strided(storage, (2, 3), (3, 1), 4, value)

    assert result.tolist() == [0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0]
    assert len(_blocks._STRIDED_INDICES_CACHE) == 0


def test_contiguous_set_matches_flat_slice_without_populating_cache():
    _blocks._clear_strided_indices_cache_for_tests()
    storage = jnp.full((12,), -1.0, dtype=jnp.float32)
    value = jnp.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=jnp.float32)

    result = _blocks.set_strided(storage, (2, 3), (3, 1), 4, value)

    assert result.tolist() == [
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        -1.0,
        -1.0,
    ]
    assert storage.tolist() == [-1.0] * 12
    assert len(_blocks._STRIDED_INDICES_CACHE) == 0


def test_non_contiguous_set_uses_cached_indices():
    _blocks._clear_strided_indices_cache_for_tests()
    storage = jnp.full((12,), -1.0, dtype=jnp.float32)
    value = jnp.array([[2.0, 6.0, 10.0], [3.0, 7.0, 11.0]], dtype=jnp.float32)

    result = _blocks.set_strided(storage, (2, 3), (1, 4), 2, value)

    assert result.tolist() == [
        -1.0,
        -1.0,
        2.0,
        3.0,
        -1.0,
        -1.0,
        6.0,
        7.0,
        -1.0,
        -1.0,
        10.0,
        11.0,
    ]
    assert len(_blocks._STRIDED_INDICES_CACHE) == 1


def test_size_one_dimension_with_arbitrary_stride_uses_contiguous_fast_path():
    _blocks._clear_strided_indices_cache_for_tests()
    data = jnp.arange(20.0)

    value = _blocks.get_strided(data, (2, 1, 3), (3, 999, 1), 10)
    result = _blocks.add_to_strided(
        jnp.zeros((20,), dtype=jnp.float32),
        (2, 1, 3),
        (3, 999, 1),
        10,
        value,
    )

    assert value.tolist() == [[[10.0, 11.0, 12.0]], [[13.0, 14.0, 15.0]]]
    assert result.tolist() == [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        0.0, 0.0, 0.0, 0.0,
    ]
    assert len(_blocks._STRIDED_INDICES_CACHE) == 0


def test_jitted_contiguous_get_with_dynamic_offset_uses_uncached_indices():
    _blocks._clear_strided_indices_cache_for_tests()
    data = jnp.arange(12.0)

    @jax.jit
    def get_with_offset(offset):
        return _blocks.get_strided(data, (2, 3), (3, 1), offset)

    result = get_with_offset(jnp.asarray(4))

    assert result.tolist() == [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    assert len(_blocks._STRIDED_INDICES_CACHE) == 0


def test_jitted_contiguous_add_to_with_dynamic_offset_uses_uncached_indices():
    _blocks._clear_strided_indices_cache_for_tests()
    storage = jnp.zeros((12,), dtype=jnp.float32)
    value = jnp.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=jnp.float32)

    @jax.jit
    def add_with_offset(offset):
        return _blocks.add_to_strided(storage, (2, 3), (3, 1), offset, value)

    result = add_with_offset(jnp.asarray(4))

    assert result.tolist() == [0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0]
    assert len(_blocks._STRIDED_INDICES_CACHE) == 0


def test_jitted_contiguous_set_with_dynamic_offset_uses_uncached_indices():
    _blocks._clear_strided_indices_cache_for_tests()
    storage = jnp.full((12,), -1.0, dtype=jnp.float32)
    value = jnp.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=jnp.float32)

    @jax.jit
    def set_with_offset(offset):
        return _blocks.set_strided(storage, (2, 3), (3, 1), offset, value)

    result = set_with_offset(jnp.asarray(4))

    assert result.tolist() == [
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        -1.0,
        -1.0,
    ]
    assert len(_blocks._STRIDED_INDICES_CACHE) == 0


def test_contiguous_scale_matches_flat_slice_without_populating_cache():
    _blocks._clear_strided_indices_cache_for_tests()
    storage = jnp.arange(10.0)
    factor = jnp.asarray(-2, dtype=storage.dtype)

    result = _blocks.scale_strided(storage, (2, 3), (3, 1), 2, factor)

    assert result.tolist() == [
        0.0,
        1.0,
        -4.0,
        -6.0,
        -8.0,
        -10.0,
        -12.0,
        -14.0,
        8.0,
        9.0,
    ]
    assert storage.tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
    ]
    assert len(_blocks._STRIDED_INDICES_CACHE) == 0


def test_non_contiguous_scale_uses_cached_indices():
    _blocks._clear_strided_indices_cache_for_tests()
    storage = jnp.arange(12.0)
    factor = jnp.asarray(3, dtype=storage.dtype)

    result = _blocks.scale_strided(storage, (2, 2), (1, 4), 1, factor)

    assert result.tolist() == [
        0.0,
        3.0,
        6.0,
        3.0,
        4.0,
        15.0,
        18.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
    ]
    assert len(_blocks._STRIDED_INDICES_CACHE) == 1


def test_jitted_scale_with_dynamic_offset_does_not_cache_tracer():
    _blocks._clear_strided_indices_cache_for_tests()
    storage = jnp.arange(10.0)
    factor = jnp.asarray(-1, dtype=storage.dtype)

    @jax.jit
    def scale_with_offset(offset):
        return _blocks.scale_strided(storage, (2, 3), (3, 1), offset, factor)

    result = scale_with_offset(jnp.asarray(2))

    assert result.tolist() == [
        0.0,
        1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
        -7.0,
        8.0,
        9.0,
    ]
    assert len(_blocks._STRIDED_INDICES_CACHE) == 0
