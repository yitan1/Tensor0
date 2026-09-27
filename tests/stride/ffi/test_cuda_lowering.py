"""CUDA lowering prepares one shared host/device descriptor before emitting IR."""

from collections import namedtuple

import jax
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride._ffi import _calls

from tests.stride.support.descriptors import OPERATIONS, address_words, reduction_words


class Context(namedtuple("Context", "avals_in avals_out")):
    def replace(self, **kwargs):
        return self._replace(**kwargs)


def lowering_case(operation):
    dtype = np.dtype("float32")
    storage = jax.core.ShapedArray((3, 6), dtype)
    scalar = jax.core.ShapedArray((), np.dtype("int32"))
    batch = jax.core.ShapedArray((3,), dtype)
    avals = {
        "copy": (storage,), "update": (storage, storage, scalar, batch),
        "dot": (storage, storage), "accumulation": (storage, scalar, batch),
        "reduction": (storage, scalar, batch),
    }[operation]
    output = jax.core.ShapedArray((3,), dtype) if operation == "dot" else storage
    context = Context(avals, (output,))
    operands = tuple(object() for _ in avals)
    kwargs = {"layout": (1, 6, 6, 1, 1, 0, 0, 1, 1, 1)}
    if operation == "dot":
        kwargs.update(conjugate_left=True, dtype=dtype)
    elif operation != "update":
        kwargs.update(output_size=6, dtype=dtype)
        if operation in ("accumulation", "reduction"):
            kwargs["coefficient_records"] = (2, 0)
    return context, operands, kwargs


def capture_lowering(monkeypatch, operation, *, prepared=None, error=None, actual_bridge=False):
    events, captured = [], {}
    metadata = object()
    results = [object(), object()]

    def target(*dtypes):
        events.append("target")
        captured["dtypes"] = dtypes
        return "test_cuda_target"

    original = getattr(_native, "_stride_prepare_layout", None)

    def prepare(name, words):
        events.append("prepare")
        captured["prepare"] = (name, words)
        if error is not None:
            raise error
        return original(name, words) if actual_bridge else prepared

    def pack(name, words):
        events.append("pack")
        captured["pack"] = (name, words)
        return _native_pack(name, words) if actual_bridge else (1, 3, 5, 7)

    _native_pack = _native._stride_pack_owner_fiber
    _native_capacity = _native._stride_dot_scratch_capacity
    _native_sum_capacity = _native._stride_sum_scratch_capacity

    def sum_capacity(name, words):
        events.append("capacity")
        captured["capacity"] = (name, words)
        return _native_sum_capacity(name, words) if actual_bridge else 2

    def capacity(words):
        events.append("capacity")
        captured["capacity"] = words
        return _native_capacity(words) if actual_bridge else 2

    def constant(words):
        events.append("constant")
        captured["constant"] = words
        return metadata

    def ffi_lowering(name, **options):
        events.append("ffi")
        captured["target"] = name
        captured["options"] = options

        def lower(context, *operands, **attributes):
            events.append("lower")
            captured.update(context=context, operands=operands, attributes=attributes)
            return results

        return lower

    monkeypatch.setattr(_calls, f"cuda_{operation}_target", target)
    monkeypatch.setattr(_native, "_stride_prepare_layout", prepare, raising=False)
    monkeypatch.setattr(_native, "_stride_pack_owner_fiber", pack, raising=False)
    monkeypatch.setattr(_native, "_stride_dot_scratch_capacity", capacity, raising=False)
    monkeypatch.setattr(_native, "_stride_sum_scratch_capacity", sum_capacity, raising=False)
    monkeypatch.setattr(_calls.mlir, "ir_constant", constant)
    monkeypatch.setattr(jax.ffi, "ffi_lowering", ffi_lowering)
    return events, captured, metadata, results


@pytest.mark.parametrize("operation", OPERATIONS)
def test_cuda_lowering_uses_prepared_host_and_device_words(monkeypatch, operation):
    context, operands, kwargs = lowering_case(operation)
    prepared = (1, 2048, 2048, 1, 2, 0, 0, 1025, 1, 1, 0, 1, 0)
    events, captured, metadata, results = capture_lowering(monkeypatch, operation, prepared=prepared)
    with jax.enable_x64(False):
        actual = getattr(_calls, f"_cuda_{operation}_lowering")(context, *operands, **kwargs)
    packed = True
    assert events == (["target", "prepare", "pack", "capacity", "constant", "ffi", "lower"]
                      if operation == "dot" else
                      ["target", "prepare", "pack", "constant", "capacity", "ffi", "lower"]
                      if operation in ("accumulation", "reduction") else
                      ["target", "prepare", "pack", "constant", "ffi", "lower"])
    assert captured["prepare"] == (operation, kwargs["layout"])
    assert captured["target"] == "test_cuda_target"
    words = captured["constant"]
    assert words.dtype == np.dtype("int64")
    np.testing.assert_array_equal(captured["attributes"]["layout"], prepared)
    if packed:
        assert captured["pack"][0] == operation
        np.testing.assert_array_equal(captured["pack"][1], prepared)
        np.testing.assert_array_equal(words, (1, 3, 5, 7))
    else:
        assert words is captured["attributes"]["layout"]
        np.testing.assert_array_equal(words, prepared)
    metadata_index = 1 if operation in ("accumulation", "reduction") else len(operands)
    expected_operands = (*operands[:metadata_index], metadata, *operands[metadata_index:])
    assert captured["operands"] == expected_operands
    avals = captured["context"].avals_in
    assert avals[metadata_index].shape == ((4,) if packed else (len(prepared),))
    assert avals[metadata_index].dtype == np.dtype("int64")
    assert (*avals[:metadata_index], *avals[metadata_index + 1:]) == context.avals_in
    options = captured["options"]
    assert options["operand_layouts"] == tuple(tuple(reversed(range(aval.ndim))) for aval in avals)
    if operation == "update":
        assert options["operand_output_aliases"] == {1: 0}
    else:
        assert "operand_output_aliases" not in options
    if operation in ("accumulation", "reduction"):
        indices = captured["attributes"]["coefficient_records"]
        assert indices.dtype == np.dtype("int64")
        np.testing.assert_array_equal(indices, (2, 0))
    if operation == "dot":
        assert actual == results[:1]
        assert captured["capacity"] == prepared
        assert captured["context"].avals_out[1].shape == (3, 2)
        assert options["result_layouts"] == ((0,), (1, 0))
        assert captured["attributes"]["conjugate_left"] == np.int64(1)
    elif operation in ("accumulation", "reduction"):
        assert actual == results[:1]
        assert captured["capacity"] == (operation, list(prepared))
        assert captured["context"].avals_out[1].shape == (3, 2)
        assert options["result_layouts"] == ((1, 0), (1, 0))
    else:
        assert actual is results
        assert captured["context"].avals_out == context.avals_out
        assert options["result_layouts"] == ((1, 0),)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_cuda_prepare_error_precedes_ir_and_ffi(monkeypatch, operation):
    context, operands, kwargs = lowering_case(operation)
    events, _, _, _ = capture_lowering(monkeypatch, operation, error=ValueError("invalid descriptor"))
    with pytest.raises(ValueError, match="invalid descriptor"):
        getattr(_calls, f"_cuda_{operation}_lowering")(context, *operands, **kwargs)
    assert events == ["target", "prepare"]


@pytest.mark.parametrize("operation", OPERATIONS)
def test_cuda_target_failure_precedes_preparation(monkeypatch, operation):
    context, operands, kwargs = lowering_case(operation)
    events, _, _, _ = capture_lowering(monkeypatch, operation)

    def unavailable(*args):
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(_calls, f"cuda_{operation}_target", unavailable)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        getattr(_calls, f"_cuda_{operation}_lowering")(context, *operands, **kwargs)
    assert events == []


@pytest.mark.parametrize("unsigned_extent", [False, True])
def test_reduction_lowering_real_bridge_canonicalizes_host_and_device(monkeypatch, unsigned_extent):
    context, operands, kwargs = lowering_case("reduction")
    if unsigned_extent:
        kwargs["layout"] = (1, 1, 1, 1, 1, 0, 0, -1, 0, 1, -(1 << 63), 1)
        expected = (1, 1, 1, 1, 1, 0, 0, -1, 0, 1, 0, 1)
    else:
        kwargs["layout"] = reduction_words()
        expected = (1, 6, 2, 1, 2, 2, 0, 3, 2, -1, 3, 1, 2, 0, 1, 1, 0)
    native_pack = _native._stride_pack_owner_fiber
    def _native_pack_reference(words):
        return native_pack("reduction", words)
    events, captured, metadata, _ = capture_lowering(monkeypatch, "reduction", actual_bridge=True)
    with jax.enable_x64(False):
        _calls._cuda_reduction_lowering(context, *operands, **kwargs)
    assert events == ["target", "prepare", "pack", "constant", "capacity", "ffi", "lower"]
    words = captured["constant"]
    np.testing.assert_array_equal(captured["attributes"]["layout"], expected)
    np.testing.assert_array_equal(words, _native_pack_reference(expected))
    assert captured["operands"][1] is metadata
    assert captured["context"].avals_in[1].shape == (len(words),)
    assert captured["context"].avals_in[1].dtype == np.dtype("int64")


def test_accumulation_lowering_real_bridge_packs_device_view(monkeypatch):
    context, operands, kwargs = lowering_case("accumulation")
    kwargs["layout"] = (1, 6, 8, 1, 2, 0, 1, 2, 3, 3, 1, 0, 1)
    native_pack = _native._stride_pack_owner_fiber
    events, captured, metadata, _ = capture_lowering(monkeypatch, "accumulation", actual_bridge=True)
    with jax.enable_x64(False):
        _calls._cuda_accumulation_lowering(context, *operands, **kwargs)
    assert events == ["target", "prepare", "pack", "constant", "capacity", "ffi", "lower"]
    semantic = captured["attributes"]["layout"]
    np.testing.assert_array_equal(captured["constant"], native_pack("accumulation", semantic.tolist()))
    assert captured["constant"] is not semantic
    assert captured["operands"][1] is metadata
    assert captured["context"].avals_in[1].shape == captured["constant"].shape


@pytest.mark.parametrize("operation", ("copy", "update", "dot"))
def test_map_and_dot_lowering_real_bridge_pack_from_host_layout(monkeypatch, operation):
    context, operands, kwargs = lowering_case(operation)
    kwargs["layout"] = address_words(destination_strides=(3, 1))
    native_pack = _native._stride_pack_owner_fiber
    events, captured, metadata, results = capture_lowering(
        monkeypatch, operation, actual_bridge=True)
    with jax.enable_x64(False):
        actual = getattr(_calls, f"_cuda_{operation}_lowering")(
            context, *operands, **kwargs)
    assert events == (["target", "prepare", "pack", "capacity", "constant", "ffi", "lower"] if operation == "dot"
                      else ["target", "prepare", "pack", "constant", "ffi", "lower"])
    semantic = captured["attributes"]["layout"]
    assert tuple(semantic) == _native._stride_prepare_layout(operation, kwargs["layout"])
    np.testing.assert_array_equal(captured["constant"], native_pack(operation, semantic.tolist()))
    assert captured["constant"] is not semantic
    assert captured["context"].avals_in[-1].shape == captured["constant"].shape
    assert captured["context"].avals_in[-1].dtype == np.dtype("int64")
    assert captured["operands"][-1] is metadata
    assert actual == (results[:1] if operation == "dot" else results)
    if operation == "dot":
        capacity = _native._stride_dot_scratch_capacity(semantic.tolist())
        assert captured["context"].avals_out[1].shape == (3, capacity)
        assert captured["options"]["result_layouts"] == ((0,), (1, 0))
