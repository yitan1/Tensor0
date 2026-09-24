"""Bounded public TensorMap CPU/CUDA acceptance, not a universal support claim."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import (
    ComplexSpace, FermionParity, SU2Irrep, TensorMap, U1Irrep, _native,
    adjoint, add, flip, hom, inner, norm, permute, scale, space,
    tensorcontract, tensortrace, transpose, twist,
)
from tensor0.structure import get_degeneracystructure


DTYPES = ["float32", "complex64", "float64", "complex128"]


@pytest.fixture(autouse=True)
def highest_matmul_precision():
    # Test storage-precision accuracy, not GPU default reduced-precision matmul.
    # This affects JAX matrix products, not native stride arithmetic.
    with jax.default_matmul_precision("highest"):
        yield


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        if not getattr(_native, "_stride_cuda_available", lambda: False)():
            pytest.skip("native CUDA extension is unavailable")
        try:
            return jax.devices("cuda")[0]
        except RuntimeError:
            pytest.skip("CUDA device is unavailable")
    return jax.devices("cpu")[0]


def _factor(family):
    if family == "trivial":
        return ComplexSpace(2)
    if family == "u1":
        return space(U1Irrep, {0: 1, 1: 1})
    return space(SU2Irrep, {1: 1})


def _data(target, dtype):
    size = get_degeneracystructure(target).total_dim
    values = (np.arange(size) % 11 + 1) / 10
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        values = values + 1j * (np.arange(size) % 5 - 2) / 7
    return values.astype(dtype)


def _check(operation, arrays, device, dtype, *, native=None):
    """Run both placements, asserting dtype before compilation (no x64 truncation)."""
    cpu = jax.devices("cpu")[0]
    args = tuple(jax.device_put(a, device) for a in arrays)
    assert args[0].dtype == np.dtype(dtype)
    compiled = jax.jit(operation)
    with jax.default_device(cpu):
        expected = compiled(*(jax.device_put(a, cpu) for a in arrays))
    with jax.default_device(device):
        actual = compiled(*args)
    tolerance = 3e-5 if dtype in ("float32", "complex64") else 2e-12
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert value.devices() == {device}
        assert value.dtype == reference.dtype
        np.testing.assert_allclose(value, reference, rtol=tolerance, atol=tolerance)
    text = str(compiled.lower(*args).compiler_ir("stablehlo"))
    if device.platform == "gpu":
        assert "_cpu_v1" not in text
        if native is not None:
            assert f"_{native}_" in text and "_cuda_v1" in text
    return actual


@pytest.mark.parametrize("family", ["trivial", "u1", "su2"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_public_forward_matrix(device, family, dtype):
    factor = _factor(family)
    target = hom((factor, factor), (factor, factor))
    values = _data(target, dtype)

    def operation(data):
        tensor = TensorMap(target, data)
        return (
            permute(tensor, ((1, 2), (0, 3))), transpose(tensor), adjoint(tensor),
            add(scale(tensor, 2), tensor, alpha=1, beta=-1),
            inner(tensor, tensor), norm(tensor), tensor @ tensor,
            tensorcontract(tensor, tensor, axes=((2, 3), (0, 1)),
                           output=(((0, 0), (0, 1)), ((1, 2), (1, 3)))),
            tensortrace(tensor, axes=((1,), (3,)), output=((0,), (2,))),
        )

    with jax.enable_x64(dtype in ("float64", "complex128")):
        result = _check(operation, (values,), device, dtype)
        np.testing.assert_allclose(result[3].storage.data, values, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result[6].storage.data, result[7].storage.data,
                                   rtol=3e-5, atol=3e-5)
        if family == "trivial":
            dense = values.reshape(2, 2, 2, 2)
            np.testing.assert_allclose(result[0].storage.data,
                                       dense.transpose(1, 2, 0, 3).reshape(-1))
            np.testing.assert_allclose(result[6].storage.data,
                                       (dense.reshape(4, 4) @ dense.reshape(4, 4)).reshape(-1),
                                       rtol=3e-5, atol=3e-5)
            np.testing.assert_allclose(result[8].storage.data,
                                       np.trace(dense, axis1=1, axis2=3).reshape(-1),
                                       rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("family", ["trivial", "u1", "su2"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_transform_trace_ad_and_external_vmap(device, family, dtype):
    factor = _factor(family)
    target = hom((factor, factor), (factor, factor))
    values = _data(target, dtype)

    def transform(data):
        return permute(TensorMap(target, data), ((1, 2), (0, 3))).storage.data

    def trace(data):
        return tensortrace(TensorMap(target, data), axes=((1,), (3,)),
                           output=((0,), (2,))).storage.data

    def differentiation(data):
        tangent = jnp.ones_like(data) * (1 + (0.25j if jnp.iscomplexobj(data) else 0))
        primal, jvp = jax.jvp(transform, (data,), (tangent,))
        traced, pullback = jax.vjp(trace, data)
        cotangent = jnp.ones_like(traced) * (1 + (0.5j if jnp.iscomplexobj(data) else 0))
        reverse = pullback(cotangent)[0]
        def loss(x):
            tensor = TensorMap(target, x)
            composed = tensor @ tensor
            return jnp.real(inner(composed, composed))
        return primal, jvp, reverse, jax.grad(loss)(data)

    with jax.enable_x64(dtype in ("float64", "complex128")):
        _check(differentiation, (values,), device, dtype)
        _check(transform, (values,), device, dtype,
               native="copy" if family == "su2" else
                      "accumulation" if family == "u1" else None)
        _check(jax.vmap(transform), (np.stack((values, values * 2)),), device, dtype)
        _check(trace, (values,), device, dtype,
               native="reduction" if family != "trivial" else None)


@pytest.mark.parametrize("dtype", DTYPES)
def test_scale_coefficient_vjp_uses_public_jax_path(device, dtype):
    target = hom((_factor("u1"),), (_factor("u1"),))
    values = _data(target, dtype)

    def operation(data, coefficient):
        def apply(c):
            return scale(TensorMap(target, data), c).storage.data
        output, reverse = jax.vjp(apply, coefficient)
        return output, reverse(jnp.ones_like(output))[0]

    with jax.enable_x64(dtype in ("float64", "complex128")):
        coefficients = [np.asarray(2, dtype)]
        if dtype.startswith("complex"):
            coefficients.append(np.asarray(2, "float32" if dtype == "complex64" else "float64"))
        for coefficient in coefficients:
            output, gradient = _check(operation, (values, coefficient), device, dtype)
            np.testing.assert_allclose(output, values * 2, rtol=1e-6, atol=1e-6)
            expected = values.sum()
            if not np.iscomplexobj(coefficient):
                expected = expected.real
            np.testing.assert_allclose(gradient, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "complex128"])
def test_fermionic_signs_and_grad(device, dtype):
    odd = space(FermionParity, {1: 1})
    target = hom((odd, odd), ())
    values = np.asarray([2 + 1j if dtype.startswith("complex") else 2], dtype)

    def operation(data):
        tensor = TensorMap(target, data)
        swapped = permute(tensor, ((1, 0), ()))
        twisted = twist(tensor, 0)
        twice_forward = flip(flip(tensor, 0), 0)
        restored = flip(flip(tensor, 0), 0, inv=True)
        gradient = jax.grad(lambda x: jnp.real(jnp.sum(
            permute(TensorMap(target, x), ((1, 0), ())).storage.data)))(data)
        return (swapped.storage.data, twisted.storage.data, twice_forward.storage.data,
                restored.storage.data, gradient)

    with jax.enable_x64(dtype == "complex128"):
        swapped, twisted, twice_forward, restored, gradient = _check(operation, (values,), device, dtype)
        np.testing.assert_allclose(swapped, -values)
        np.testing.assert_allclose(twisted, -values)
        np.testing.assert_allclose(twice_forward, -values)
        np.testing.assert_allclose(restored, values)
        np.testing.assert_allclose(gradient, [-1])


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_su2_narrow_x64_structural_coefficients(device, dtype):
    half = _factor("su2")
    target = hom((half, half, half, half.dual()), ())
    values = _data(target, dtype)
    operation = jax.jit(lambda data: tensortrace(TensorMap(target, data),
        axes=((3,), (0,)), output=((1, 2), ())).storage.data)
    with jax.enable_x64(), jax.default_device(device):
        data = jax.device_put(values, device)
        assert data.dtype == np.dtype(dtype)
        if device.platform == "gpu":
            with pytest.raises(NotImplementedError, match="coefficient dtype float64"):
                operation(data)
        else:
            result = operation(data)
            assert result.dtype == data.dtype
            assert result.devices() == {device}
            assert np.all(np.isfinite(result))


@pytest.mark.parametrize("family", ["u1", "su2"])
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_public_factorizations_and_solve_forward(device, family, dtype):
    from tensor0 import eigh_full, qr_compact, svd_compact
    from tensor0.tensor.linalg import left_solve

    factor = (space(U1Irrep, {0: 2, 1: 1}) if family == "u1"
              else space(SU2Irrep, {0: 2, 1: 1}))
    target = hom((factor,), (factor,))
    values = _data(target, dtype)

    def operation(data):
        tensor = TensorMap(target, data)
        q, r = qr_compact(tensor)
        u, s, vh = svd_compact(tensor)
        positive = tensor @ adjoint(tensor)
        diagonal, vectors = eigh_full(positive)
        solved = left_solve(positive, tensor)
        return q @ r, u @ s @ vh, vectors @ diagonal @ adjoint(vectors), positive @ solved

    with jax.enable_x64():
        qr, svd, eig, solved = _check(operation, (values,), device, dtype)
        for reconstructed in (qr, svd, solved):
            np.testing.assert_allclose(reconstructed.storage.data, values, rtol=2e-11, atol=2e-11)
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            tensor = TensorMap(target, jax.device_put(values, cpu))
            expected = (tensor @ adjoint(tensor)).storage.data
        np.testing.assert_allclose(eig.storage.data, expected, rtol=2e-11, atol=2e-11)
