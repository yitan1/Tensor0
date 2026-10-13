"""Homogeneous pytree batches have leading batch axes and trailing storage."""

import jax
import jax.numpy as jnp
import pytest

import tensor0 as t0
from tests.cases import assert_allclose


def _tensor(sector=t0.Trivial, dtype=jnp.float32):
    sectors = {(): 2} if sector == t0.Trivial else {0: 2, 1: 1}
    p = t0.space(sector, sectors)
    s = t0.hom((p,), (p,))
    return t0.from_blocks(s, {
        k: jnp.eye(n, dtype=dtype) * (i + 1)
        for i, (k, n) in enumerate(p.sectors)
    })


def _stack(*values):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *values)


def _item(batch, i):
    return jax.tree.map(lambda x: x[i], batch)


@pytest.mark.parametrize('sector', [t0.Trivial, t0.U1Irrep, t0.SU2Irrep, t0.FermionParity])
def test_direct_outputs_stack_remap_and_nested_batches(sector):
    a = _tensor(sector)
    batch = _stack(a, 2 * a)
    direct = jax.vmap(lambda data: t0.TensorMap(a.space, data) @ a)(batch.storage.data)
    mapped = jax.jit(jax.vmap(lambda x: x @ a))(batch)
    assert mapped.batch_shape == (2,)
    assert mapped.shape == a.shape
    assert mapped.space == a.space
    assert_allclose(direct.storage.data, mapped.storage.data)
    assert_allclose(_item(mapped, 1).to_dense(), (2 * a @ a).to_dense())
    nested = _stack(batch, batch)
    result = jax.jit(jax.vmap(jax.vmap(lambda x: x @ a)))(nested)
    assert result.batch_shape == (2, 2)
    assert_allclose(result.storage.data, jnp.stack([mapped.storage.data] * 2))
    with pytest.raises(ValueError, match='unbatched storage'):
        jax.vmap(lambda x: x.norm())(nested)


@pytest.mark.parametrize('sector', [t0.Trivial, t0.U1Irrep, t0.SU2Irrep])
def test_structured_decomposition_outputs(sector):
    a = _tensor(sector)
    batch = _stack(a, 2 * a)

    def local(x):
        u, s, v = t0.svd_compact(x)
        return {'svd': (u, s, v), 'values': t0.eigh_vals(x), 'eigh': t0.eigh_full(x)}

    output = jax.jit(jax.vmap(local))(batch)
    assert output['svd'][1].batch_shape == (2,)
    assert output['values'].batch_shape == (2,)
    reconstruction = jax.vmap(lambda u, s, v: u @ s @ v)(*output['svd'])
    assert_allclose(reconstruction.storage.data, batch.storage.data)
    assert_allclose(_item(output['values'], 1).to_diagonal().to_dense(),
                    _item(output['eigh'][0], 1).to_dense())
    stacked = _stack(local(a), local(2 * a))
    for actual, expected in zip(jax.tree.leaves(output), jax.tree.leaves(stacked), strict=True):
        assert_allclose(actual, expected)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.complex64])
def test_jit_ad_and_su2_batch_riesz_gradient(dtype):
    a = _tensor(t0.SU2Irrep, dtype)
    if dtype == jnp.complex64:
        a = a * (1 + 2j)
    batch = _stack(_stack(a, 2 * a), _stack(3 * a, 4 * a))
    local = lambda x: jnp.real(x.inner(x))
    loss = lambda x: jnp.sum(jax.vmap(jax.vmap(local))(x))
    coordinate = jax.jit(jax.grad(loss))(batch)
    gradient = jax.jit(t0.grad(loss))(batch)
    (_, aux), vg = jax.jit(t0.value_and_grad(lambda x: (loss(x), x), has_aux=True))(batch)
    assert_allclose(aux.storage.data, batch.storage.data)
    assert_allclose(gradient.storage.data, 2 * batch.storage.data)
    assert_allclose(vg.storage.data, gradient.storage.data)
    expected = jnp.stack([jax.grad(local)(_item(_item(batch, i), j)).storage.data
                          for i in range(2) for j in range(2)]).reshape(batch.storage.data.shape)
    assert_allclose(coordinate.storage.data, expected)
    assert not bool(jnp.allclose(coordinate.storage.data, gradient.storage.data))
    per_item = jax.jit(jax.vmap(jax.vmap(t0.grad(local))))(batch)
    assert_allclose(per_item.storage.data, gradient.storage.data)
    primal, tangent = jax.jit(lambda x: jax.jvp(jax.vmap(jax.vmap(lambda y: y @ a)), (x,), (x,)))(batch)
    assert_allclose(primal.storage.data, tangent.storage.data)


@pytest.mark.parametrize('kind', ['tensor', 'diagonal', 'vector'])
def test_pytree_validation_placeholders_and_constructor_compatibility(kind):
    a = _tensor()
    value = a if kind == 'tensor' else (t0.svd_compact(a)[1] if kind == 'diagonal' else t0.svd_vals(a))
    leaves, tree = jax.tree.flatten(value)
    for placeholder in (object(), None, 0):
        rebuilt = jax.tree.unflatten(tree, [placeholder])
        assert rebuilt.storage.data is placeholder
    batch = _stack(value, value)
    assert batch.batch_shape == (2,)
    with pytest.raises(ValueError, match='must be 1D'):
        if isinstance(value, t0.TensorMap):
            t0.TensorMap(value.space, batch.storage.data)
        elif isinstance(value, t0.DiagonalTensorMap):
            t0.DiagonalTensorMap(value.index_space, batch.storage.data)
        else:
            t0.SectorVector(value._visible_space(), batch.storage.data)
    for wrong in (jnp.zeros(()), jnp.zeros((2, leaves[0].shape[-1] + 1))):
        with pytest.raises(ValueError, match='trailing dimension'):
            jax.tree.unflatten(tree, [wrong])
    assert jax.vmap(lambda x: x)(batch).batch_shape == (2,)
    assert jax.eval_shape(jax.vmap(lambda x: x), batch).batch_shape == (2,)


def test_incompatible_metadata_cannot_stack():
    a = _tensor(t0.U1Irrep)
    p = t0.space(t0.U1Irrep, {0: 2, 2: 1})
    b = t0.ones(t0.hom((p,), (p,)))
    with pytest.raises(ValueError):
        _stack(a, b)


# Include fast paths that used to return a constant or identity without reading storage.
_TENSOR_OPERATIONS = [
    lambda x, a: x.copy(), lambda x, a: x.astype(jnp.float32), lambda x, a: x.similar(),
    lambda x, a: x.block(()), lambda x, a: x.blocks(), lambda x, a: x.subblocks(),
    lambda x, a: x.subblock(((), ())), lambda x, a: x.to_dense(),
    lambda x, a: t0.zero_like(x), lambda x, a: t0.scale(x, 1),
    lambda x, a: t0.add(a, x), lambda x, a: t0.inner(a, x),
    lambda x, a: t0.norm(x), lambda x, a: t0.norm(x, jnp.inf),
    lambda x, a: t0.adjoint(x), lambda x, a: t0.real(x), lambda x, a: t0.imag(x),
    lambda x, a: t0.complex(x), lambda x, a: t0.tr(x), lambda x, a: t0.diag(x),
    lambda x, a: t0.is_diagonal(x), lambda x, a: t0.inverse(x),
    lambda x, a: t0.pseudoinverse(x), lambda x, a: a @ x, lambda x, a: x @ a,
    lambda x, a: t0.equal(x, a), lambda x, a: t0.allclose(x, a, rtol=1e-5, atol=1e-5),
    lambda x, a: t0.permute(x, ((0,), (1,))),
    lambda x, a: t0.braid(x, ((0,), (1,)), (0, 1)),
    lambda x, a: t0.transpose(x, ((0,), (1,))), lambda x, a: t0.repartition(x, 1),
    lambda x, a: t0.twist(x, ()), lambda x, a: t0.flip(x, ()),
    lambda x, a: t0.insertleftunit(x), lambda x, a: t0.insertrightunit(x),
    lambda x, a: t0.tensor_product(x, a),
    lambda x, a: t0.tensortrace(x, axes=((), ()), output=((0,), (1,))),
    lambda x, a: t0.contract(t0.idx(x, 'i,j'), output='i;j'),
    lambda x, a: t0.ncon([x], [(-1, -2)]),
    lambda x, a: t0.left_solve(a, x), lambda x, a: t0.right_solve(x, a),
    lambda x, a: t0.is_isometric(x), lambda x, a: t0.is_unitary(x),
    lambda x, a: t0.is_positive_definite(x), lambda x, a: t0.is_hermitian(x),
    lambda x, a: t0.svd_vals(x), lambda x, a: t0.qr_compact(x),
    lambda x, a: t0.eigh_full(x), lambda x, a: t0.rank(x), lambda x, a: t0.cond(x),
]


@pytest.mark.parametrize('operation', _TENSOR_OPERATIONS)
def test_tensor_numeric_boundaries_reject_batches(operation):
    a = _tensor()
    with pytest.raises(ValueError, match='unbatched storage'):
        operation(_stack(a, a), a)


@pytest.mark.parametrize('method,args', [
    ('copy', ()), ('astype', (jnp.float32,)), ('diag', ()), ('block', ((),)),
    ('blocks', ()), ('to_tensor_map', ()), ('to_dense', ()), ('zero_like', ()),
    ('scale', (1,)), ('norm', ()), ('adjoint', ()), ('real', ()), ('imag', ()),
    ('complex', ()), ('inverse', ()), ('pseudoinverse', ()), ('is_diagonal', ()),
])
def test_diagonal_numeric_boundaries_reject_batches(method, args):
    a = t0.svd_compact(_tensor())[1]
    with pytest.raises(ValueError, match='unbatched storage'):
        getattr(_stack(a, a), method)(*args)
    for operation in (lambda: a @ _stack(a, a), lambda: a.add(_stack(a, a))):
        with pytest.raises(ValueError, match='unbatched storage'):
            operation()


@pytest.mark.parametrize('method,args', [
    ('values', ()), ('pairs', ()), ('get', ((999,),)), ('block', ((),)),
    ('blocks', ()), ('to_diagonal', ()), ('copy', ()), ('similar', ()),
])
def test_sectorvector_numeric_boundaries_reject_batches(method, args):
    a = t0.svd_vals(_tensor())
    with pytest.raises(ValueError, match='unbatched storage'):
        getattr(_stack(a, a), method)(*args)


def test_scalar_unit_empty_and_mismatched_comparison_fast_paths():
    scalar = t0.TensorMap(t0.hom((), (), sector_type=t0.Trivial), jnp.ones((1,)))
    with pytest.raises(ValueError, match='unbatched storage'):
        t0.scalar(_stack(scalar, scalar))
    unit = t0.insertleftunit(_tensor())
    with pytest.raises(ValueError, match='unbatched storage'):
        t0.removeunit(_stack(unit, unit), 2)
    a = _tensor()
    p = t0.space(t0.Trivial, {(): 1})
    other = t0.ones(t0.hom((p,), (p,)))
    for compare in (lambda: t0.equal(_stack(a, a), other),
                    lambda: t0.allclose(other, _stack(a, a), rtol=0, atol=0)):
        with pytest.raises(ValueError, match='unbatched storage'):
            compare()
    p = t0.space(t0.U1Irrep, {})
    vector = t0.SectorVector(p, jnp.empty((0,)))
    for method in ('values', 'pairs', 'blocks'):
        with pytest.raises(ValueError, match='unbatched storage'):
            getattr(_stack(vector, vector), method)()
    empty = t0.TensorMap(t0.hom((p,), (p,)), jnp.empty((0,)))
    assert _stack(empty, empty).batch_shape == (2,)
    assert_allclose(jax.vmap(lambda x: x.norm())(_stack(empty, empty)), jnp.zeros((2,)))


def test_jacobian_requires_an_explicit_array_boundary():
    a = _tensor()
    # Output/storage and input/coordinate dimensions differ; this is not a batch leaf.
    with pytest.raises(ValueError, match='trailing dimension'):
        jax.jacfwd(lambda v: t0.TensorMap(a.space, v[0] * a.storage.data))(jnp.ones((3,)))
    jacobian = jax.jacfwd(lambda v: (v[0] * a).storage.data)(jnp.ones((3,)))
    assert jacobian.shape == (a.dim, 3)


@pytest.mark.parametrize('size', [0, 1, 3])
def test_small_and_empty_batch_axes(size):
    a = _tensor()
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (size, *x.shape)), a)
    assert batch.batch_shape == (size,)
    actual = jax.jit(jax.vmap(lambda x: x @ a))(batch)
    assert actual.storage.data.shape == (size, a.dim)
    with pytest.raises(ValueError, match='unbatched storage'):
        batch.to_dense()


def test_nonzero_leading_axis_mapping_and_storage_last_validation():
    a = _tensor()
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (2, 3, *x.shape)), a)
    result = jax.jit(jax.vmap(jax.vmap(lambda x: 2 * x), in_axes=1, out_axes=1))(batch)
    assert result.batch_shape == (2, 3)
    assert_allclose(result.storage.data, 2 * batch.storage.data)
    single_axis_batch = _item(batch, 0)
    with pytest.raises(ValueError, match='trailing dimension'):
        jax.vmap(lambda x: x, out_axes=-1)(single_axis_batch)


@pytest.mark.parametrize('sector', [t0.SU2Irrep, t0.FermionParity])
def test_symmetric_stride_transforms_trace_and_ad_with_direct_outputs(sector):
    a = _tensor(sector)
    product = t0.tensor_product(a, a)
    batch = _stack(product, 2 * product)

    def local(x):
        swapped = t0.permute(x, ((1, 0), (3, 2)))
        traced = t0.tensortrace(swapped, axes=((1,), (3,)), output=((0,), (2,)))
        return {'swapped': swapped, 'traced': traced}

    result = jax.jit(jax.vmap(local))(batch)
    expected = _stack(local(product), local(2 * product))
    for actual, reference in zip(jax.tree.leaves(result), jax.tree.leaves(expected), strict=True):
        assert_allclose(actual, reference)
    objective = lambda x: jnp.real(local(x)['traced'].inner(local(x)['traced']))
    gradient = jax.jit(jax.grad(lambda xs: jnp.sum(jax.vmap(objective)(xs))))(batch)
    expected_gradient = _stack(jax.grad(objective)(product), jax.grad(objective)(2 * product))
    assert_allclose(gradient.storage.data, expected_gradient.storage.data)


@pytest.mark.parametrize('sector', [t0.Trivial, t0.SU2Irrep])
def test_vjp_structured_batched_tensor_outputs(sector):
    a = _tensor(sector)
    batch = _stack(a, 2 * a)
    local = lambda x: {'tensor': 2 * x, 'nested': (x @ a,)}

    def mapped_vjp(xs):
        output, pullback = jax.vjp(jax.vmap(local), xs)
        return output, pullback(output)[0]

    output, cotangent = jax.jit(mapped_vjp)(batch)
    expected = []
    for value in (a, 2 * a):
        single_output, pullback = jax.vjp(local, value)
        expected.append(pullback(single_output)[0])
    assert output['tensor'].batch_shape == (2,)
    assert output['nested'][0].batch_shape == (2,)
    assert_allclose(cotangent.storage.data, _stack(*expected).storage.data)
