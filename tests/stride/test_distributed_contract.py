"""Two-device Dot and public transform contracts in isolated processes."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

from ._support import native_available


@pytest.mark.parametrize("operation,mode", [
    ("dot", "auto"), ("dot", "explicit"), ("dot", "left_storage"), ("dot", "right_storage"),
    ("transform", "auto"), ("transform", "explicit"),
])
def test_distributed_contract(operation, mode):
    if not native_available():
        pytest.skip("native CPU stride is unavailable")
    environment = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=2")
    script = f"from tests.stride.test_distributed_contract import run_{operation}; run_{operation}({mode!r})"
    completed = subprocess.run([sys.executable, "-c", script],
                               cwd=Path(__file__).resolve().parents[2], env=environment,
                               capture_output=True, text=True, timeout=240)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def run_dot(mode):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
    from tensor0._stride import StridedView, dotc, dotu
    from tensor0._stride._jax import accumulation_p
    from tensor0._stride._layout import AffineRecord
    from tests.stride.test_dot_contract import LAYOUTS, operands
    from tests.stride.test_dot_contract import dot_functions as functions
    from tests.stride.test_update_contract import address_pairs

    jax.config.update("jax_enable_x64", True)
    mesh = Mesh(np.asarray(jax.devices()), ("device",),
                axis_types=(AxisType.Explicit if mode == "explicit" else AxisType.Auto,))
    assert len(jax.devices()) == 2
    if mode.endswith("storage"):
        shardings = tuple(NamedSharding(mesh, P("device") if mode == f"{side}_storage" else P())
                          for side in ("left", "right"))
        arguments = tuple(jax.device_put(np.arange(8, dtype=np.float32), sharding) for sharding in shardings)
        run = lambda left, right: dotu(StridedView(left, (4,), (1,), 0), StridedView(right, (4,), (1,), 0))
        try:
            jax.jit(run, in_shardings=shardings, out_shardings=NamedSharding(mesh, P())).lower(*arguments).compile()
        except Exception as error:
            assert "cannot shard the packed storage axis" in str(error), str(error)
        else:
            raise AssertionError("Dot accepted a partitioned input storage axis")
        return

    def verify(compiled, arguments, expected, shardings, targets=(), global_sum=False):
        actual = compiled(*arguments)
        for value, reference, sharding in zip(jax.tree.leaves(actual), jax.tree.leaves(expected),
                                            jax.tree.leaves(shardings), strict=True):
            assert value.dtype == reference.dtype and value.shape == reference.shape
            np.testing.assert_allclose(value, reference, rtol=3e-6, atol=3e-6)
            assert value.sharding.is_equivalent_to(sharding, value.ndim)
            for shard in value.addressable_shards:
                np.testing.assert_allclose(shard.data, np.asarray(reference)[shard.index], rtol=3e-6, atol=3e-6)
        hlo = compiled.as_text().lower()
        for name in ("all-gather", "all-to-all", "collective-permute"):
            assert name not in hlo, hlo
        assert ("all-reduce" in hlo) == global_sum, hlo
        for target in targets:
            assert target in hlo

    for batch_shape, batch_spec in (((4,), ("device",)), ((2, 4), (None, "device"))):
        storage_sharding = NamedSharding(mesh, P(*batch_spec, None))
        result_sharding = NamedSharding(mesh, P(*batch_spec))
        for left_dtype, right_dtype, dtype in (("float32", "complex64", "complex64"),
            ("complex128", "float64", "float32"), ("float64", "float32", "float64"),
            ("complex64", "complex128", "complex128")):
            reference_arguments = operands(left_dtype, right_dtype, batch_shape)
            arguments = tuple(jax.device_put(value, storage_sharding) for value in reference_arguments)
            for operation in (dotu, dotc):
                for layout in LAYOUTS[:2]:
                    run, reference = functions(operation, layout, dtype)
                    compiled = jax.jit(run, in_shardings=(storage_sharding, storage_sharding),
                                       out_shardings=result_sharding).lower(*arguments).compile()
                    verify(compiled, arguments, reference(*reference_arguments), result_sharding,
                           ("tensor0_stride_dot_",))
                    inferred = jax.jit(run).lower(*arguments).compile()
                    verify(inferred, arguments, reference(*reference_arguments), result_sharding)
                    forward = jax.jit(lambda left, right: jax.jvp(run, (left, right), (left, right)),
                        in_shardings=(storage_sharding, storage_sharding), out_shardings=(result_sharding, result_sharding)
                    ).lower(*arguments).compile()
                    verify(forward, arguments, jax.jvp(reference, reference_arguments, reference_arguments),
                           (result_sharding, result_sharding))
                    host_cotangent = jnp.full(batch_shape, 1 + 2j if dtype.startswith("complex") else 2, dtype)
                    cotangent = jax.device_put(host_cotangent, result_sharding)
                    reverse = jax.jit(lambda left, right, value: jax.vjp(run, left, right)[1](value),
                        in_shardings=(storage_sharding, storage_sharding, result_sharding),
                        out_shardings=(storage_sharding, storage_sharding)
                    ).lower(*arguments, cotangent).compile()
                    verify(reverse, (*arguments, cotangent), jax.vjp(reference, *reference_arguments)[1](host_cotangent),
                           (storage_sharding, storage_sharding), ("tensor0_stride_accumulation_",))

    for batch_shape, batch_spec, layout in (((), (), LAYOUTS[3]), ((4,), (None,), LAYOUTS[0]),
                                           ((0,), ("device",), LAYOUTS[0]), ((4,), ("device",), LAYOUTS[2])):
        storage_sharding, output_sharding = NamedSharding(mesh, P(*batch_spec, None)), NamedSharding(mesh, P(*batch_spec))
        reference_arguments = operands("float32", "float32", batch_shape)
        arguments = tuple(jax.device_put(value, storage_sharding) for value in reference_arguments)
        run, reference = functions(dotu, layout, "float32")
        compiled = jax.jit(run, in_shardings=(storage_sharding, storage_sharding),
                           out_shardings=output_sharding).lower(*arguments).compile()
        verify(compiled, arguments, reference(*reference_arguments), output_sharding)

    storage_sharding = NamedSharding(mesh, P(None, "device", None))
    result_sharding = NamedSharding(mesh, P(None, "device"))
    reference_arguments = operands("float32", "complex128", (2, 4))
    arguments = tuple(jax.device_put(value, storage_sharding) for value in reference_arguments)
    for operation in (dotu, dotc):
        run, reference = functions(operation, LAYOUTS[1], "complex128")
        def objective(function, left, right):
            output = function(left, right)
            return jnp.real(jnp.sum(output * jnp.conj(output)))
        gradient = jax.grad(lambda left, right: objective(run, left, right), argnums=(0, 1))
        expected_gradient = jax.grad(lambda left, right: objective(reference, left, right), argnums=(0, 1))
        higher = jax.jit(lambda left, right: jax.jvp(gradient, (left, right), (left, right))[1],
            in_shardings=(storage_sharding, storage_sharding), out_shardings=(storage_sharding, storage_sharding)
        ).lower(*arguments).compile()
        verify(higher, arguments, jax.jvp(expected_gradient, reference_arguments, reference_arguments)[1],
               (storage_sharding, storage_sharding))
        mapped = jax.jit(jax.vmap(run), in_shardings=(storage_sharding, storage_sharding),
                         out_shardings=result_sharding).lower(*arguments).compile()
        verify(mapped, arguments, jax.vmap(reference)(*reference_arguments), result_sharding)

    records = (AffineRecord((2, 2), (1, 1), 0, (1, 1), 1),
               AffineRecord((2, 2), (0, -1), 3, (-1, 1), 3))
    for dtype, coefficient_dtype in (("float32", "complex128"), ("complex64", "float64")):
        for coefficient_shape in ((), (1,), (2, 4)):
            storage_sharding = NamedSharding(mesh, P(None, "device", None))
            coefficient_sharding = NamedSharding(mesh, P() if coefficient_shape in ((), (1,)) else P(None, "device"))
            host = (jnp.arange(40) % 5 / 4).reshape(2, 4, 5).astype(dtype)
            if dtype.startswith("complex"):
                host = host + .5j
            factor_host = (jnp.full(coefficient_shape, 2, coefficient_dtype) if coefficient_shape in ((), (1,))
                           else (jnp.arange(8) % 3).reshape(coefficient_shape).astype(coefficient_dtype))
            source, factor = jax.device_put(host, storage_sharding), jax.device_put(factor_host, coefficient_sharding)
            def run(data, alpha):
                return accumulation_p.bind(data, alpha, records=records, coefficient_records=(0,), output_size=6, dtype=data.dtype)
            def reference(data, alpha):
                alpha = alpha.reshape(()) if alpha.shape == (1,) else alpha
                result = jnp.zeros((*data.shape[:-1], 6), dtype=data.dtype)
                for record_index, record in enumerate(records):
                    for source_index, destination_index in address_pairs(record):
                        contribution = data[..., source_index] * (alpha if record_index == 0 else 1)
                        if not jnp.iscomplexobj(data):
                            contribution = jnp.real(contribution)
                        result = result.at[..., destination_index].add(contribution.astype(data.dtype))
                return result
            ct_host = jnp.full((2, 4, 6), 1 + 2j if dtype.startswith("complex") else 2, dtype)
            cotangent = jax.device_put(ct_host, storage_sharding)
            reverse = jax.jit(lambda data, alpha, value: jax.vjp(run, data, alpha)[1](value),
                in_shardings=(storage_sharding, coefficient_sharding, storage_sharding),
                out_shardings=(storage_sharding, coefficient_sharding)
            ).lower(source, factor, cotangent).compile()
            verify(reverse, (source, factor, cotangent), jax.vjp(reference, host, factor_host)[1](ct_host),
                   (storage_sharding, coefficient_sharding), ("tensor0_stride_dot_",),
                   global_sum=coefficient_shape in ((), (1,)))
            direct = jax.jit(lambda data, alpha, value: jax.linear_transpose(lambda coefficient: run(data, coefficient), alpha)(value)[0],
                in_shardings=(storage_sharding, coefficient_sharding, storage_sharding), out_shardings=coefficient_sharding
            ).lower(source, factor, cotangent).compile()
            verify(direct, (source, factor, cotangent), jax.vjp(reference, host, factor_host)[1](ct_host)[1],
                   coefficient_sharding, ("tensor0_stride_dot_",), global_sum=coefficient_shape in ((), (1,)))

def run_transform(mode):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
    from tensor0 import FermionParity, SU2Irrep, TensorMap, flip, twist, tensortrace, hom, space, to_dense, from_dense
    from tensor0 import braid, transpose, repartition
    from tensor0.structure import get_degeneracystructure
    import tensor0.operations.transforms as transforms
    import tensor0.operations.contractions.primitives as contractions
    from tensor0._stride._ops._transform import _strided_affine_transform, _strided_tree_transform
    from tensor0._stride._ops._reduction import _strided_tensortrace
    from tests.stride.test_grouped_transform_contract import oracle

    jax.config.update("jax_enable_x64", True)
    mesh = Mesh(np.asarray(jax.devices()), ("device",),
                axis_types=(AxisType.Explicit if mode == "explicit" else AxisType.Auto,))
    assert len(jax.devices()) == 2

    def verify(compiled, arguments, expected, shardings):
        actual = compiled(*arguments)
        for value, wanted, sharding in zip(jax.tree.leaves(actual), jax.tree.leaves(expected),
                                          jax.tree.leaves(shardings), strict=True):
            assert value.shape == wanted.shape and value.dtype == wanted.dtype
            np.testing.assert_allclose(value, wanted, rtol=3e-6, atol=3e-6)
            assert value.sharding.is_equivalent_to(sharding, value.ndim)
            for shard in value.addressable_shards:
                np.testing.assert_allclose(shard.data, np.asarray(wanted)[shard.index], rtol=3e-6, atol=3e-6)
        hlo = compiled.as_text().lower()
        for name in ("all-gather", "all-reduce", "all-to-all", "collective-permute"):
            assert name not in hlo, hlo

    seen = []
    def affine(source, **arguments):
        arguments["entries"] = tuple(arguments["entries"])
        seen.append(arguments)
        return original_affine(source, **arguments)

    trees = []
    def tree(source, **arguments):
        trees.append(arguments)
        return original_tree(source, **arguments)


    parity = space(FermionParity, {0: 2, 1: 1})
    half = space(SU2Irrep, {1: 1})
    tree_half = space(SU2Irrep, {1: 2})
    cases = (
        ("twist", hom((parity, parity), (parity, parity))),
        ("flip", hom((parity, parity), (parity, parity))),
        ("trace", hom((half, half), (half, half))),
        ("full_trace", hom((half,), (half,))),
        ("grouped_trace", hom((half, half, half, half.dual()), ())),
        ("braid", hom((tree_half, tree_half), (tree_half, tree_half))),
        ("transpose", hom((tree_half, tree_half), (tree_half, tree_half))),
        ("repartition", hom((tree_half, tree_half), (tree_half, tree_half))),
    )
    original_affine, original_trace = transforms._strided_affine_transform, contractions._strided_tensortrace
    assert original_affine is _strided_affine_transform
    assert original_trace is _strided_tensortrace
    original_tree = transforms._strided_tree_transform
    assert original_tree is _strided_tree_transform
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(transforms, "_strided_affine_transform", affine)
        patch.setattr(transforms, "_strided_tree_transform", tree)
        for name, target in cases:
            size = get_degeneracystructure(target).total_dim
            def operation(tensor):
                if name == "twist":
                    return twist(tensor, 0)
                if name == "flip":
                    return flip(tensor, (0, 2))
                if name == "trace":
                    return tensortrace(tensor, axes=((1,), (3,)), output=((0,), (2,)))
                if name == "full_trace":
                    return tensortrace(tensor, axes=((0,), (1,)), output=((), ()))
                if name == "braid":
                    return braid(tensor, ((1, 0), (3, 2)), (3, 1, 2, 0))
                if name == "transpose":
                    return transpose(tensor)
                if name == "repartition":
                    return repartition(tensor, 3)
                return tensortrace(tensor, axes=((3,), (0,)), output=((1, 2), ()))

            single = lambda data: operation(TensorMap(target, data)).storage.data
            seen.clear()
            trees.clear()
            output_tensor = operation(TensorMap(target, jnp.ones(size)))
            output_size = output_tensor.storage.data.size
            native_calls = 1
            native_operation = "reduction" if "trace" in name else "accumulation"
            if "trace" in name:
                columns = []
                for basis in jnp.eye(size, dtype=jnp.float64):
                    dense = to_dense(TensorMap(target, basis))
                    reduced = (jnp.trace(dense, axis1=1, axis2=3) if name == "trace" else
                               jnp.trace(dense) if name == "full_trace" else
                               jnp.trace(jnp.transpose(dense, (1, 2, 3, 0)), axis1=2, axis2=3))
                    columns.append(np.asarray(from_dense(output_tensor.space, reduced).storage.data))
                matrix = np.stack(columns, axis=1)
            elif trees:
                assert len(trees) == 1
                metadata = trees[0]
                assert metadata["transformer"].kind == "generic"
                groups = metadata["transformer"].generic_data
                matrix = np.asarray(oracle(jnp.eye(size, dtype=jnp.float64), metadata["source_layout"],
                    metadata["destination_layout"], jnp.float64, metadata["permutation"], groups)).T
                scalar_groups = [np.shape(group.transform) == (1, 1) for group in groups]
                native_calls = 1 if all(scalar_groups) else 3 if any(scalar_groups) else 2
                native_operation = "accumulation" if any(scalar_groups) else "copy"
            else:
                assert len(seen) == 1
                metadata = seen[0]
                matrix = np.zeros((output_size, size))
                for source_index, destination_index, factor in metadata["entries"]:
                    source_block = metadata["source_subblocks"][source_index]
                    destination_block = metadata["destination_subblocks"][destination_index]
                    for coordinates in np.ndindex(tuple(source_block.sizes)):
                        source_offset = source_block.offset + sum(index * stride for index, stride in
                                                                  zip(coordinates, source_block.strides))
                        destination_offset = destination_block.offset + sum(index * stride for index, stride in
                                                                            zip(coordinates, destination_block.strides))
                        matrix[destination_offset, source_offset] += factor

            for dtype in ("float32", "complex64"):
                operator = jnp.asarray(matrix, dtype=dtype)
                reference = lambda data: data @ operator.T
                for batch_shape, batch_spec in (((4,), ("device",)), ((2, 4), (None, "device"))):
                    run = jax.vmap(single) if len(batch_shape) == 1 else jax.vmap(jax.vmap(single))
                    host = (jnp.arange(np.prod(batch_shape) * size) % 7 / 4).reshape(*batch_shape, size).astype(dtype)
                    if dtype.startswith("complex"):
                        host = host + .5j * (host + 1)
                    sharding = NamedSharding(mesh, P(*batch_spec, None))
                    source = jax.device_put(host, sharding)
                    compiled = jax.jit(run, in_shardings=sharding, out_shardings=sharding).lower(source).compile()
                    verify(compiled, (source,), reference(host), sharding)
                    hlo = compiled.as_text().lower()
                    assert hlo.count('custom_call_target="tensor0_stride_') == native_calls
                    assert f"tensor0_stride_{native_operation}_" in hlo
                    verify(jax.jit(run).lower(source).compile(), (source,), reference(host), sharding)
                    forward = jax.jit(lambda data: jax.jvp(run, (data,), (data,)),
                        in_shardings=sharding, out_shardings=(sharding, sharding)).lower(source).compile()
                    verify(forward, (source,), jax.jvp(reference, (host,), (host,)), (sharding, sharding))
                    ct_host = jnp.full((*batch_shape, output_size), 1 + 2j if dtype.startswith("complex") else 2, dtype)
                    cotangent = jax.device_put(ct_host, sharding)
                    reverse = jax.jit(lambda data, ct: jax.vjp(run, data)[1](ct)[0],
                        in_shardings=(sharding, sharding), out_shardings=sharding).lower(source, cotangent).compile()
                    verify(reverse, (source, cotangent), jax.vjp(reference, host)[1](ct_host)[0], sharding)

                    if name in ("twist", "flip"):
                        def roundtrip_single(data):
                            transformed = operation(TensorMap(target, data))
                            restored = (twist(transformed, 0) if name == "twist" else
                                        flip(transformed, (0, 2), inv=True))
                            return restored.storage.data
                        roundtrip = (jax.vmap(roundtrip_single) if len(batch_shape) == 1 else
                                     jax.vmap(jax.vmap(roundtrip_single)))
                        compiled = jax.jit(roundtrip, in_shardings=sharding,
                                           out_shardings=sharding).lower(source).compile()
                        verify(compiled, (source,), host, sharding)
                        reverse_roundtrip = jax.jit(lambda data: jax.vjp(roundtrip, data)[1](data)[0],
                            in_shardings=sharding, out_shardings=sharding).lower(source).compile()
                        verify(reverse_roundtrip, (source,), host, sharding)

                def objective(function, data):
                    result = function(data)
                    return jnp.real(jnp.sum(result * jnp.conj(result)))
                gradient = jax.grad(lambda data: objective(run, data))
                expected_gradient = jax.grad(lambda data: objective(reference, data))
                higher = jax.jit(lambda data: jax.jvp(gradient, (data,), (data,))[1],
                    in_shardings=sharding, out_shardings=sharding).lower(source).compile()
                verify(higher, (source,), jax.jvp(expected_gradient, (host,), (host,))[1], sharding)

            host = jnp.zeros((0, size), jnp.float32)
            sharding = NamedSharding(mesh, P("device", None))
            source = jax.device_put(host, sharding)
            compiled = jax.jit(jax.vmap(single), in_shardings=sharding, out_shardings=sharding).lower(source).compile()
            verify(compiled, (source,), jnp.zeros((0, output_size), jnp.float32), sharding)

    assert transforms._strided_affine_transform is original_affine
    assert contractions._strided_tensortrace is original_trace
    assert transforms._strided_tree_transform is original_tree
