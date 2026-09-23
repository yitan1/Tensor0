"""Isolated dot partitioning checks."""


def run_dot(mode):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
    from tensor0._stride import StridedView, dotc, dotu
    from tensor0._stride._jax import accumulation_p
    from tensor0._stride._layout import AffineRecord
    from tests.stride.support.oracles.dot import LAYOUTS
    from tests.stride.support.oracles.dot import operands
    from tests.stride.support.oracles.dot import dot_functions as functions
    from tests.stride.support.oracles.update import address_pairs

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

    # Dtype/layout arithmetic is exhaustive on one device. Here each numeric
    # direction runs through both conjugation modes, with both ranks/layouts
    # represented, rather than duplicating their Cartesian product.
    cases = (
        ((4,), ("device",), ("float32", "complex64", "complex64"), LAYOUTS[0]),
        ((2, 4), (None, "device"), ("complex128", "float64", "float32"), LAYOUTS[1]),
        ((4,), ("device",), ("float64", "float32", "float64"), LAYOUTS[1]),
        ((2, 4), (None, "device"), ("complex64", "complex128", "complex128"), LAYOUTS[0]),
    )
    for batch_shape, batch_spec, (left_dtype, right_dtype, dtype), layout in cases:
        storage_sharding = NamedSharding(mesh, P(*batch_spec, None))
        result_sharding = NamedSharding(mesh, P(*batch_spec))
        reference_arguments = operands(left_dtype, right_dtype, batch_shape)
        arguments = tuple(jax.device_put(value, storage_sharding) for value in reference_arguments)
        for operation in (dotu, dotc):
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
