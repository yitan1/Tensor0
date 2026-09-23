"""Isolated reduction partitioning checks."""


def _verify_sharded_result(compiled, arguments, expected, shardings, global_sum=False):
    import jax
    import numpy as np

    actual = compiled(*arguments)
    for value, wanted, sharding in zip(jax.tree.leaves(actual), jax.tree.leaves(expected),
                                      jax.tree.leaves(shardings), strict=True):
        assert value.shape == wanted.shape and value.dtype == wanted.dtype
        np.testing.assert_allclose(value, wanted, rtol=3e-6, atol=3e-6)
        assert value.sharding.is_equivalent_to(sharding, value.ndim)
        for shard in value.addressable_shards:
            np.testing.assert_allclose(shard.data, np.asarray(wanted)[shard.index], rtol=3e-6, atol=3e-6)
    hlo = compiled.as_text().lower()
    for name in ("all-gather", "all-to-all", "collective-permute"):
        assert name not in hlo, hlo
    assert ("all-reduce" in hlo) == global_sum, hlo


def _run_reduction_worker(mode):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
    from tensor0._stride import StridedView, reduce_sum
    from tests.stride.support.oracles.reduction import execute
    from tests.stride.support.oracles.reduction import operands
    from tests.stride.support.oracles.reduction import reference
    from tests.stride.support.oracles.reduction import functions
    from tests.stride.support.samples import values

    jax.config.update("jax_enable_x64", True)
    mesh = Mesh(np.asarray(jax.devices()), ("device",),
                axis_types=(AxisType.Explicit if mode == "explicit" else AxisType.Auto,))
    assert len(jax.devices()) == 2
    if mode.endswith("storage"):
        source_sharding = NamedSharding(mesh, P("device") if mode == "input_storage" else P())
        output_sharding = NamedSharding(mesh, P("device") if mode == "output_storage" else P())
        source = jax.device_put(np.arange(8, dtype=np.float32), source_sharding)
        function = lambda data: reduce_sum(StridedView(data, (4, 2), (2, 1), 0), (1,))
        try:
            jax.jit(function, in_shardings=source_sharding, out_shardings=output_sharding).lower(source).compile()
        except Exception as error:
            assert "cannot shard the packed storage axis" in str(error), str(error)
        else:
            raise AssertionError("Reduction accepted a partitioned storage axis")
        return


    for batch_shape, batch_spec in (((4,), ("device",)), ((2, 4), (None, "device"))):
        storage = NamedSharding(mesh, P(*batch_spec, None))
        batch = NamedSharding(mesh, P(*batch_spec))
        # Cover each coefficient form and both batch ranks in each mesh mode;
        # swap numeric kinds rather than crossing all precision/shape choices.
        kinds = ("float32", "complex128", "complex128") if len(batch_shape) == 1 else (
            "complex128", "float32", "float32")
        for dtype, coefficient_shape in zip(kinds, ((), (1,), batch_shape), strict=True):
            shared = coefficient_shape in ((), (1,))
            coefficient = NamedSharding(mesh, P()) if shared else batch
            shardings = (storage, coefficient, coefficient, batch)
            host = operands(dtype, batch_shape, coefficient_shape)
            host = (*host[:3], (jnp.arange(np.prod(batch_shape)) % 3).reshape(batch_shape).astype(dtype))
            arguments = tuple(jax.device_put(value, sharding) for value, sharding in zip(host, shardings))
            compiled = jax.jit(execute, in_shardings=shardings, out_shardings=storage).lower(*arguments).compile()
            _verify_sharded_result(compiled, arguments, reference(*host), storage)
            assert "tensor0_stride_reduction_" in compiled.as_text()
            _verify_sharded_result(jax.jit(execute).lower(*arguments).compile(), arguments, reference(*host), storage)
            forward = jax.jit(lambda *values: jax.jvp(execute, values, values),
                in_shardings=shardings, out_shardings=(storage, storage)).lower(*arguments).compile()
            _verify_sharded_result(forward, arguments, jax.jvp(reference, host, host), (storage, storage))
            ct_host = jnp.full((*batch_shape, 7), 1 + 2j if dtype.startswith("complex") else 2, dtype)
            cotangent = jax.device_put(ct_host, storage)
            reverse = jax.jit(lambda *values: jax.vjp(execute, *values[:-1])[1](values[-1]),
                in_shardings=(*shardings, storage), out_shardings=shardings).lower(*arguments, cotangent).compile()
            _verify_sharded_result(reverse, (*arguments, cotangent), jax.vjp(reference, *host)[1](ct_host), shardings, shared)

        for shape, strides, offset in (((2, 3), (3, -1), 2), ((2, 0), (1, 1), 8)):
            host = jnp.arange(np.prod(batch_shape) * 8, dtype=jnp.float32).reshape(*batch_shape, 8)
            source = jax.device_put(host, storage)
            for axes in (((1,), None) if 0 in shape else ((1,), None, ())):
                def run(data):
                    return reduce_sum(StridedView(data, shape, strides, offset), axes)
                indices = np.array([offset + sum(index * stride for index, stride in zip(coords, strides))
                                    for coords in np.ndindex(shape)], dtype=np.int32).reshape(shape)
                logical = host[..., indices]
                logical_axes = tuple(range(len(shape))) if axes is None else axes
                expected = jnp.sum(logical, axis=tuple(len(batch_shape) + axis for axis in logical_axes))
                output = NamedSharding(mesh, P(*batch_spec, *((None,) * (expected.ndim - len(batch_shape)))))
                compiled = jax.jit(run, in_shardings=storage, out_shardings=output).lower(source).compile()
                _verify_sharded_result(compiled, (source,), expected, output)

    storage = NamedSharding(mesh, P("device", None))
    batch = NamedSharding(mesh, P("device"))
    for source_dtype, coefficient_dtype, dtype in (("float32", "complex128", "complex64"),
                                                  ("complex64", "float64", "float32")):
        run, expected_run = functions("reduction", dtype)
        host = (values((4, 5), source_dtype), jnp.full((4,), 2, coefficient_dtype),
                jnp.full((4,), .5, dtype))
        shardings = (storage, batch, batch)
        arguments = tuple(jax.device_put(value, sharding) for value, sharding in zip(host, shardings))
        ct_host = jnp.full((4, 6), 1 + 2j if dtype.startswith("complex") else 2, dtype)
        cotangent = jax.device_put(ct_host, storage)
        compiled = jax.jit(run, in_shardings=shardings, out_shardings=storage).lower(*arguments).compile()
        _verify_sharded_result(compiled, arguments, expected_run(*host), storage)
        reverse = jax.jit(lambda *inputs: jax.vjp(run, *inputs[:-1])[1](inputs[-1]),
            in_shardings=(*shardings, storage), out_shardings=shardings).lower(*arguments, cotangent).compile()
        _verify_sharded_result(reverse, (*arguments, cotangent), jax.vjp(expected_run, *host)[1](ct_host), shardings)
        direct = jax.jit(lambda source, first, second, ct:
            jax.linear_transpose(lambda *factors: run(source, *factors), first, second)(ct),
            in_shardings=(*shardings, storage), out_shardings=(batch, batch)).lower(*arguments, cotangent).compile()
        _verify_sharded_result(direct, (*arguments, cotangent), jax.vjp(expected_run, *host)[1](ct_host)[1:], (batch, batch))

    for batch_shape, batch_spec in (((0,), ("device",)), ((), ())):
        storage = NamedSharding(mesh, P(*batch_spec, None))
        batch = NamedSharding(mesh, P(*batch_spec))
        host = operands("float32", batch_shape, batch_shape)
        shardings = (storage, batch, batch, batch)
        arguments = tuple(jax.device_put(value, sharding) for value, sharding in zip(host, shardings))
        compiled = jax.jit(execute, in_shardings=shardings, out_shardings=storage).lower(*arguments).compile()
        _verify_sharded_result(compiled, arguments, reference(*host), storage)


def _run_mapped_coefficients_worker(operation, mode):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
    from tests.stride.support.oracles.reduction import functions
    from tests.stride.support.samples import values

    jax.config.update("jax_enable_x64", True)
    mesh = Mesh(np.asarray(jax.devices()), ("device",),
                axis_types=(AxisType.Explicit if mode == "explicit" else AxisType.Auto,))
    assert len(jax.devices()) == 2


    # Every mapped/shared axis pattern runs in both mesh modes and operations.
    # Both numeric kinds have local gradients and gradients needing all-reduce.
    cases = (
        ("float32", (), (), (), (0, 0, 0)),
        ("complex64", (), (1,), (1,), (1, 0, 0)),
        ("float32", (), (), (), (None, 0, 0)),
        ("complex64", (3,), (), (), (0, 0, 0)),
        ("float32", (3,), (1,), (3,), (0, 0, None)),
        ("complex64", (3,), (3,), (), (None, 1, None)),
        ("float32", (3,), (3,), (3,), (0, None, None)),
    )
    for dtype, row_batch, first_shape, second_shape, axes in cases:
        run, reference = functions(operation, dtype)
        host, shardings = [], []
        shapes = ((*row_batch, 5), first_shape, second_shape)
        for index, (row_shape, axis) in enumerate(zip(shapes, axes)):
            shape = row_shape if axis is None else (4, *row_shape)
            value = values(shape, dtype if index == 0 else "float64")
            if index == 1:
                value = (jnp.arange(value.size) % 3).reshape(shape).astype(jnp.float64)
            if index == 2:
                value = value + .5
            if axis is not None:
                value = jnp.moveaxis(value, 0, axis)
            spec = [None] * value.ndim
            if axis is not None:
                spec[axis] = "device"
            host.append(value)
            shardings.append(NamedSharding(mesh, P(*spec)))
        host, shardings = tuple(host), tuple(shardings)
        arguments = tuple(jax.device_put(value, sharding) for value, sharding in zip(host, shardings))
        output_sharding = NamedSharding(mesh, P("device", *((None,) * (len(row_batch) + 1))))
        mapped, oracle = jax.vmap(run, in_axes=axes), jax.vmap(reference, in_axes=axes)
        compiled = jax.jit(mapped, in_shardings=shardings, out_shardings=output_sharding).lower(*arguments).compile()
        _verify_sharded_result(compiled, arguments, oracle(*host), output_sharding)
        assert compiled.as_text().count(f'custom_call_target="tensor0_stride_{operation}_') == 1
        forward = jax.jit(lambda *inputs: jax.jvp(mapped, inputs, inputs), in_shardings=shardings,
            out_shardings=(output_sharding, output_sharding)).lower(*arguments).compile()
        _verify_sharded_result(forward, arguments, jax.jvp(oracle, host, host), (output_sharding, output_sharding))
        ct_host = jnp.full((4, *row_batch, 6), 1 + 2j if dtype == "complex64" else 2, dtype)
        cotangent = jax.device_put(ct_host, output_sharding)
        reverse = jax.jit(lambda *inputs: jax.vjp(mapped, *inputs[:-1])[1](inputs[-1]),
            in_shardings=(*shardings, output_sharding), out_shardings=shardings).lower(*arguments, cotangent).compile()
        _verify_sharded_result(reverse, (*arguments, cotangent), jax.vjp(oracle, *host)[1](ct_host), shardings,
               global_sum=any(axis is None for axis in axes))
