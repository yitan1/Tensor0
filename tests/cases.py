from dataclasses import dataclass

import jax.numpy as jnp

from tensor0 import (
    FermionParity,
    HomSpace,
    SU2Irrep,
    U1Irrep,
    U1SU2Irrep,
    get_degeneracystructure,
    hom,
    space,
)


@dataclass(frozen=True)
class HomCase:
    name: str
    space: HomSpace


@dataclass(frozen=True)
class DenseCase:
    name: str
    space: HomSpace
    dense_shape: tuple[int, ...]


@dataclass(frozen=True)
class TransformCase:
    name: str
    space: HomSpace
    permutation: tuple[tuple[int, ...], tuple[int, ...]]
    expected_space: HomSpace
    dense_axes: tuple[int, ...]


@dataclass(frozen=True)
class TensorMapCase:
    name: str
    space: HomSpace


@dataclass(frozen=True)
class TensorMapMatmulCase:
    name: str
    left_space: HomSpace
    right_space: HomSpace
    result_space: HomSpace


def float_data_for(space_obj: HomSpace):
    total_dim = get_degeneracystructure(space_obj).total_dim
    return jnp.arange(1, total_dim + 1, dtype=jnp.float32) / 10.0


def zero_based_float_data_for(space_obj: HomSpace):
    total_dim = get_degeneracystructure(space_obj).total_dim
    return jnp.arange(total_dim, dtype=jnp.float32)


def assert_allclose(left, right):
    assert left.shape == right.shape
    assert bool(jnp.allclose(left, right, rtol=1e-5, atol=1e-6))


def assert_tensormap_blocks_allclose(left, right):
    assert left.space == right.space
    for coupled, block in right.blocks():
        assert_allclose(left.block(coupled), block)


def dense_roundtrip_cases():
    u1_left = space(U1Irrep, {0: 2, 1: 1})
    u1_right = space(U1Irrep, {0: 1, 1: 2})
    u1_mid = space(U1Irrep, {0: 1, -1: 2})
    u1_out = space(U1Irrep, {0: 1, 1: 1})

    parity = space(FermionParity, {0: 2, 1: 1})
    half = space(SU2Irrep, {1: 1})
    su2_left = space(SU2Irrep, {0: 2, 1: 3})
    su2_right = space(SU2Irrep, {0: 5, 1: 7})
    half_with_charge = space(U1SU2Irrep, {(0, 1): 1})

    return (
        DenseCase(
            "u1-two-leg",
            hom((u1_left,), (u1_right,)),
            (3, 3),
        ),
        DenseCase(
            "u1-multileg",
            hom((u1_left, u1_mid), (u1_out,)),
            (3, 3, 2),
        ),
        DenseCase(
            "fermion-parity-two-leg",
            hom((parity,), (parity,)),
            (3, 3),
        ),
        DenseCase(
            "su2-half-half-singlet",
            hom((), (half, half)),
            (2, 2),
        ),
        DenseCase(
            "su2-innerline",
            hom((half, half, half, half), ()),
            (2, 2, 2, 2),
        ),
        DenseCase(
            "u1-su2-product",
            hom((), (half_with_charge, half_with_charge)),
            (2, 2),
        ),
        DenseCase(
            "su2-strided-subblocks",
            hom((su2_left, su2_right), (su2_left, su2_right)),
            (8, 19, 8, 19),
        ),
    )


def tensor_map_cases():
    u1 = space(U1Irrep, {0: 2, 1: 3})
    u1_left = space(U1Irrep, {0: 2, 1: 3})
    u1_right = space(U1Irrep, {0: 5, 1: 7})
    half = space(SU2Irrep, {1: 1})
    product_type = U1Irrep @ FermionParity
    product = space(product_type, {(0, 0): 2, (1, 1): 3})

    return (
        TensorMapCase("u1-endomorphism", hom((u1,), (u1,))),
        TensorMapCase("u1-strided", hom((u1_left, u1_right), (u1_left, u1_right))),
        TensorMapCase("su2-half", hom((half,), (half,))),
        TensorMapCase("u1-fermion-product", hom((product,), (product,))),
    )


def tensor_map_matmul_cases():
    u1_left = space(U1Irrep, {0: 2, 1: 3})
    u1_mid = space(U1Irrep, {0: 5, 1: 7})
    u1_right = space(U1Irrep, {0: 11, 1: 13})

    product_type = U1Irrep @ FermionParity
    product_left = space(product_type, {(0, 0): 2, (1, 1): 3})
    product_mid = space(product_type, {(0, 0): 5, (1, 1): 7})
    product_right = space(product_type, {(0, 0): 11, (1, 1): 13})

    return (
        TensorMapMatmulCase(
            "u1",
            hom((u1_left,), (u1_mid,)),
            hom((u1_mid,), (u1_right,)),
            hom((u1_left,), (u1_right,)),
        ),
        TensorMapMatmulCase(
            "u1-fermion-product",
            hom((product_left,), (product_mid,)),
            hom((product_mid,), (product_right,)),
            hom((product_left,), (product_right,)),
        ),
    )


def factorization_cases():
    u1_left = space(U1Irrep, {0: 2, 1: 4})
    u1_right = space(U1Irrep, {0: 3, 1: 2})

    parity_left = space(FermionParity, {0: 2, 1: 3})
    parity_right = space(FermionParity, {0: 4, 1: 2})

    su2_left = space(SU2Irrep, {0: 2, 1: 3})
    su2_right = space(SU2Irrep, {0: 5, 1: 7})

    product_left = space(U1SU2Irrep, {(0, 0): 2, (0, 1): 1})
    product_right = space(U1SU2Irrep, {(0, 0): 1, (0, 1): 2})

    u1_a = space(U1Irrep, {0: 2, 1: 3})
    u1_b = space(U1Irrep, {0: 5, -1: 7})
    u1_c = space(U1Irrep, {0: 4, 1: 9})

    return (
        HomCase("u1-rectangular", hom((u1_left,), (u1_right,))),
        HomCase(
            "fermion-parity",
            hom((parity_left,), (parity_right,)),
        ),
        HomCase("su2", hom((su2_left,), (su2_right,))),
        HomCase("u1-su2-product", hom((product_left,), (product_right,))),
        HomCase("u1-multileg", hom((u1_a, u1_b), (u1_c,))),
    )


def transform_cases():
    u1_left = space(U1Irrep, {0: 2, 1: 1})
    u1_right = space(U1Irrep, {0: 1, 1: 2})
    u1_out = space(U1Irrep, {1: 1})
    u1_space = hom((u1_left, u1_right), (u1_out,))

    half = space(SU2Irrep, {1: 1})
    su2_space = hom((half, half, half), (half,))

    half_with_charge = space(U1SU2Irrep, {(0, 1): 1})
    product_space = hom(
        (half_with_charge, half_with_charge, half_with_charge),
        (half_with_charge,),
    )

    return (
        TransformCase(
            "u1",
            u1_space,
            ((1,), (0, 2)),
            hom((u1_right,), (u1_left.dual(), u1_out)),
            (1, 0, 2),
        ),
        TransformCase(
            "su2-half-spin",
            su2_space,
            ((1, 2), (0, 3)),
            hom((half, half), (half.dual(), half)),
            (1, 2, 0, 3),
        ),
        TransformCase(
            "u1-su2-product",
            product_space,
            ((1, 2), (0, 3)),
            hom(
                (half_with_charge, half_with_charge),
                (half_with_charge.dual(), half_with_charge),
            ),
            (1, 2, 0, 3),
        ),
    )
