# Contractions

Tensor0 provides low-level axis APIs and higher-level label APIs for
symmetry-aware contractions. The general axis and label APIs return a
`TensorMap`; even a fully contracted network returns a rank-zero `TensorMap`,
whose value is extracted explicitly with `scalar(...)`. The `tr(...)`
convenience operation instead returns a JAX scalar directly.

Visible axes are numbered as codomain axes followed by domain axes. Output
arguments always specify both axis order and the resulting HomSpace partition.

## Choosing an Operation

| Operation | Scope | Index specification | Result |
|---|---|---|---|
| `tr` | Full square-map trace | Implicit fused HomSpace blocks | JAX scalar |
| `tensortrace` | General single-tensor trace | Axis tuples | `TensorMap` |
| `tensorcontract` | Two tensors | Axis tuples and output references | `TensorMap` |
| `contract` | One or more tensors | Named labels | `TensorMap` |
| `ncon` | One or more tensors | Signed integer labels | `TensorMap` |

Use the axis APIs when code already knows exact axis positions. Use `contract`
for readable hand-written tensor networks and `ncon` for conventional integer
network descriptions or programmatically generated networks.

## Binary Contraction

`tensorcontract` contracts paired axes and places every open axis through an
explicit `(operand, axis)` reference. Operand `0` is the left tensor and
operand `1` is the right tensor.

```python
result = tensorcontract(
    left,
    right,
    axes=((1,), (0,)),
    output=(((0, 0),), ((1, 1),)),
)
```

This contracts left axis `1` with right axis `0`, keeps left axis `0` in the
result codomain, and keeps right axis `1` in the result domain. Contracted
spaces must be dual-compatible. Every original axis must appear exactly once
in either `axes` or `output`.

Pass `conjugate=(True, False)` or `conjugate=(False, True)` to contract an
operand through its adjoint orientation. The flags are static operation
metadata, not dynamic JAX array values.

## Single-Tensor Trace

`tensortrace` pairs axes of one tensor and explicitly orders the remaining
axes:

```python
partial = tensortrace(
    tensor,
    axes=((1,), (3,)),
    output=((0,), (2,)),
)
```

For a full trace, both output groups are empty:

```python
rank_zero = tensortrace(
    matrix,
    axes=((0,), (1,)),
    output=((), ()),
)
value = scalar(rank_zero)
```

Unlike `tr(matrix)`, the general trace operation deliberately preserves the
rank-zero TensorMap representation. `conjugate=True` traces the tensor after
applying its adjoint orientation.

## Named-Label Networks

`idx` binds comma-separated Python identifiers to visible axes in codomain-
then-domain order:

```python
result = contract(
    idx(left, "a,x"),
    idx(middle, "x,y"),
    idx(right, "y,b"),
    output=("a", "b"),
)
```

Open labels occur once and must appear exactly once in `output`. Contracted
labels occur exactly twice and must join dual-compatible spaces. The output can
be either a pair of comma-separated strings or one semicolon-separated string:

```python
output=("a,c", "b,d")
output="a,c;b,d"
```

Tuple operands provide a compact alternative to `idx`:

```python
result = contract(
    (left, "a,x"),
    (right, "x,b"),
    output="a;b",
)
```

A third tuple entry or `idx(..., conjugate=True)` requests conjugation.

Without `order`, named contraction folds operands from left to right. An exact
tuple/list of contracted labels selects another deterministic association:

```python
result = contract(
    idx(left, "a,x"),
    idx(middle, "x,y"),
    idx(right, "y,b"),
    output=("a", "b"),
    order=("y", "x"),
)
```

The earliest remaining label chooses a tensor pair. If that pair shares more
than one label, all shared labels are contracted in the same binary operation.
The order must contain every contracted label exactly once, including labels
consumed by a self trace. Tensor products combine disconnected components
after all contraction labels are consumed.

## Integer-Label Networks

`ncon` uses positive labels for contractions and negative labels for open
indices:

```python
result = ncon(
    (left, middle, right),
    ((-1, 1), (1, 2), (2, -2)),
    order=(2, 1),
    output=((-1,), (-2,)),
)
```

Positive labels occur exactly twice; negative labels occur exactly once; zero
is invalid. By default, positive labels are processed in increasing order.
When output is omitted, negative labels retain their effective codomain/domain
side and are ordered as `-1`, `-2`, and so on within each side.

Explicit output is always a `(codomain_labels, domain_labels)` pair because a
TensorMap result must preserve the HomSpace partition. `conjugate` is an
optional tuple/list containing one boolean per tensor.

## Twist

`twist(tensor, indices, inv=False)` applies the ribbon twist to selected
visible indices without changing the tensor's HomSpace:

```python
twisted = twist(tensor, (0, 2))
restored = twist(twisted, (0, 2), inv=True)
```

Bosonic and otherwise trivial twists return the original tensor. Fermionic
odd sectors acquire the corresponding sign. Current public contraction paths
support symmetric braiding; anyonic and planar contraction are outside the v1
scope.

## Index Flip

`flip(tensor, indices, inv=False)` changes the arrow presentation of selected
visible indices and applies the corresponding Z-isomorphism weights. If two
matching legs will be contracted, flipping both legs leaves the contraction
unchanged. See the [transform guide](usage.md#transforms) for its full inverse
and HomSpace semantics.

## JAX Boundary

TensorMap storage is the dynamic pytree leaf. HomSpace, labels, output groups,
conjugation flags, and contraction order are static metadata used while JAX
traces the Python operation.

```python
@jax.jit
def network(left, middle, right):
    return contract(
        idx(left, "a,x"),
        idx(middle, "x,y"),
        idx(right, "y,b"),
        output=("a", "b"),
        order=("y", "x"),
    )
```

Changing only storage values reuses the same static structure. Changing a
HomSpace or a closed-over label/output/order specification creates a different
specialization. Differentiation flows through storage arrays; operation
metadata is not differentiable.

## Current Boundaries

Tensor0 does not currently provide automatic contraction-order optimization,
hyperedges, repeated open labels, diagonal extraction through repeated output
labels, ellipsis/broadcasting, mutable destinations, or a native JAX numerical
kernel. Dense conversion remains a small correctness tool and is not used by
the contraction implementation.
