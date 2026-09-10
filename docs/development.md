# Development

Tensor0 is developed from a source checkout with Python, Rust, uv, maturin, and
pytest.

## Local Setup

```bash
uv sync --group dev
uv run maturin develop
```

## Verification

```bash
cargo test
uv run pytest tests -q
uv run python examples/basic_usage.py
uv run python examples/contractions.py
```

## Documentation Site

Build the documentation site locally with:

```bash
uv run --with mkdocs==1.6.1 mkdocs build --strict
```

The public documentation site is deployed to GitHub Pages by
`.github/workflows/docs.yml` on pushes to `main` and by manual workflow
dispatch.

The documentation site is public-facing. Internal planning notes under `local/`
are not part of the public navigation.

## Affine View Updates

`tensor0._stride` provides `materialize(view, *, dtype=None)`, `scale(view,
alpha)`, `add(left, right, *, alpha=1, beta=1)`, `dotu`, `dotc`, and
`reduce_sum`. Ordinary JAX arrays continue to use JAX operations. View updates
return new storage while preserving the destination layout, dtype, and values
outside the selection.

`scale` and `add` share one forward update primitive. Internal assignment and
accumulation callers use `_execute_update(base, source, *, source_factor,
base_factor, plan)` with factors `(1, 0)` and `(1, 1)`. For `add`, the source is
`right`: `source_factor=beta` and `base_factor=alpha`. Fresh maps, reductions,
dot products, and multi-record transform plans remain independent.

The Python update lowering only checks operands, resolves stage dtypes, encodes
the plan, and calls the ordinary update FFI. It does not select an assignment,
accumulation, or selected-scale executor or narrow coefficients for one.
Native preparation retains the resolved stage types in the existing prepared
state. Stage types participate in FFI specialization, but dynamic coefficient
values do not. Execution binds coefficients to concrete scalar operations;
it does not choose a separate copy, conversion, or scale executor.

Ordinary updates share one validation, initialization, and traversal flow.
Complete coverage does not initialize the output: kernels read the original
base when needed. Partial updates copy the base once to preserve unselected
storage. Copy, half-to-float conversion, scale, and weighted kernels run inside
this flow; uncommon valid type combinations retain the generic scalar kernel.
Alias ownership and AD scatter rules remain separate internal responsibilities,
but neither reintroduces a second ordinary update execution protocol.

Fresh maps and updates share scalar functors when their resolved scalar
expressions are equivalent. The shared affine leaf dispatcher owns layout
and CPU kernel matching, including half conversions and float32 rank-4 tiles.
Two-term updates use an explicit original-base write policy rather than treating
the output as initialized input. Differential scaling shares coefficient binding
and traversal, but retains ordinary multiplication at zero and one.

Binding extracts static factors and stage types from records before entering
array kernels. Bound scalar operations accept only the source value; records
inside kernels describe addresses and layouts, not arithmetic. Equivalent
static and dynamic single multiplies share the half and complex64 contiguous
and tiled kernels. Fresh maps retain explicit static zero/one multiplication;
updates short-circuit their effective source coefficient. Reduction mapping reuses these
bound operations without changing reduction scheduling or accumulation order.
Reduction task dispatch uses the same mapping binder as fresh maps; range
kernels receive its selected operation rather than an identity/fallback wrapper.
For updates that read the original base, binding constructs both the source
operation and the original-value write policy before invoking the leaf kernel.
Base-free updates retain their single direct-write branch.
Private opt-in leaf diagnostics test kernel selection independently of numerical
comparisons and custom-call counts; benchmark timing is not a unit-test gate.
Shared leaf kernels do not imply identical end-to-end timings: existing
plan-based parallel thresholds can still choose different worker counts for
static mappings and identity-mapped dynamic updates.

In its own dtype, a zero update coefficient omits its term entirely; a unit
coefficient uses the original term without multiplication or product conversion.
If both terms are absent, the selection receives zero. A sole unit
term writes its value with only the final storage conversion.
Static constants and dynamic scalar or batch-shaped coefficients follow the
same contract. This intentionally differs from evaluating every multiplication
on NaN or infinity inputs.

Scalar calculation and storage types are independent. For ordinary updates,
JAX promotion, including Python scalar weak typing, first resolves
`effective_factor = source_factor * record.scale` when a static scale exists;
otherwise the effective factor is the original source coefficient. This
coefficient product uses ordinary multiplication, without zero/one shortcuts
on either original factor. Python resolves the possible source/base products
and four additions of unit or multiplied terms using JAX's corresponding scalar
operations, retaining weak typing during inference. Native binding selects the
actual expression after coefficient classification. A sole term skips addition
and its promotion; two present terms use their actual addition type. Only the
complete scalar result is cast to the destination dtype. The output dtype stays
fixed, and runtime coefficient values do not create new cache entries.

A record with `scale=None` has no static multiply. Static scales retain their
type and weakness in plan keys. Updates compose coefficients and classify
zero/one in the coefficient dtype, before any source-product conversion.
They no longer evaluate `source_factor * (record.scale * source)`: rounding,
overflow, nonfinite results, and coefficient derivatives can therefore differ
from that old expression. Fresh map, Reduction, and Dot keep their existing
formulas, so a static fresh map and an overwrite update are not unconditionally
bitwise equivalent. In a fresh map any explicit static scalar, including `1`,
still performs an ordinary multiply. `materialize`
defaults to the source dtype; explicit output casts follow JAX's conversion
rules, including complex-to-real projection and integer conversion rather than
Julia exceptions, subject to the floating-point compatibility limits below.
Reductions retain their explicit accumulator dtype, applied after mapping.

Python resolves scalar types once; FFI carries per-stage types and independent
coefficient operands. Native scalar operators reuse the existing traversals and
cast on write without whole-storage preconversion. Supported source/storage
pairs remain bounded by the native kernels; unsupported pairs fail explicitly.

The scalar reference evaluates promotions, arithmetic stages, and final casts
separately. Finite floating-point calculations without range exceptions are
checked with dtype-appropriate absolute and relative tolerances, not bitwise
equality. NaN/Inf propagation, NaN payloads, arithmetic signed zeros, and
recovery after intermediate overflow are not guaranteed to agree across
kernels or backends. Real/complex products may use componentwise formulas;
kernels do not retry exceptional lanes merely to reproduce scalar propagation.
These allowances do not permit changing stage dtypes, reassociating the
specified coefficient product with source multiplication, or weakening integer
and boolean arithmetic, conversion safety, or the explicit zero/one shortcuts.
Pure copies preserve data directly, and unselected storage remains unchanged.
No extra pass sanitizes nonfinite values or normalizes arithmetic signed zeros.

Subnormal preservation or flushing to zero may differ between Tensor0 native
execution and a JAX backend, including during the final storage cast. Tensor0
does not promise bitwise equality with JAX in this range or add a separate
flushing pass to emulate a particular backend. This allowance does not relax
the required stage dtypes, conversion order, or explicit zero/one arithmetic
contract; it does not permit prematurely rounding coefficients to storage dtype.

Cross-backend numerical comparisons should use application-appropriate absolute
and relative tolerances. A nonzero subnormal is not universally equivalent to
zero: callers that depend on exact zero tests or subsequent amplification must
check their backend's behavior. Equivalent Tensor0 native execution paths are
checked for finite-value accuracy and the same structural contracts, not
identical exceptional-value propagation.

AD uses an explicit differential of the underlying weighted expression, not
the derivative of the zero/one dispatch. With effective coefficient `c`, the
selected tangent, followed by the differential of the final cast, is:

```text
c * dsource + dc * source + base_factor * dbase + dbase_factor * base
```

When a static scale exists, `dc` differentiates the coefficient product
`source_factor * record.scale`, including its resolved conversion stages.
There is no independent static source-mapping stage in ordinary updates.
The differential retains the unshortened expression's promotion stages,
independently of the forward branch's intermediate types. It is a stipulated
linear-update derivative, not a guarantee about bitwise changes at rounding
or short-circuit boundaries; finite differences near those boundaries are not
its reference definition.

Outside the selection it is `dbase`. Dynamic zero and
unit coefficients retain their coefficient derivatives. Integer and boolean
operands retain JAX's `float0` convention. No additional finite-gradient
guarantees are made for nonfinite inputs.

The differential executor uses ordinary multiplication even at zero and one.
A structurally absent coefficient tangent uses a single product, with no
synthetic zero term or addition; an explicit zero tangent still participates
in the two-product expression. The same rule applies to alias derivatives.
Forward specialization metadata stores only zero, one, or unknown after dtype
conversion; other constant values remain operands and share compilation caches.

Execution classifies each coefficient before selecting its actual operation
and associated dtype. Differential bindings always
retain multiplication. Generic product operators own their operand conversions;
unit forward terms retain their input types. Floating-point writes encode
the destination directly, while integer and boolean writes retain their checked
conversion policy. Only stages belonging to the executed forward expression
remain; the differential keeps all stages of its ordinary arithmetic.

`ConvertScalar<Dtype>` returns the native value
for the specified FFI dtype, distinguishing F16 from BF16 despite their shared
storage representation. Runtime `ConvertScalar(value, dtype)` reuses the typed
floating-point and boolean conversions and decodes the result into `ScalarValue`.
Both forms share integer bit conversion, including checked floating-to-integer
conversion and integer narrowing, without expanding every integer width in the
runtime dispatcher. Bound operators use the typed result directly; assignment performs
the actual store, while `WriteOp` retains ownership of update semantics.

`ConvertScalar<Result, Source>` converts known native values directly for
floating-point and complex dtypes, including half/bfloat decoding, encoding,
and complex-to-real projection. `ScalarCast` delegates to this overload.
Conversions involving integers retain the checked scalar implementation.

Floating-point/complex updates bind native-valued operators for a single source
product, or when all present term types equal the addition type. The common
computation dtype may differ from storage: the source term stays in that native
type until the base term has been combined, followed by the final storage cast.
Zero/one classification precedes coefficient conversion to the computation type;
differentials retain ordinary multiplication. Existing copy, SIMD, same-dtype,
and promoted-F64 scale paths remain preferred. Native arithmetic dispatch uses
the already resolved computation dtype, not a new promotion policy or FFI mode.
F32 source/storage updates also bind native-valued operators when one term is
F32, the other is F64, and addition is F64. Each product retains its own
computation type before the F64 addition and final F32 cast; a unit term retains
the original F32 value. The same binding supports ordinary differential
multiplication without applying forward coefficient shortcuts.
`ScalarValue` remains a binding container and the fallback for other distinct-stage
expressions and conversions outside this native-valued domain. Fresh mapping,
Reduction, and Dot share conversions but retain their formulas and scheduling.

The internal fused tangent kernel is only a differentiation mechanism, not an
alternative forward arithmetic mode. Likewise, selected-scale aliasing remains
a separate restricted ownership mechanism; passing the same array as both
update inputs does not authorize donation or in-place permutation.

Run the focused continuous/transposed-layout regression benchmark with
`uv run python -m benchmarks.standard.diagnostics.stride_update --size 512 --repeats 30`.
It reports synchronized execution time, native calls, and compiled output,
temporary, and alias bytes for conversion, updates, and a wide-coefficient JVP.
