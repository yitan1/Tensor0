using JSON
using LinearAlgebra
using TensorKit
using TensorOperations: @tensor


const SCHEMA_VERSION = 1
const DATA_MODULUS = 65_521
const TREE_MULTIPLIER = 257
const COORDINATE_MULTIPLIER = 1_009
const VALUE_MODULUS = 1_021
const VALUE_CENTER = 510
const VALUE_DIVISOR = 2_048
const BENCHMARK_THREADS = parse(Int, get(ENV, "CROSS_BACKEND_THREADS", "1"))

const COMPILED_TOPOLOGIES = Dict(
    "mpo" => (
        operands = ["FL", "A", "M", "A", "FR"],
        labels = [
            [4, 2, 1],
            [1, 3, 6],
            [2, 5, 3, 7],
            [4, 5, 8],
            [6, 7, 8],
        ],
        conjugate = [false, false, false, true, false],
        order = collect(1:8),
    ),
    "pepo" => (
        operands = ["FL", "FU", "A", "P", "A", "FR", "FD"],
        labels = [
            [18, 7, 4, 2, 1],
            [1, 3, 6, 9, 10],
            [2, 17, 5, 3, 11],
            [4, 16, 8, 5, 6, 12],
            [7, 15, 8, 9, 13],
            [10, 11, 12, 13, 14],
            [14, 15, 16, 17, 18],
        ],
        conjugate = [false, false, false, false, true, false, false],
        order = collect(1:18),
    ),
    "mera" => (
        operands = ["h", "u", "u", "u", "w", "u", "w", "w", "w", "rho", "w", "w"],
        labels = [
            [9, 3, 4, 5, 1, 2],
            [1, 2, 7, 12],
            [3, 4, 11, 13],
            [8, 5, 15, 6],
            [6, 7, 19],
            [8, 9, 17, 10],
            [10, 11, 22],
            [12, 14, 20],
            [13, 14, 23],
            [18, 19, 20, 21, 22, 23],
            [16, 15, 18],
            [16, 17, 21],
        ],
        conjugate = [
            false,
            false,
            true,
            false,
            false,
            true,
            true,
            false,
            true,
            false,
            false,
            true,
        ],
        order = [1, 2, 3, 4, 6, 5, 7, 10, 8, 9, 11, 14, 20, 23, 12, 13, 19, 22, 15, 18, 16, 17, 21],
    ),
)

BENCHMARK_THREADS >= 1 || error("CROSS_BACKEND_THREADS must be positive")
Threads.nthreads() == BENCHMARK_THREADS ||
    error("JULIA_NUM_THREADS does not match CROSS_BACKEND_THREADS")
LinearAlgebra.BLAS.set_num_threads(BENCHMARK_THREADS)


function elementary_space(specification, sector)
    sectors = [(Int(pair[1]), Int(pair[2])) for pair in specification["sectors"]]
    if sector == "trivial"
        return ComplexSpace(Int(specification["dimension"]))
    elseif sector == "z2"
        return Z2Space((label => degeneracy for (label, degeneracy) in sectors)...)
    elseif sector == "u1"
        return U1Space((label => degeneracy for (label, degeneracy) in sectors)...)
    elseif sector == "su2"
        return SU2Space((label // 2 => degeneracy for (label, degeneracy) in sectors)...)
    end
    throw(ArgumentError("unsupported sector: $sector"))
end


function scalar_type(name)
    name == "float64" && return Float64
    name == "complex128" && return ComplexF64
    throw(ArgumentError("unsupported dtype: $name"))
end


mix_data_seed(seed, value) = mod(seed * TREE_MULTIPLIER + value + 1, DATA_MODULUS)
su2_label(sector) = Int(2 * sector.j)


function mix_fusion_tree(seed, marker, tree)
    seed = mix_data_seed(seed, marker)
    seed = mix_data_seed(seed, length(tree.uncoupled))
    for sector in tree.uncoupled
        seed = mix_data_seed(seed, su2_label(sector))
    end
    seed = mix_data_seed(seed, su2_label(tree.coupled))
    for isdual in tree.isdual
        seed = mix_data_seed(seed, Int(isdual))
    end
    seed = mix_data_seed(seed, length(tree.innerlines))
    for sector in tree.innerlines
        seed = mix_data_seed(seed, su2_label(sector))
    end
    return seed
end


function fusion_tree_seed(tensor_name, pair)
    bytes = codeunits(tensor_name)
    seed = mix_data_seed(17, length(bytes))
    for value in bytes
        seed = mix_data_seed(seed, Int(value))
    end
    seed = mix_fusion_tree(seed, 11, pair[1])
    return mix_fusion_tree(seed, 29, pair[2])
end


function fill_fusion_tree!(target, tensor_name, pair, coefficient)
    seed = fusion_tree_seed(tensor_name, pair)
    for index in CartesianIndices(target)
        code = seed
        for (axis, coordinate) in enumerate(Tuple(index))
            code = mod(
                code + axis * COORDINATE_MULTIPLIER * coordinate,
                DATA_MODULUS,
            )
        end
        centered = mod(code, VALUE_MODULUS) - VALUE_CENTER
        target[index] = coefficient * (1 + centered / VALUE_DIVISOR)
    end
    return target
end


function fusion_tree_tensor(T, product_space, tensor_name, coefficient)
    value = zeros(T, product_space)
    for pair in fusiontrees(value)
        fill_fusion_tree!(subblock(value, pair), tensor_name, pair, coefficient)
    end
    return value
end


function validated_topology_name(topology)
    name = String(topology["name"])
    haskey(COMPILED_TOPOLOGIES, name) ||
        throw(ArgumentError("no compiled @tensor kernel for topology: $name"))
    expected = COMPILED_TOPOLOGIES[name]
    actual = (
        operands = String.(topology["operands"]),
        labels = [Int.(labels) for labels in topology["labels"]],
        conjugate = Bool.(topology["conjugate"]),
        order = Int.(topology["order"]),
    )
    for field in keys(expected)
        getproperty(actual, field) == getproperty(expected, field) ||
            throw(
                ArgumentError(
                    "topology $name field $field does not match its compiled @tensor kernel",
                ),
            )
    end
    return name
end


function contract_mpo(FL, A, M, FR)
    @tensor order=(1, 2, 3, 4, 5, 6, 7, 8) C =
        FL[4, 2, 1] * A[1, 3, 6] * M[2, 5, 3, 7] * conj(A[4, 5, 8]) *
        FR[6, 7, 8]
    return C
end


function contract_pepo(FL, FU, A, P, FR, FD)
    @tensor order=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18) C =
        FL[18, 7, 4, 2, 1] * FU[1, 3, 6, 9, 10] *
        A[2, 17, 5, 3, 11] * P[4, 16, 8, 5, 6, 12] *
        conj(A[7, 15, 8, 9, 13]) * FR[10, 11, 12, 13, 14] *
        FD[14, 15, 16, 17, 18]
    return C
end


function contract_mera(h, u, w, rho)
    @tensor order=(1, 2, 3, 4, 6, 5, 7, 10, 8, 9, 11, 14, 20, 23, 12, 13, 19, 22, 15, 18, 16, 17, 21) C =
        h[9, 3, 4, 5, 1, 2] * u[1, 2, 7, 12] *
        conj(u[3, 4, 11, 13]) * u[8, 5, 15, 6] * w[6, 7, 19] *
        conj(u[8, 9, 17, 10]) * conj(w[10, 11, 22]) * w[12, 14, 20] *
        conj(w[13, 14, 23]) * rho[18, 19, 20, 21, 22, 23] *
        w[16, 15, 18] * conj(w[16, 17, 21])
    return C
end


prepare_mpo(FL, A, M, FR) = () -> contract_mpo(FL, A, M, FR)
prepare_pepo(FL, FU, A, P, FR, FD) = () -> contract_pepo(FL, FU, A, P, FR, FD)
prepare_mera(h, u, w, rho) = () -> contract_mera(h, u, w, rho)


function prepared_operation(workload)
    sector = workload["sector"]
    data_policy = workload["data"]
    expected_data_policy = sector == "su2" ? "fusion_tree_v1" : "uniform_v1"
    data_policy == expected_data_policy ||
        throw(
            ArgumentError(
                "data policy must be fusion_tree_v1 exactly for SU2",
            ),
        )
    spaces = Dict(
        specification["name"] => elementary_space(specification, sector)
        for specification in workload["spaces"]
    )
    T = scalar_type(workload["dtype"])
    tensors = Dict{String, Any}()
    for specification in workload["topology"]["tensors"]
        legs = [
            isdual ? spaces[name]' : spaces[name]
            for (name, isdual) in zip(specification["spaces"], specification["duals"])
        ]
        product_space = reduce(⊗, legs)
        coefficient = specification["scale"]
        workload["dtype"] == "complex128" && (coefficient *= 1 + 0.125im)
        value = if data_policy == "fusion_tree_v1"
            fusion_tree_tensor(T, product_space, specification["name"], coefficient)
        else
            uniform = ones(T, product_space)
            fill!(uniform, coefficient)
            uniform
        end
        tensors[specification["name"]] = value
    end

    topology = workload["topology"]
    name = validated_topology_name(topology)
    if name == "mpo"
        return prepare_mpo(tensors["FL"], tensors["A"], tensors["M"], tensors["FR"])
    elseif name == "pepo"
        return prepare_pepo(
            tensors["FL"],
            tensors["FU"],
            tensors["A"],
            tensors["P"],
            tensors["FR"],
            tensors["FD"],
        )
    end
    return prepare_mera(tensors["h"], tensors["u"], tensors["w"], tensors["rho"])
end


Base.@noinline sink(value) = value


function measure_workload(workload, warmup, repeat)
    operation = prepared_operation(workload)

    result = sink(operation())
    for _ in 1:warmup
        result = sink(operation())
    end

    samples_ms = Float64[]
    for _ in 1:repeat
        start = time_ns()
        result = sink(operation())
        stop = time_ns()
        push!(samples_ms, (stop - start) / 1.0e6)
    end
    scalar = ComplexF64(result)
    return Dict(
        "workload_id" => workload["id"],
        "workload_hash" => workload["workload_hash"],
        "status" => "ok",
        "samples_ms" => samples_ms,
        "output" => Dict(
            "real" => real(scalar),
            "imag" => imag(scalar),
        ),
    )
end


function backend_metadata()
    return Dict(
        "name" => "tensorkit",
        "version" => string(pkgversion(TensorKit)),
        "runtime" => "Julia $(VERSION)",
        "threads" => Threads.nthreads(),
        "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "thread_settings" => Dict(
            name => get(ENV, name, "unset")
            for name in (
                "JULIA_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CROSS_BACKEND_THREADS",
            )
        ),
    )
end


function execute(request)
    get(request, "schema_version", nothing) == SCHEMA_VERSION ||
        throw(ArgumentError("unsupported schema_version"))
    get(request, "message_type", nothing) == "benchmark_request" ||
        throw(ArgumentError("invalid message_type"))
    measurement = request["measurement"]
    get(measurement, "kind", nothing) == "steady_state" ||
        throw(ArgumentError("unsupported measurement kind"))
    warmup = Int(measurement["warmup"])
    repeat = Int(measurement["repeat"])
    results = Any[]
    for workload in request["workloads"]
        try
            push!(results, measure_workload(workload, warmup, repeat))
        catch error
            push!(
                results,
                Dict(
                    "workload_id" => workload["id"],
                    "workload_hash" => workload["workload_hash"],
                    "status" => "error",
                    "error" => "$(typeof(error)): $(sprint(showerror, error))",
                ),
            )
        end
    end
    return Dict(
        "schema_version" => SCHEMA_VERSION,
        "message_type" => "benchmark_response",
        "run_id" => request["run_id"],
        "round" => request["round"],
        "backend" => backend_metadata(),
        "results" => results,
    )
end


length(ARGS) == 1 || error("usage: julia runner.jl REQUEST.json")
request = JSON.parsefile(only(ARGS))
JSON.print(stdout, execute(request))
println()
