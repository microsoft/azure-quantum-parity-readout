# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# All times in us
struct TimeTracesParameters
    trace_duration::Float64
    dt::Float64
    sweepPoints::OrderedDict{Symbol, AbstractRange{Float64}}
    σC::Float64
    τ_qpp::Float64
    periodic_parameters::Dict{Symbol, Float64}
    charge_noise::Dict{Symbol, Float64}

    function TimeTracesParameters(
        trace_duration, dt, sweep_parameters...
        ;
        σR, Vrf,
        τ_qpp,
        periodic_parameters=Dict{Symbol, Float64}(),
        charge_noise=Dict{Symbol, Float64}()
    )
        params = Pair{Symbol, StepRangeLen}[]
        for (k, v) in sweep_parameters
            if typeof(v) <: AbstractRange{Float64}
                push!(params, k=>v)
            else # Allows to define 1-element axis as number instead of range
                @assert typeof(v)<: Number
                push!(params, k=>v:v)
            end
        end
        σC = σR_to_σC(σR, Vrf, dt)
        sweep_definition = OrderedDict(params...)
        αs = Dict(k=>α for (k,α) in charge_noise if k in keys(sweep_definition))
        return new(trace_duration, dt, sweep_definition, σC, τ_qpp, periodic_parameters, αs)
    end
end

function get_sweep_shape(params::TimeTracesParameters)
    dims = length.(values(params.sweepPoints))
    mask = dims.>1
    return dims[mask]
end

function get_sweep_parameters(params::TimeTracesParameters)
    return [k for (k,v) in params.sweepPoints if length(v)>1]
end

function get_sweep_iterator(params::TimeTracesParameters)
    return Iterators.product(values(params.sweepPoints)...)
end

function number_of_traces(params::TimeTracesParameters)
    return prod(get_sweep_shape(params))
end

function number_of_sweep_parameters(params::TimeTracesParameters)
    return length(get_sweep_shape(params)) # Filter out 1-element parameters
end

function number_of_parameters(params::TimeTracesParameters)
    return length(params.sweepPoints) # All parameters including 1-element parameters
end

function experiment_duration(params::TimeTracesParameters)
    return number_of_traces(params)*params.trace_duration
end

function get_number_of_point_per_trace(params::TimeTracesParameters)
    return Int(params.trace_duration/params.dt)
end

function get_time_coords(params::TimeTracesParameters)
    return 0.0:params.dt:experiment_duration(params)-params.dt
end

function generate_long_one_over_f_trajectory(α, params::TimeTracesParameters)
    Sω(ω) = α^2/ω
    ωmax = π/params.dt
    Texp =experiment_duration(params)
    _, traj = generate_noise_trajectory(Sω, Texp, ωmax)
    N_points_per_trace = get_number_of_point_per_trace(params)
    N_traces = number_of_traces(params)
    ng_pts_all = reshape(traj, N_points_per_trace, N_traces)
    return ng_pts_all
end

function generate_telegraph(τ_qpp, params::TimeTracesParameters)
    t_range = get_time_coords(params)
    Texp = experiment_duration(params)
    tj = TelegraphTrajectory(Texp, τ_qpp)
    N_points_per_trace = get_number_of_point_per_trace(params)
    N_traces = number_of_traces(params)
    p_pts_all = reshape(tj.(t_range), N_points_per_trace, N_traces)
    return p_pts_all
end

function telegraph_signal_from_parity(Cqp, Cqm, p_pts_all)
    mult_p = @. (1+p_pts_all)/2
    mult_n = @. (1-p_pts_all)/2;
    return @. mult_p*Cqp + mult_n*Cqm
end

function _scale_factor(T)
    if T<:Complex
        return sqrt(2.0)
    end
    if T<:Real
        return 1.0
    end
end

# The variance of real(randn(Complex, ...)) is 1/sqrt(2) while it is 1 for real data type.
# We define σC as the noise per quadrature, so we rescale to get the same variance or
# the real whether Cqi is a real of complex array
function add_meaurement_noise!(Cqi, σC)
    T = eltype(Cqi)
    axpy!(σC*_scale_factor(T), randn(T, size(Cqi)...), Cqi)
    return Cqi
end

function σR_to_σC(σR, V_rf, dt)
    # Assuming dt in ns, V_rf in uV, σR in fF μV √μs
    return σR / (2 * V_rf * sqrt(dt/1_000))
end

function generate_CQ_Traces(params::TimeTracesParameters, itpp, itpn, noise_traces; periodic_parameters=Dict{Symbol, Float64}())
    N_points_per_trace = get_number_of_point_per_trace(params)
    N_traces = number_of_traces(params)

    @assert eltype(itpp) == eltype(itpn)
    T = eltype(itpp)
    Cqp = zeros(T, N_points_per_trace, N_traces)
    Cqm = zeros(T, N_points_per_trace, N_traces)
    N = length(params.sweepPoints)
    xs = [zeros(Float64, N_points_per_trace) for j in 1:N]
    noise_index_mapping = Dict(k=>findfirst(keys(params.sweepPoints) .== k) for k in keys(noise_traces))
    periods = [Inf for i in 1:N] # Default, no periodicity
    for (k,p) in periodic_parameters
        # Make sure the named parameters are coherent
        @assert k in keys(params.sweepPoints)
        periods[findfirst(keys(params.sweepPoints) .== k)] = p
    end
    for (j, pt) in enumerate(get_sweep_iterator(params))
        for (j,p) in enumerate(pt)
            fill!(xs[j], p % periods[j])
        end
        # Add 1/f noise on noisy indices
        for (k,i) in noise_index_mapping
            axpy!(1.0, @view(noise_traces[k][:,j]), xs[i])
        end

        Cqp[:,j] .= itpp.(xs...)
        Cqm[:,j] .= itpn.(xs...)
    end
    return Cqp, Cqm
end

function generate_time_traces(
    params::TimeTracesParameters, itps;
    fix_parity=nothing, return_xarray=false
)
    αs = params.charge_noise
    p_pts_all =  generate_telegraph(params.τ_qpp, params)
    if !isnothing(fix_parity)
        p_pts_all .= fix_parity
    end
    noise_traces = Dict(k=> generate_long_one_over_f_trajectory(α, params) for (k,α) in αs)

    traces =  generate_time_traces(
        params, noise_traces, p_pts_all, itps[1], itps[-1]
    )
    if return_xarray
        dt= params.dt
        ts = dt:dt:get_number_of_point_per_trace(params)*dt
        coords = OrderedDict(k=>v for (k,v) in params.sweepPoints if size(v,1)>1)
        dims = vcat([:time], get_sweep_parameters(params))
        args = Dict(:dims=>dims, :coords=>merge(OrderedDict(:time=>ts./1e6), coords))
        CQ = xr.DataArray(traces[:telegraph]; args...)
        CQ_even = xr.DataArray(traces[:even]; args...)
        CQ_odd = xr.DataArray(traces[:odd]; args...)
        return xr.Dataset(Dict(:CQ=>CQ, :CQ_even=>CQ_even, :CQ_odd=>CQ_odd))
    else
        return traces[:telegraph]
    end
end

function generate_time_traces(
    params::TimeTracesParameters,
    noise_traces::Dict{Symbol, T},
    telegraph_signal::AbstractArray,
    itp_even, itp_odd
) where {T<:AbstractArray{Float64}}
    # Check and update traces dimensions
    N_points_per_trace = get_number_of_point_per_trace(params)
    N_traces = number_of_traces(params)
    for k in keys(noise_traces)
        @assert length(noise_traces[k]) == N_points_per_trace*N_traces
        noise_traces[k] = reshape(noise_traces[k], N_points_per_trace, N_traces)
    end
    @assert length(telegraph_signal) == N_points_per_trace*N_traces
    telegraph_signal = reshape(telegraph_signal, N_points_per_trace, N_traces)

    # Generate noisy traces
    Cq_even, Cq_odd = generate_CQ_Traces(
        params, itp_even, itp_odd, noise_traces;
        periodic_parameters=params.periodic_parameters
    )
    Cqi = telegraph_signal_from_parity(Cq_even, Cq_odd, telegraph_signal)
    add_meaurement_noise!(Cqi, params.σC);
    add_meaurement_noise!(Cq_even, params.σC);
    add_meaurement_noise!(Cq_odd, params.σC);

    N_points_per_trace = get_number_of_point_per_trace(params)
    dims = get_sweep_shape(params)
    return Dict(
        :telegraph => reshape(Cqi, N_points_per_trace, dims...),
        :even => reshape(Cq_even, N_points_per_trace, dims...),
        :odd => reshape(Cq_odd, N_points_per_trace, dims...)
    )
end

function _permutation_indices(ds, axes)
    dims = collect(Symbol.(ds.squeeze(drop=true).dims))
    dims = dims[dims.!=:parity]
    @assert sort(axes) == sort(dims)
    return [argmax(dims.==a) for a in axes]
end

function build_interpolators_from_xarray(ds, axes)
    perm = _permutation_indices(ds, axes)
    data_even = permutedims(ds.sel(parity=1).squeeze(drop=true).data, perm)
    data_odd = permutedims(ds.sel(parity=-1).squeeze(drop=true).data, perm)
    coords = tuple([ds[a].data for a in axes]...)
    return Dict(
        +1=>linear_interpolation(coords, data_even, extrapolation_bc=Interpolations.Flat()),
        -1=>linear_interpolation(coords, data_odd, extrapolation_bc=Interpolations.Flat())
    )
end
