# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


using Distributed
using DataFrames
import Pandas as pdjl

function sweep_parameters(fct,params, sweep_params::AbstractDict; verbose=true, drop_keys=[], pmap_kwargs=Dict())
    ks = collect(keys(sweep_params))
    vals = collect(values(sweep_params))
    @assert length(ks)==length(vals)
    iters = collect(Iterators.product(vals...))
    N = prod(size(iters))
    iters = reshape(iters,N)
    wp = WorkerPool(workers())
    if !(:batch_size in keys(pmap_kwargs))
        pmap_kwargs[:batch_size] = max(floor(Int, N/nworkers()/10), 1)
    end
    dfrows = pmap(wp, 1:N; pmap_kwargs...) do i
        v = iters[i]
        if verbose
            @show v
        end
        par = copy(params)
        for (k,vi) in zip(ks, v)
            par[k]= vi
        end
        res = fct(;par...)
        for dk in drop_keys
            pop!(par, dk)
        end
        merge(res, par)
    end
    return DataFrame(dfrows)
end

function to_xarray(df::DataFrame, index_keys, vars)
    return pdjl.DataFrame(df[:,vcat(index_keys, vars)]).pyo.set_index(index_keys).to_xarray()
end


function convert_CQ_results_to_xarray(results, GHz_to_μeV_coords, unitless_coords, GHz_to_mK_coords=[:temperature])
    all_coords = Symbol.(names(results))
    GHz_to_μeV_coords = [k for k in GHz_to_μeV_coords if k in all_coords]
    sort!(results, vcat(GHz_to_μeV_coords, unitless_coords, GHz_to_mK_coords));
    ds = to_xarray(results, vcat(GHz_to_μeV_coords, unitless_coords, GHz_to_mK_coords), [:CQ]);
    ds = ds.assign_coords(Dict(k=>ds[k]/Units.μeV_to_GHz for k in GHz_to_μeV_coords)); # Rescale energy coords to ueV instead of GHz
    ds = ds.assign_coords(Dict(k=>ds[k]/Units.mK_to_GHz for k in GHz_to_mK_coords));
    return ds
end

function convert_TQD_results_to_xarray(results)
    GHz_coords = [:tm1, :tm2, :t12, :t23, :EC1, :EC2, :EC3, :E_M, :E_0, :u1, :u2]
    unitless_coords = [:ng1, :ng2, :ng3, :phi, :parity, :A, :fd, :lever_arm, :γng, :n_freq, :n_freq_lindblad]
    return convert_CQ_results_to_xarray(results, GHz_coords, unitless_coords)
end

function convert_DQD_results_to_xarray(results)
    GHz_coords = [:t0, :EC]
    unitless_coords = [:ng, :A, :fd, :lever_arm, :γng, :n_freq, :n_freq_lindblad]
    return convert_CQ_results_to_xarray(results, GHz_coords, unitless_coords)
end

function generate_default_parameters()
    eunits = Units.μeV_to_GHz
    fixed_parameters = Dict{Symbol, Any}(
        # Base Hamiltonian
        :EC1 => 140.0*eunits,
        :EC2 => 45.0*eunits,
        :EC3 => 100.0*eunits,
        :E_M=>0.0*eunits,
        :phi=>0.5,
        :t12=>8*eunits,
        :tm1=>6*eunits,
        :tm2=>6*eunits,
        :t23=>8*eunits,
        # Drive
        :A=>5.0/90, # Drive amplitude in ng units.
        :fd=>0.5, # GHz
        :lever_arm=>0.45, # unitless
        # Bath
        :temperature=>50.0*Units.mK_to_GHz,
        :ωc => 50.0*eunits,
        :γng=>1.0, # GHz
        :δt_factor=>0.05,
        # Floquet steady-state soler
        :n_freq=>5,
        :n_freq_lindblad=>0,
    )
end

function generate_default_charge_noise_parameters(ds::PyObject)
    EC = [ds["EC$i"].data[1] for i in 1:3]
    return generate_default_charge_noise_parameters(EC)
end

function generate_default_charge_noise_parameters(fixed_parameters::Dict)
    EC = [fixed_parameters[Symbol("EC$i")]/Units.μeV_to_GHz for i in 1:3]
    return generate_default_charge_noise_parameters(EC)
end

function generate_default_charge_noise_parameters(EC::AbstractVector)
    out = Dict{Symbol,Float64}()
    for i in 1:3
        out[Symbol("ng$i")] = round(1.35*sqrt(EC[i]/120)/(2*EC[i]),digits=6)
    end
    return sort(out)
end

function generate_default_ng_grid()
    return sort(vcat(0.0:0.1:1.0, 0.35:0.1:0.65, 0.425:0.05:0.575))
end
