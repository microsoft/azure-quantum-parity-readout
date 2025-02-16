---
jupyter:
  jupytext:
    custom_cell_magics: kql
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: julia
    language: julia
    name: julia
---

```julia
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
```

```julia
using Pkg
Pkg.activate("..")
```

```julia
# Ensure PyCall is using the local Python environment
ENV["PYTHON"] = abspath(joinpath(dirname(@__FILE__), "../../.venv/bin/python"))
Pkg.build("PyCall")
```

```julia
using ParityReadoutSimulator, PyCall, PyPlot, Random, ProgressMeter
xr = ParityReadoutSimulator.xr
@pyinclude joinpath(dirname(pathof(ParityReadoutSimulator)), "plotting.py")
```

```julia
data_path = ParityReadoutSimulator.get_simulated_data_path()
```

# Generating thermometry reference data

```julia
eunits = Units.μeV_to_GHz
fixed_parameters = Dict{Symbol, Any}(
    :EC=>50.0*eunits,
    :lever_arm=>0.5,
    :fd=>0.5,
    :ωc => 100.0 * Units.μeV_to_GHz,
    :δt_factor=>0.05,
    :n_freq_lindblad=>0,
    :n_freq=>5,
);
```

```julia
# Inner loop parameters
sweeped_parameters = Dict(
    :ng=>0.0:0.01:1.0,
    :A=>[0.001],
    :t0=>collect(1:15).* Units.μeV_to_GHz
);

# We keep the temperature and charge noise parameters out of the main loop to allow for caching of the bath correlation function
γs = vcat(0.1:0.1:1.0, 5.:5.:20.,) # GHz
Ts = collect(25.:10.:750.); # mK
```

```julia
# Optional: Distribute computation over multiple workers for faster compute.
using Distributed
number_of_workers = Sys.CPU_THREADS
if nworkers() == 1 && number_of_workers > 1
    addprocs(number_of_workers, exeflags=["--threads=1 --heap-size-hint=2.5G"])
    @everywhere using ParityReadoutSimulator
end;
```

```julia
# Pre-compute the required bath correlation functions
bath_params_to_indices = Dict(p=>i for (i, p) in enumerate(Iterators.product(γs, Ts)))
bath_indices_to_params = Dict(i=>p for (p, i) in bath_params_to_indices)
Np = length(bath_params_to_indices);

wp = WorkerPool(workers())
@time bath_caches = pmap(wp,1:Np) do i
    ωc::Float64 = fixed_parameters[:ωc]
    γ, T = bath_indices_to_params[i]
    params = Dict{Symbol, Any}(:ωc=>ωc, :temperature=>T*Units.mK_to_GHz, :γng=>γ)
    cache_bath_correlation_function(params)
    Dict(:bcfs=>params[:bcfs], :bath_parameters=>params[:bath_parameters])
end;
```

```julia
dsets = PyObject[]
@time @showprogress for i in 1:Np
    γ, T = bath_indices_to_params[i]
    fixed_parameters[:temperature] = T*Units.mK_to_GHz
    fixed_parameters[:γng] = γ
    fixed_parameters = merge(fixed_parameters, bath_caches[i])
    results = sweep_parameters(
        evaluate_CQ_DQD_model,
        fixed_parameters,
        sweeped_parameters
        ;
        verbose=false, drop_keys=[:bcfs], pmap_kwargs=Dict()
    )
    ds0 = convert_DQD_results_to_xarray(results);
    push!(dsets, ds0)
end
ds = xr.merge([ds for ds in dsets]);
```

```julia
overwrite_data = false
if overwrite_data # We use h5netcdf to allow for complex numbers
    file_name = joinpath(data_path,"thermometry_reference_simulated_dataset.h5")
    ds.to_netcdf(file_name, engine="h5netcdf", invalid_netcdf=true)
end
```

```julia
ds.CQ.real.sel(temperature=55.0, γng=0.1,A=0.001).squeeze(drop=true).plot.line(x=:ng,);
```

```julia
ds.CQ.real.sel(t0=5.0,γng=0.1,A=0.001).squeeze(drop=true).plot();
```
