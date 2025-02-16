---
jupyter:
  jupytext:
    custom_cell_magics: kql
    formats: md,ipynb
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
using ParityReadoutSimulator, PyCall, PyPlot, Random
xr = ParityReadoutSimulator.xr
@pyinclude joinpath(dirname(pathof(ParityReadoutSimulator)), "plotting.py")
```

## Loading pre-computed data

```julia
data_path = ParityReadoutSimulator.get_simulated_data_path()
file_name = joinpath(data_path,"FigS3_data.h5")
```

```julia
# Alternatively, see end of file to generate new data locally
ds = xr.load_dataset(file_name,engine="h5netcdf");
```

## Generating noise trajectories and plotting

```julia
ng1_value = 0.35
ng3_value = 0.4

time_trace_params = TimeTracesParameters(
    45*Units.ms_to_ns,
    90.0*Units.μs_to_ns,
    :ng1=>ng1_value:ng1_value,
    :ng2=>0.0:0.05:1.0,
    :ng3=>ng3_value:ng3_value,
    ;
    charge_noise = generate_default_charge_noise_parameters(ds),
    σR=20.0, # in noise units
    Vrf=10.0, # uV
    τ_qpp=1.0*Units.ms_to_ns,
)
itps = build_interpolators_from_xarray(ds.CQ, [:ng1, :ng2, :ng3]);
```

```julia
let
    Random.seed!(10) # Optional: Set seed to reproduce figure S3 of manuscript.
    # Generate noise trajectories
    telegraph_trajectory = generate_telegraph(time_trace_params.τ_qpp, time_trace_params)
    charge_noise_trajectories = Dict(
        k => generate_long_one_over_f_trajectory(α, time_trace_params) for (k,α) in time_trace_params.charge_noise
    )
    ng2_v = time_trace_params.sweepPoints[:ng2]
    ms = Units.ms_to_ns

    # Start figure
    fig = figure(figsize = (1.5*3.5, 4), layout="constrained")
    subfigs = fig.subfigures(nrows=2, height_ratios = [0.8, 1])
    ax = [subfigs[1].gca()]
    more_ax = subfigs[2].subplots(ncols=2)
    ax = vcat(ax, more_ax)

    # Panel a
    noiseless_ng2 = zeros(size(charge_noise_trajectories[:ng2])...)
    for (k,ng2) in enumerate(ng2_v)
        noiseless_ng2[:,k] .= ng2
    end
    ng2 = reshape(noiseless_ng2 .+ charge_noise_trajectories[:ng2], :)
    time = get_time_coords(time_trace_params) / (1000*ms)

    ax[1].plot(time, ng1_value .+ reshape(charge_noise_trajectories[:ng1], :), label=L"N_{g1}(t)", linewidth=0.3)
    ax[1].plot(time, ones(length(time)) .* ng1_value, label=L"\overline{N_{g1}}(t)", linewidth=1.0)

    ax[1].plot(time, ng2, label=L"N_{g2}(t)", linewidth=0.3)
    ax[1].plot(time, reshape(noiseless_ng2, :), label=L"\overline{N_{g2}}(t)", linewidth=1.0)

    ax[1].plot(time, ng3_value .+ reshape(charge_noise_trajectories[:ng3], :), label=L"N_{g3}(t)", linewidth=0.3)
    ax[1].plot(time, ones(length(time)) .* ng3_value, label=L"\overline{N_{g3}}(t)", linewidth=1.0)

    # Panel b
    traces = generate_time_traces(
        time_trace_params, charge_noise_trajectories, telegraph_trajectory, itps[+1], itps[-1],
    )[:telegraph]
    ax[2].scatter(time, real.(reshape(traces, :)), s=0.2, marker="o", alpha=0.1, label=L"C_Q(t)", color="royalblue")

    # Panel c
    avg_time = range(0, experiment_duration(time_trace_params), length=number_of_traces(time_trace_params)) / (1000ms)
    kurt = [kurtosis(slice) for slice in eachslice(
        real.(traces), dims=(2)
    )]

    ax[3].plot(avg_time, kurt, "x-")

    ax[1].set_yticks(0.0:0.5:1.0)
    ax[2].set_yticks(0.0:0.5:1.5)
    ax[3].set_yticks(-1.5:0.5:0.0)

    ax[1].set_xlabel("Time [s]")
    ax[2].set_xlabel("Time [s]")
    ax[3].set_xlabel("Time [s]")

    ax[1].legend(fontsize=8, ncols=3)

    ax[1].set_ylabel(L"N_{g}")
    ax[2].set_ylabel(L"$C_Q$ [fF]")
    ax[3].set_ylabel(L"$K(C_Q)$")

    py"add_subfig_label"(ax[1], "a")
    py"add_subfig_label"(ax[2], "b")
    py"add_subfig_label"(ax[3], "c")
    plt.savefig("FigS3.pdf")
end
```

## Generating simulation dataset (optional)

```julia
# Energy parameters are in angular GHz units. Divide by Units.μeV_to_GHz to obtain parameters in μeV
fixed_parameters = generate_default_parameters()
# Override coupling parameter defaults
eunits = Units.μeV_to_GHz
fixed_parameters = merge(
    fixed_parameters,
    Dict(
        :t12=>8*eunits,
        :tm1=>6*eunits,
        :tm2=>6*eunits,
        :t23=>8*eunits,
    )
)

# Calculate the bath correlation function based on the parameters of the spectral function
cache_bath_correlation_function(fixed_parameters);
```

```julia
# Parameters to be swept.
# Default ng grid that is coarser away from charge resonance and finer near ng ~ 0.5
ng_adaptive_grid = generate_default_ng_grid()
sweeped_parameters = Dict(
    :ng1=>ng_adaptive_grid,
    :ng2=>ng_adaptive_grid,
    :ng3=>ng_adaptive_grid,
    :phi=>0.5, # Magnetic flux with largest signal
    :parity=>[1,-1],
);
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
# Parallelize computation over available workers and store results in a DataFrame
@time results = sweep_parameters(
    evaluate_CQ_TQD_model,
    fixed_parameters,
    sweeped_parameters
    ;
    verbose=false,
    drop_keys=[:bcfs] # Drop the pre-computed bath correlation function from the results DataFrame
)

# To simplify data analysis and sharing, we convert the results to an xarray and convert back parameters from GHz to μeV units.
ds = convert_TQD_results_to_xarray(results);
```

```julia
overwrite_data = false
if overwrite_data # We use h5netcdf to allow for complex numbers
    ds.to_netcdf(file_name, engine="h5netcdf", invalid_netcdf=true)
end
```

```julia

```
