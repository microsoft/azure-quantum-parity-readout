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
using ParityReadoutSimulator, PyCall, PyPlot, Random
xr = ParityReadoutSimulator.xr
@pyinclude joinpath(dirname(pathof(ParityReadoutSimulator)), "plotting.py")
```

# Fig S8: QD-MZM


## Loading pre-computed data

```julia
data_path = ParityReadoutSimulator.get_simulated_data_path()
file_name = joinpath(data_path,"FigS8_data.h5")
fn_FigS11_QDL = joinpath(data_path,"FigS11_QDL.h5")
fn_FigS11_QDR = joinpath(data_path,"FigS11_QDR.h5")
```

```julia
# Alternatively, see end of file to generate new data locally
ds = xr.load_dataset(file_name,engine="h5netcdf");
```

## Generating noise trajectories and plotting

```julia
time_trace_params = TimeTracesParameters(
    45*Units.ms_to_ns,
    90.0*Units.μs_to_ns,
    # We want the measured QD (QD2) to be the fast axis of the sweep
    :ng2=>0.0:0.01:1.0,
    :ng1=>0.0:0.025:1.0,
    :ng3=>0.0:0.0,
    ;
    charge_noise=generate_default_charge_noise_parameters(ds),
    σR=20.0, # in noise units
    Vrf=10.0, # uV
    τ_qpp=1.0*Units.ms_to_ns,
)
Random.seed!(42)
data = Dict{Float64,Matrix{ComplexF64}}()
for tm1 in [2.,6.]
    itps = build_interpolators_from_xarray(ds.sel(tm1=tm1,tm2=6.0).squeeze(drop=true).CQ, [:ng2, :ng1, :ng3]);
    traces = generate_time_traces(time_trace_params, itps; return_xarray=false)
    # We only need the time-average of the traces
    data[tm1] = dropdims(mean(traces, dims=1), dims=1)
end
```

```julia
let
    fig, ax = subplots(nrows=2, ncols=2, figsize=(1.5*3.5, 4), sharex=true, sharey=true)

    vmax_r = maximum([maximum(real, Z) for Z in values(data)])
    vmax_i = maximum([maximum(imag, Z) for Z in values(data)])

    for (fignum,tL) in enumerate([2., 6.])
        img = ax[fignum, 1].imshow(real.(data[tL]),
            origin="lower", aspect="auto",
            extent=[0., 1., 0., 1.],
            cmap="Blues",
            vmin=0., vmax=vmax_r
        )
        colorbar(img, ax=ax[fignum, 1], label=L"$\mathrm{Re}\,C_Q$ [fF]")
        img = ax[fignum, 2].imshow(imag.(data[tL]),
            origin="lower", aspect="auto",
            extent=[0., 1., 0., 1.],
            cmap="Purples",
            vmin=0., vmax=0.07
        )
        colorbar(img, ax=ax[fignum, 2], label=L"$\mathrm{Im}\,C_Q$ [fF]")
    end

    ax[1,1].set_ylabel(L"N_{g2}")
    ax[2,1].set_ylabel(L"N_{g2}")
    ax[2,1].set_xlabel(L"N_{g1}")
    ax[2,2].set_xlabel(L"N_{g1}")

    py"add_subfig_label"(ax[1,1], "a")
    py"add_subfig_label"(ax[1,2], "b")
    py"add_subfig_label"(ax[2,1], "c")
    py"add_subfig_label"(ax[2,2], "d")

    fig.tight_layout()
    plt.savefig("figS8.pdf")
end
```

## Generating QD-MZM data for Fig S11

```julia
Random.seed!(42)

time_trace_params_L = TimeTracesParameters(
    45*Units.ms_to_ns,
    90.0*Units.μs_to_ns,
    # We want the measured QD (QD2) to be the fast axis of the sweep
    :ng2=>0.0:0.01:1.0,
    :ng1=>0.0:0.025:1.0,
    :ng3=>0.0:0.0,
    ;
    charge_noise=generate_default_charge_noise_parameters(ds),
    σR=20.0, # in noise units
    Vrf=10.0, # uV
    τ_qpp=1.0*Units.ms_to_ns,
)

time_trace_params_R = TimeTracesParameters(
    45*Units.ms_to_ns,
    90.0*Units.μs_to_ns,
    # We want the measured QD (QD2) to be the fast axis of the sweep
    :ng2=>0.0:0.01:1.0,
    :ng1=>0.0:0.0,
    :ng3=>0.0:0.025:1.0,
    ;
    charge_noise=generate_default_charge_noise_parameters(ds),
    σR=20.0, # in noise units
    Vrf=10.0, # uV
    τ_qpp=1.0*Units.ms_to_ns,
)

tr_params = [time_trace_params_L,time_trace_params_R]


data = Dict{Integer,Any}()
tm1 = 6.0;
tm2= 4.0;
itps_L= build_interpolators_from_xarray(ds.sel(tm1=tm1,tm2=tm2).squeeze(drop=true).CQ, [:ng2, :ng1, :ng3]);
itps_R = build_interpolators_from_xarray(ds.sel(tm1=tm1,tm2=tm2).squeeze(drop=true).CQ, [:ng2, :ng1, :ng3]);
itps_list = [itps_L,itps_R]

for i in 1:2
    traces = generate_time_traces(tr_params[i], itps_list[i]; return_xarray=true)
    # We only need the time-average of the traces
    data[i] = traces.CQ.mean(dim="time").expand_dims(Dict(:tm1=>[tm1],:tm2=>[tm2],:E_M=>ds.E_M))
end

```

```julia
let
    fig, ax = subplots(nrows=2, ncols=2, figsize=(1.5*3.5, 4), sharex=false, sharey=true)

    vmax = 1
    vmaxi = 0.07

    for i in 1:2
        data[i].real.plot(y=:ng2,cmap="Blues",vmin=0,vmax=vmax,ax=ax[i,1])
        data[i].imag.plot(y=:ng2,cmap="Purples",vmin=0,vmax=vmaxi,ax=ax[i,2])
    end


    py"add_subfig_label"(ax[1,1], "a")
    py"add_subfig_label"(ax[1,2], "b")
    py"add_subfig_label"(ax[2,1], "c")
    py"add_subfig_label"(ax[2,2], "d")

    fig.tight_layout()
end
```

```julia
overwrite_data = false
if overwrite_data
    data[1].to_netcdf(fn_FigS11_QDL,engine="h5netcdf", invalid_netcdf=true)
    data[2].to_netcdf(fn_FigS11_QDR,engine="h5netcdf", invalid_netcdf=true)
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
        :t12=>12*eunits,
        :t23=>12*eunits,
        :phi=>0.0,
    )
)

# Calculate the bath correlation function based on the parameters of the spectral function
cache_bath_correlation_function(fixed_parameters);
```

```julia
ng_adaptive_grid = generate_default_ng_grid()
sweeped_parameters = Dict(
    :ng1=>ng_adaptive_grid,
    :ng2=>ng_adaptive_grid,
    :ng3=>ng_adaptive_grid,
    :parity=>[1,-1],
    :tm1=>[2.0,6.0].*eunits,
    :tm2=>[4,6].*eunits,
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
@time results = sweep_parameters(
    evaluate_CQ_TQD_model,
    fixed_parameters,
    sweeped_parameters
    ;
    verbose=false, drop_keys=[:bcfs], pmap_kwargs=Dict()
)
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
