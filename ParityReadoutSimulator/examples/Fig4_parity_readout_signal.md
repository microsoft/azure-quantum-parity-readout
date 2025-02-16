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

# Figure 4: Parity Readout Signal Simulation


## Loading pre-computed data

```julia
data_path = ParityReadoutSimulator.get_simulated_data_path()
file_name = joinpath(data_path,"Fig4_data.h5")
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
    :ng1=>0.3:0.3,
    :ng2=>0.0:0.01:1.0,
    :ng3=>0.3:0.3,
    :phi=>LinRange(0.0, 4.0, 70)
    ;
    charge_noise = generate_default_charge_noise_parameters(ds),
    σR=20.0, # in noise units
    Vrf=10.0, # uV
    τ_qpp=1.0*Units.ms_to_ns,
    periodic_parameters=Dict(:phi=>2.0)
)
itps = build_interpolators_from_xarray(ds.CQ, [:ng1, :ng2, :ng3, :phi]);
```

```julia
Random.seed!(4)
let
    i_Ng2=49
    traces = generate_time_traces(time_trace_params, itps; return_xarray=false)
    flux_v = time_trace_params.sweepPoints[:phi]
    ng2_v = time_trace_params.sweepPoints[:ng2]
    kurt = [kurtosis(slice) for slice in eachslice(real.(traces), dims=(2,3))]

    fig, ax = subplots(ncols=1,nrows=2, figsize=(5.5,3.5), sharex=true)

    bins, Z =  make_CQ_vs_flux_histograms(time_trace_params, itps, i_Ng2; N_bin=200, CQ_range=(-0.3, 1.5))
    Zmix = mix_histograms(Z)
    img = ax[1].imshow(Zmix', aspect="auto", origin="lower",
        extent=[minimum(flux_v), maximum(flux_v),minimum(bins), maximum(bins)]
    )
    ax[2].set_xlabel(L"$\Phi$ [h/2e]")
    ax[1].set_ylabel(L"$C_Q$ [fF]")

    cbar = colorbar(img, label="", ax=ax[1],ticks=[])
    cbar.solids.set_edgecolor("white")
    cbar.outline.set_color("white")

    img2 = ax[2].imshow(kurt, aspect="auto", origin="lower", cmap="Greys_r",
        extent=[minimum(flux_v), maximum(flux_v), 0, 1],
        vmin=-2.1, vmax=0.)
    ax[2].set_ylabel(L"N_{g2}")
    colorbar(img2, label=L"$K(C_Q)$", ax=ax[2])
    py"add_rectangle_horizontal"(ax[2], ng2_v[i_Ng2], 0.15, wpad=0.05)

    py"add_subfig_label"(ax[2], "b")
    py"add_subfig_label"(ax[1], "a")

    fig.tight_layout()
    plt.savefig("Fig4.pdf")
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
# In this example, we keep a coarse grid for quick compute time
ng_adaptive_grid = generate_default_ng_grid()
sweeped_parameters = Dict(
    :ng1=>0.2:0.1:0.4,
    :ng2=>ng_adaptive_grid,
    :ng3=>0.2:0.1:0.4,
    :phi=>0.0:0.2:2.0,
    :parity=>[1,-1],
);
```

```julia
# Optional: Distribute computation over multiple workers for faster compute.
# Default computation takes ~ 3 minutes on single core
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
