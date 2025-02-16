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

## Loading pre-computed data

```julia
data_path = ParityReadoutSimulator.get_simulated_data_path()
file_name = joinpath(data_path,"FigS4_data.h5")
```

```julia
# Alternatively, see end of file to generate new data locally
ds_qMZM = xr.load_dataset(file_name, engine="h5netcdf");
```

## Fig S4 b and c

```julia
# S4(b)
flux_v = 0.:0.05:4
ng2_v = 0.0:0.01:1.0
time_trace_params = TimeTracesParameters(
    45*Units.ms_to_ns, 90*Units.μs_to_ns,
    :ng1 => 0.35:0.35,
    :ng2 => ng2_v,
    :ng3 => 0.65:0.65,
    :phi => flux_v
    ;
    charge_noise = generate_default_charge_noise_parameters(ds_qMZM),
    σR=20.0, # in noise units
    Vrf=10.0, # uV
    τ_qpp=1.0*Units.ms_to_ns,
    periodic_parameters=Dict(:phi=>2.0)
)
```

```julia
Random.seed!(42)
lx = length(ds_qMZM.E_0)
ly = length(ds_qMZM.u1)
fig, ax = subplots(
    ncols=lx,
    nrows=ly, figsize=(1.5*lx,1.0*ly),
    sharex=true,sharey=true
)

img = nothing
all_traces = Dict{Tuple{Float64, Float64}, Array{ComplexF64, 3}}()
@time for i in 1:ly, j in 1:lx
    ds = ds_qMZM.isel(u1=i-1, E_0=j-1).squeeze(drop=true)
    u1 = ds_qMZM.u1.isel(u1=i-1).data[1]
    E_0 = ds_qMZM.E_0.isel(E_0=j-1).data[1]
    itps = build_interpolators_from_xarray(ds.CQ, [:ng1,:ng2,:ng3, :phi])
    traces = generate_time_traces(time_trace_params, itps)
    all_traces[(u1,E_0)] = traces
    kurt = [kurtosis(slice) for slice in eachslice(real.(traces), dims=(2,3))]
    img = ax[i,j].imshow(kurt, aspect="auto", origin="lower", cmap="Greys_r",
        extent=[minimum(flux_v), maximum(flux_v), 0, 1],
        vmin=-1.5, vmax=0.
    )
    ax[i,j].set_xticks([0,2,4])
    if j == 1
        ax[i,j].set_ylabel(L"u_1 =" * "$(round(u1,sigdigits=2))"*L"\,\mathrm{\mu eV}" * "\n" * L"N_{g2}", size=9)
    end
    if i == ly
        ax[i,j].set_xlabel(L"\Phi\  [h/2e]")
    end
    if i == 1
        ax[i,j].set_title(L"E_0 =" * "$(round(E_0,sigdigits=2))"*L"\,\mathrm{\mu eV}", size=9)
    end
end

cbar = fig.colorbar(img, ax=ax[:])
cbar.ax.set_ylabel(L"K(C_Q)")

py"add_subfig_label"(ax[1,1], "b", text_x_shift=-0.3, text_y_shift=0.15)
fig.savefig("qMZM_sims_b.pdf",bbox_inches="tight")
```

```julia
# S4(c)
fig, ax = subplots(
    ncols=3,
    nrows=1, figsize=(1.5*3,1.5*1),
    sharex=true,sharey=true
)
Random.seed!(42)
i_Ng2 = 52
for i in 1:3
    ds = ds_qMZM.isel(u1=0, E_0=i-1).squeeze(drop=true)
    u1 = ds.u1.data[1]
    E_0 = ds.E_0.data[1]
    itps = build_interpolators_from_xarray(ds.CQ, [:ng1,:ng2,:ng3, :phi])
    bins, Z =  make_CQ_vs_flux_histograms(time_trace_params, itps, i_Ng2; N_bin=200, CQ_range=(-0.3, 1.5))
    Zmix = mix_histograms(Z)

    img = ax[i].imshow(Zmix', aspect="auto", origin="lower",
        extent=[minimum(flux_v), maximum(flux_v),minimum(bins), maximum(bins)]
    )

    ax[i].set_xlabel(L"$\Phi$ [h/2e]")
    ax[1].set_ylabel(L"u_1="*"$(round(u1,digits=1))"*L"\,\mathrm{\mu eV}"*"\n"*L"$C_Q$ [fF]")
    ax[i].set_title(L"E_0="*"$(E_0)"*L"\,\mathrm{\mu eV}")
end

py"add_subfig_label"(ax[1,1], "c", text_x_shift=-0.4, text_y_shift=0.10)
fig.savefig("qMZM_sims_c.pdf",bbox_inches="tight")
```

## Generate simulation dataset

```julia
# Optional: Distribute computation over multiple workers for faster compute.
# Default computation of all data takes ~ 15 minutes on 60 cores
using Distributed
number_of_workers = Sys.CPU_THREADS
if nworkers() == 1 && number_of_workers > 1
    addprocs(number_of_workers, exeflags=["--threads=1 --heap-size-hint=2.5G"])
    @everywhere using ParityReadoutSimulator
end;
```

```julia
# Energy parameters are in angular GHz units. Divide by Units.μeV_to_GHz to obtain parameters in μeV
fixed_parameters = generate_default_parameters()
# Override coupling parameter defaults
eunits = Units.μeV_to_GHz
fixed_parameters = merge(
    fixed_parameters,
    Dict(
        :t12=>10*eunits,
        :tm1=>5*eunits,
        :tm2=>5*eunits,
        :t23=>10*eunits,
        :phi=>0.5,
    )
)

# Calculate the bath correlation function based on the parameters of the spectral function
cache_bath_correlation_function(fixed_parameters);
```

```julia
# Parameters to be swept.
ng_adaptive_grid = generate_default_ng_grid()
# We extend the fine grid portion to account for shift of the Cq peak due to the u1
ng_adaptive_grid = sort(unique(vcat(ng_adaptive_grid, 0.325:0.025:0.675)))
sweeped_parameters = Dict{Symbol,Any}(
    :ng1=>[0.25,0.35,0.45],
    :ng2=>ng_adaptive_grid,
    :ng3=>[0.55,0.65,0.75],
    :phi=>0.0:0.2:2.0,
    :parity=>[1,-1],
    :E_M=>[0.0]*eunits,
    :E_0=>[0.0,2.0, 10.0]*eunits,
);
```

```julia
dsets = PyObject[]
for u1 in [0.1, 2.0, 4.0]
    sweeped_parameters[:u1] = u1*eunits
    sweeped_parameters[:u2] = 0.7*sweeped_parameters[:u1]
    @time results = sweep_parameters(
        evaluate_CQ_TQD_model,
        fixed_parameters,
        sweeped_parameters
        ;
        verbose=false, drop_keys=[:bcfs], pmap_kwargs=Dict()
    )
    ds0 = convert_TQD_results_to_xarray(results);
    push!(dsets, ds0)
end
ds_qMZM = xr.merge([ds.squeeze(:u2, drop=true) for ds in dsets]); # We drop u2 from the xarray since it is always 0.7*u1 here.
```

```julia
overwrite_data = false
if overwrite_data # We use h5netcdf to allow for complex numbers
    ds_qMZM.to_netcdf(file_name, engine="h5netcdf", invalid_netcdf=true)
end
```

```julia

```
