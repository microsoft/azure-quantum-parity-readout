---
jupyter:
  jupytext:
    custom_cell_magics: kql
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
  kernelspec:
    display_name: python
    language: python
    name: python
---

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
```

```python
import sys
sys.path.append("..")

from thermometry.curve_sets import extract_Cqs_v_and_plot
```

```python
from analysis_code.thermometry_helpers import (
    saturation_temperature_model,
    parallelized_find_houghlines,
    parallel_fit_DQD_model,
    tmc_to_tpuck,
    Tmc,
    Tpuck,
    prepare_data_into_dict,
    fit_tsat_curve_errorbars,
    numerically_estimate_err_bar_dist
)
```

```python
from analysis_code.common import (
    CONVERTED_DATA_FOLDER,
    add_subfig_label,
)
```

```python
import numpy as np
import xarray as xr
from copy import deepcopy

import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor
import re
from tqdm import tqdm
```

# Fig.S19


The script reproduces fig.S19 (thermometry).

The list of dataset variables:
| dataset variable | variable in the paper        |
| ---------------- | ---------------------------- |
| `Cq`             | $\mathrm{Re}\, \tilde C_{Q}$ |
| `iCq`            | $\mathrm{Im}\, \tilde C_{Q}$ |
| `V_QC1`          |     $V_\mathrm{QC1}$         |
| `V_QD1`          |     $V_\mathrm{QD1}$         |
| `V_DG2`          |     $V_\mathrm{DG2}$         |


```python
dataset_paths = sorted(
    CONVERTED_DATA_FOLDER.glob("thermometry/*.h5"),
    # Order the files by the temperature
    key=lambda f: int(re.findall("thermometry_(\d+)mK_Cq", f.stem)[0]),
)
```

```python
# load all the datasets
datasets = {dataset: xr.load_dataset(dataset, engine="h5netcdf").squeeze() for dataset in dataset_paths}

# Helper code to convert dataset names to the recorded mixing chamber temperature
def ds_name_to_temp(name):
    return datasets[name].T_MC.to_numpy()*1000
```

```python
sizedataset = [len(datasets[dsp].V_QC1) for dsp in dataset_paths]
dataset_space = [(dataset_paths[k], j) for k in range(len(sizedataset)) for j in range(sizedataset[k])]
gategate_map_datasets = [datasets[dataset_paths[k]]["Cq"].isel(V_QC1=j).squeeze().to_numpy() for k in range(len(sizedataset)) for j in range(sizedataset[k])]

# find all the lines in the various gate-gate maps using a modified version fo the hough lines algorithm
lines = []
with ProcessPoolExecutor(max_workers=2) as exe: # use 2 threads
    lines = list(tqdm(exe.map(parallelized_find_houghlines, gategate_map_datasets), total=len(dataset_space)))
```

```python
# Get a collection of grouped Cqs(ng) curves that follow the hough lines extracted above
Cqs_v = extract_Cqs_v_and_plot(datasets, lines, plot=False)
```

```python
%%time
np.random.seed(0)

# leverarms to sweep due to uncertainty
leverarm_sweep = [0.37, 0.375, 0.38]

# a map of data that can be fit separate from each other to allow for multi-threading
cost_function_inputs = [(
    (i, k, la), # an index useful for plotting later (ds_path, cutter_value (choice of gate_gate map), leverarm value)
    Cqs_v[i][k], # the slices to be fit
    ds_name_to_temp(i) - 2 # an initial condition for temperature
) for i in Cqs_v.keys() for k in Cqs_v[i].keys() for la in leverarm_sweep]
```

```python
%%time
# Do all fits - this can take a few minutes.
opts = []
with ProcessPoolExecutor(max_workers=None) as exe:
    opts = list(tqdm(exe.map(parallel_fit_DQD_model, cost_function_inputs), total=len(cost_function_inputs)))
```

```python
nominal_leverarm = 0.375

opts_for_plot = {leverarm: [((i[0][0], i[0][1]), ds_name_to_temp(i[0][0]), i[1]) for i in opts if i is not None and i[0][-1] == leverarm] for leverarm in leverarm_sweep}
```

```python
# Choose the slices we want to plot for the paper figure

idx_line_to_plot_1 = 6 # 30mK ds pane 7 (idx 6)
pane_line_1 = 0 # idx of the line in the 30mK pane chosen above

idx_line_to_plot_2 = 62 # 150mK ds pane 3 (idx 2)
pane_line_2 = 1 # idx of the line in the 30mK pane chosen above

line_to_plot_1 = lines[idx_line_to_plot_1][pane_line_1]
ds_1, cutter_slice_1 = xr.load_dataset(dataset_space[idx_line_to_plot_1][0]), dataset_space[idx_line_to_plot_1][1]

line_to_plot_2 = lines[idx_line_to_plot_2][pane_line_2]
ds_2, cutter_slice_2 = xr.load_dataset(dataset_space[idx_line_to_plot_2][0]), dataset_space[idx_line_to_plot_2][1]
```

```python
# Checking tmc vs. tpuck fit consistent with measured data
Tmctheory = np.linspace(0, 160, 101)

plt.scatter(Tmc, Tpuck)
plt.plot(Tmctheory, tmc_to_tpuck(Tmctheory))
plt.plot(Tmctheory, Tmctheory, "k--")
plt.xlim(0, None)
plt.ylim(0, None)

plt.xlabel("Mixing Chamber Temperature [mK]")
plt.ylabel("Puck Temperature [mK]")
```

```python
# Initialize Figure with subplots
fig = plt.figure(figsize=(16, 8), dpi=190, layout="constrained")

rowspan = 2
fig_shape = (3*rowspan+1, 6)

gategateax1 = plt.subplot2grid(fig_shape, (0, 0), colspan=1, rowspan=rowspan)
gategateax2 = plt.subplot2grid(fig_shape, (0, 1), colspan=1, rowspan=rowspan, sharex=gategateax1, sharey=gategateax1)
gategateax2.get_yaxis().set_visible(False)

fit_ax1 = plt.subplot2grid(fig_shape, (rowspan, 0), colspan=1, rowspan=rowspan)
fit_ax2 = plt.subplot2grid(fig_shape, (rowspan, 1), colspan=1, rowspan=rowspan, sharex=fit_ax1, sharey=fit_ax1)
fit_ax2.get_yaxis().set_visible(False)

Te_ax = plt.subplot2grid(fig_shape, (2*rowspan, 0), colspan=2, rowspan=rowspan+1)
fit_ax = [fit_ax1, fit_ax2]
gategateaxs = [gategateax1, gategateax2]

for subax in gategateaxs:
    subax.clear()

x, y = ds_1.V_DG2, ds_1.V_QD1

global im

# plot Gate-Gate Maps
for ids, (ds, cutter_slice) in enumerate([(ds_1, cutter_slice_1), (ds_2, cutter_slice_2)]):
    slice = ds.isel(
        V_QC1=cutter_slice,
    ).squeeze()
    gategateaxs[ids].set_title("")

    im = gategateaxs[ids].imshow(slice.Cq,
                                 aspect=14,
                                 origin='lower',
                                 interpolation="none",
                                 extent=[min(x), max(x), min(y), max(y)],
                                 vmin=-50, vmax=1300,
                                 cmap="Greys")

    gategateaxs[ids].set_ylabel("$V_\mathrm{QD1}$ [V]")
    gategateaxs[ids].set_xlabel("$V_\mathrm{DG2}$ [V]")
    if ids==1:
        fig.colorbar(im, label=r"$\mathrm{Re}\,\tilde{C}_\mathrm{Q}$ [aF]", orientation="horizontal", ax=gategateaxs, shrink=0.4, location="top")

    gategateaxs[ids].set_xticks([-2.04, -2.0325])

add_subfig_label(gategateaxs[0], "a")
add_subfig_label(gategateaxs[1], "b")

# draw lines on peaks that the Cq(Vg2) are taken from
gategateaxs[0].plot(x[line_to_plot_1.xs], y[line_to_plot_1.ys], ":", color="red", zorder=2)
gategateaxs[1].plot(x[line_to_plot_2.xs], y[line_to_plot_2.ys], ":", color="red", zorder=2)
```

```python
opts_dict_for_plot = {(i[0], i[1]): (k, j) for i, j, k in opts_for_plot[nominal_leverarm]}
```

```python
for subax in fit_ax:
    subax.clear()

opt_to_plot_1, temp_1 = opts_dict_for_plot[dataset_space[idx_line_to_plot_1]]
opt_to_plot_2, temp_2 = opts_dict_for_plot[dataset_space[idx_line_to_plot_2]]


for plot_idx, (opt_to_plot, line_idx) in enumerate([(opt_to_plot_1, pane_line_1), (opt_to_plot_2, pane_line_2)]):

    temp, t0, Γ = opt_to_plot.get_temp(), opt_to_plot.get_tm()[line_idx], opt_to_plot.get_Γ()[line_idx]
    offset, shifts = opt_to_plot.get_offset(), opt_to_plot.get_shifts()[line_idx]
    leverarm = opt_to_plot.get_leverarm()

    leverarm_xscaling = opt_to_plot.coulomb_diamond.xscale(opt_to_plot.model.coulomb_diamond.Ec)

    Cq_cuts = deepcopy(opt_to_plot.Cqs_v[line_idx])

    ng_model = opt_to_plot.model.V0s
    CQ_model = opt_to_plot.model.curve(
        t0,
        temp,
        Γ,
    ) * opt_to_plot.coulomb_diamond.yscale(target_alpha=opt_to_plot.model.coulomb_diamond.alpha, alpha=leverarm)

    for cut_index in range(len(Cq_cuts)):
        ng_data = Cq_cuts[cut_index][0]
        ng_data = ng_data*leverarm_xscaling - shifts[cut_index]

        real_CQ_data = Cq_cuts[cut_index][1]
        real_CQ_data -= np.real(offset)

        # imag data stored in Cq_cuts[cut_index][2]

        fit_ax[plot_idx].plot(ng_data, real_CQ_data/1000)

    fit_ax[plot_idx].plot(ng_model, np.real(CQ_model)/1000, "--", color="k",
        label="$T_e={}$ mK \n $t_0={}$ μeV".format(
         np.round(np.round(temp, 0), decimals=1),
         np.round(t0, decimals=1),
     ))

    fit_ax[plot_idx].set_xlabel("$N_g$")
    fit_ax[plot_idx].set_xlim(-0.5, 0.5)
    fit_ax[plot_idx].set_ylim(-50/1000, None)

    fit_ax[plot_idx].set_ylabel(r"$\mathrm{Re}\,\tilde{C}_\mathrm{Q}$ [fF]")
    fit_ax[plot_idx].legend(fontsize=7, loc=2, frameon=False)

for idx, temp in enumerate([temp_1, temp_2]):
    gategateaxs[idx].set_title(f"""$T_\mathrm{{puck}}={
        int(np.round(tmc_to_tpuck([temp])[0], 0))
    }$mK""")

add_subfig_label(fit_ax[0], "c")
add_subfig_label(fit_ax[1], "d")

fig
```

```python
Te_ax.clear()

data_to_plot = prepare_data_into_dict(opts_for_plot, nominal_leverarm)
for i in opts_for_plot[nominal_leverarm]:
    _, temp, ovec = i
    Te_ax.scatter(tmc_to_tpuck([temp]), ovec.get_temp(), color="k", alpha=0.3)

tx = np.linspace(0, 300, 201)

Te_ax.plot(tx, tx, "--", c="k", alpha=0.3, label="$T_e=T_\mathrm{puck}$")

power = 5

# get the Te for the 0.375 dataset
fitted_Te, _ = fit_tsat_curve_errorbars(data_to_plot, power)

# here we are plotting the scatter of 0.375 leverarm but the aggregate fit result of the std (the std of all 3 leverarms we swept over)
# in this case, the mean is the same as the fit for the 0.375 data.

fits_for_different_leverarms = [fit_tsat_curve_errorbars(prepare_data_into_dict(opts_for_plot, leverarm), power) for leverarm in leverarm_sweep]

fitted_Te, fitted_Te_err_bar = numerically_estimate_err_bar_dist(fits_for_different_leverarms)

# plotting fitted function on top of all leverarm data

Te_ax.plot(tx, saturation_temperature_model(power, fitted_Te, tx), c='royalblue', label=f"$T_\mathrm{{sat}}={int(np.round(fitted_Te, 0))} \\pm {int(np.round(fitted_Te_err_bar, 0))}$ mK")
Te_ax.fill_between(tx, *[saturation_temperature_model(power, Te_quartile, tx) for Te_quartile in [fitted_Te-fitted_Te_err_bar, fitted_Te+fitted_Te_err_bar]], color='royalblue', alpha=0.3)

means = {k: np.mean(v) for k, v in data_to_plot.items()}
medians = {k: np.median(v) for k, v in data_to_plot.items()}
Te_ax.scatter(tmc_to_tpuck(medians.keys()), medians.values(), label="median")

Te_ax.set_xlabel("$T_\mathrm{puck}$ [mK]")
Te_ax.set_ylabel("Fitted $T_{e}$ [mK]")
Te_ax.legend(frameon=False)

add_subfig_label(Te_ax, "e")

fig.savefig("deviceB_thermometry.pdf", bbox_inches='tight')
fig
```

```python

```
