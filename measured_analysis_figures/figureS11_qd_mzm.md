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
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
```

```python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from analysis_code.plotting_code import (
    plot_kurtosis,
    plot_qdmzm,
)

from analysis_code.timetrace_analysis import (
    determine_ng_range,
    get_detuning,
    calc_kurtosis,
)

from analysis_code.common import (
    CONVERTED_DATA_FOLDER,
    SIMULATION_DATA_FOLDER,
    panel_labeller,
    add_subfig_label,
    prepare_data_for_plotting,
)

```

# Fig.S11


The script reproduces fig.S11 (QD-MZM coupling for device A1 and simulations).

The list of dataset variables:
| dataset variable       | variable in the paper        |
| ---------------------- | ---------------------------- |
| `Cq`                   | $\mathrm{Re}\, \tilde C_{Q}$ |
| `iCq`                  | $\mathrm{Im}\, \tilde C_{Q}$ |
| `V_qd_1_plunger_gate`  | $V_\mathrm{QD1}$             |
| `V_lin_qd`             | $V_\mathrm{QD2}$ *           |
| `V_qd_3_plunger_gate`  | $V_\mathrm{QD3}$             |
| `V_wire`               | $V_{WP1}$                    |
| `B_perp`               | $B_x$                        |
| `Ng_qd_1_plunger_gate` | $N_{g1}$                     |
| `Ng_qd_3_plunger_gate` | $N_{g3}$                     |

*The offset is subtracted for `V_lin_qd` values

```python
%%time

mpr_ds = xr.open_dataset(CONVERTED_DATA_FOLDER / 'mpr_A1'/ 'mpr_A1_Cq.h5')
qd1mzm_ds = xr.open_dataset(CONVERTED_DATA_FOLDER / 'qdmzm_A1'/ 'qd1mzm_Cq.h5')
qd3mzm_ds = xr.open_dataset(CONVERTED_DATA_FOLDER / 'qdmzm_A1'/ 'qd3mzm_Cq.h5')
```

```python
mpr_ds = prepare_data_for_plotting(mpr_ds)
qd1mzm_ds = prepare_data_for_plotting(qd1mzm_ds)
qd3mzm_ds = prepare_data_for_plotting(qd3mzm_ds)
```

```python
for ds in [mpr_ds, qd1mzm_ds, qd3mzm_ds]:
    ds.V_lin_qd.attrs.update({"long_name" : r"$V_\mathrm{QD2}$"})
```

```python
mpr_ds['kurtosis'] = calc_kurtosis(mpr_ds['Cq'])
```

```python
mpr_ds_sim = xr.load_dataset(SIMULATION_DATA_FOLDER / "FigS11_S12_noisy_data.h5",engine="h5netcdf") *1e3
qd1mzm_ds_sim = xr.load_dataset(SIMULATION_DATA_FOLDER / "FigS11_QDL.h5",engine="h5netcdf") * 1e3
qd3mzm_ds_sim = xr.load_dataset(SIMULATION_DATA_FOLDER / "FigS11_QDR.h5",engine="h5netcdf") *1e3
```

```python
mpr_ds_sim = prepare_data_for_plotting(mpr_ds_sim)
qd1mzm_ds_sim = prepare_data_for_plotting(qd1mzm_ds_sim)
qd3mzm_ds_sim = prepare_data_for_plotting(qd3mzm_ds_sim)
```

```python
mpr_ds_sim['kurtosis'] = calc_kurtosis(mpr_ds_sim['CQ'].real)
```

```python
V_wire0 = -1.8314
V_wire1 = -1.831

flux_period = 1.9 # in mT

avg_ds = mpr_ds.mean(dim = 'time')

ng_range0 = determine_ng_range(avg_ds, V_wire0, 1, verbose = False)
ng_range1 = determine_ng_range(avg_ds, V_wire1, 1, verbose = False)

flux_range = np.ptp(mpr_ds.B_perp.data)/flux_period
```

```python
mpr_sel = {"tm1" : 6, "ng1" : 0.7, "ng3" : 0.35, "phi" : slice(0, flux_range)}

data_dict = {
    "Measured":
        {
            "qd1mzm" : qd1mzm_ds,
            "qd3mzm" : qd3mzm_ds,
            "mpr"    : mpr_ds,
            "small_EM_sel" : {'V_wire' : V_wire0, "V_lin_qd" : slice(*ng_range0)},
            "large_EM_sel" : {'V_wire' : V_wire1, "V_lin_qd" : slice(*ng_range1)},
            "plot_kwargs" : {"Cq_range" : (-500, 500)},
            "kurt_plot_kwargs" : {"ylabel" : r"$V_\mathrm{QD2}$ [mV]"},
         },
    "Simulated":
        {
            "qd1mzm" : qd1mzm_ds_sim,
            "qd3mzm" : qd3mzm_ds_sim,
            "mpr"    : mpr_ds_sim.sel(**mpr_sel),
            "small_EM_sel" : {'E_M' : 0},
            "large_EM_sel" : {'E_M' : 3.0},
            "plot_kwargs" : {"Cq_range" : (0, 1000)},
            "kurt_plot_kwargs" : {"xlabel" : r"$\Phi$ [$h/2e$]", "ylabel" : r"$N_\mathrm{g2}$"}
         }
}
```

```python
line_kwargs = dict(lw=1.5, ls="--", c="k")
cbar_kwargs = dict(aspect=12, pad=0.06)
kurtosis_cbar_kwargs = dict(label="$K(C_\mathrm{Q})$", **cbar_kwargs)

fig, axss = plt.subplots(2, 4, figsize=(12, 4.5))

label = panel_labeller("abefcdgh")

for axs, data_type in zip(axss[:,:2], ['Measured', 'Simulated']):
    for ax, N_qd in zip(axs, ['1', '3']):

        ds = data_dict[data_type][f'qd{N_qd}mzm']
        sel = data_dict[data_type]['small_EM_sel']
        plot_kwargs = data_dict[data_type]['plot_kwargs']
        frame = ds.sel(**sel,)

        plot_qdmzm(frame, ax, N_qd, data_type = data_type, **plot_kwargs)

        mpr_ds_sel = data_dict[data_type][f'mpr'].sel(**sel)
        Vqd = get_detuning(mpr_ds_sel, N_qd, data_type)
        ax.axvline(Vqd, **line_kwargs)

        ax.set_title('')
        add_subfig_label(ax, label = label.next(), description=data_type)

for axs, data_type in zip(axss[:,2:], ['Measured', 'Simulated']):
    for ax, EM in zip(axs, ['small', 'large']):

        ds = data_dict[data_type][f'mpr'].kurtosis
        sel = data_dict[data_type][f'{EM}_EM_sel']
        kurt_plot_kwargs = data_dict[data_type]["kurt_plot_kwargs"]
        frame = ds.sel(**sel)

        x = 'B_perp' if data_type == "Measured" else 'phi'
        plot_kurtosis(ax, frame, x=x,cbar_kwargs = kurtosis_cbar_kwargs, **kurt_plot_kwargs)

        add_subfig_label(ax, label = label.next(), description=data_type)


fig.tight_layout()
plt.savefig(
    f"qd_mzm.pdf",
    dpi=fig.dpi,
    bbox_inches="tight",
    pad_inches=0.01,
)
```
