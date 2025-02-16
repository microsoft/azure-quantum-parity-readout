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
from xarray_einstats.stats import kurtosis

from analysis_code.plotting_code import (
    plot_kurtosis,
    plot_qdmzm,
    plot_histogram,
    plot_sim_histogram,
)

from analysis_code.timetrace_analysis import (
    determine_ng_range,
    get_detuning,
    histogram,
)

from analysis_code.common import (
    CONVERTED_DATA_FOLDER,
    SIMULATION_DATA_FOLDER,
    panel_labeller,
    add_subfig_label,
    prepare_data_for_plotting,
)
```

# Fig.S12

The script reproduces fig.S12 (evolution of histograms for device A1 and simulations).

The list of dataset variables:
| dataset variable      | variable in the paper        |
| --------------------- | ---------------------------- |
| `Cq`                  | $\mathrm{Re}\, \tilde C_{Q}$ |
| `iCq`                 | $\mathrm{Im}\, \tilde C_{Q}$ |
| `V_qd_1_plunger_gate` | $V_\mathrm{QD1}$             |
| `V_lin_qd`            | $V_\mathrm{QD2}$ *           |
| `V_qd_3_plunger_gate` | $V_\mathrm{QD3}$             |
| `V_wire`              | $V_{WP1}$                    |
| `B_perp`              | $B_x$                        |

*The offset is subtracted for `V_lin_qd` values

```python
%%time

mpr_ds = xr.open_dataset(CONVERTED_DATA_FOLDER / 'mpr_A1'/ 'mpr_A1_Cq.h5')
mpr_ds_sim = xr.load_dataset(SIMULATION_DATA_FOLDER / "FigS11_S12_noisy_data.h5",engine="h5netcdf") * 1e3
```

```python
mpr_ds = prepare_data_for_plotting(mpr_ds)
mpr_ds_sim = prepare_data_for_plotting(mpr_ds_sim)
```

```python
V_wire = -1.8314
flux_period = 1.9

flux_range = np.ptp(mpr_ds.B_perp.data)/flux_period

data_dict = {
    "Measured": {
        "ds": mpr_ds.Cq,
        "sel": [
            {"V_wire": V_wire, "V_lin_qd": mpr_ds["V_lin_qd"][32]},
            {"V_wire": V_wire + 0.39e-3, "V_lin_qd": mpr_ds["V_lin_qd"][19]},
            {"V_wire": V_wire + 0.8e-3, "V_lin_qd": mpr_ds["V_lin_qd"][57]},
        ],
    },
    "Simulated": {
        "ds": mpr_ds_sim.sel({"phi" : slice(0, flux_range)}),
        "sel": [
            {"tm1": 6, "E_M": 0, "ng2": 0.50},
            {"tm1": 6, "E_M": 3, "ng2": 0.41},
            {"tm1": 0.1, "E_M": 6, "ng2": 0.33},
        ],
    },
}
```

```python
print(np.round(mpr_ds["V_lin_qd"][19].data,2))
```

```python
label = panel_labeller()

fig, axss = plt.subplots(2, 3, figsize=(12, 4), sharey=True)

for axs, data_type in zip(axss, ["Measured", "Simulated"]):

    params = data_dict[data_type]
    ds = params["ds"]

    for ax, sel in zip(axs, params["sel"]):

        ds_sel = ds.sel(**sel, method="nearest")

        if data_type == "Measured":
            counts_da = histogram(ds_sel, bins=50)
            plot_histogram(
                ax=ax, counts_da=counts_da, Cq_tick_base=1000, add_colorbar=False
            )

            dV_wire = sel["V_wire"] - V_wire
            description = rf"$\Delta V_{{WP1}} = {{{dV_wire*1e3 :1.2f}}}$ mV"
            add_subfig_label(ax, label=label.next(), description=description)
            ax.set_ylim(-1000, 1700)

        else:
            bin_range = (ds_sel.CQ.real.min().values, ds_sel.CQ.real.max().values)
            counts_even = histogram(ds_sel.CQ_even.real, bins=50, bin_range=bin_range)
            counts_odd = histogram(ds_sel.CQ_odd.real, bins=50, bin_range=bin_range)
            plot_sim_histogram(counts_even, counts_odd, ax)

            tm1 = sel["tm1"]
            E_M = sel["E_M"]
            description = (
                rf"$t_{{m1}} = {{{tm1}}}, t_{{m2}} = 4, E_M = {{{E_M}}}$ $[\mu eV]$ "
            )
            add_subfig_label(ax, label=label.next(), description=description)

        ax.set_ylim(-200,1600)

fig.tight_layout()
fig.savefig("scenarios.pdf", dpi=fig.dpi, bbox_inches="tight", pad_inches=0.01)
```

```python

```
