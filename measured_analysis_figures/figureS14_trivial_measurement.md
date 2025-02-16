---
jupyter:
  jupytext:
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
from IPython.display import HTML, display
display(HTML("<style>.container { width:95% !important; }</style>"))
```

```python
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from analysis_code.plotting_code import (
    plot_kurtosis,
    plot_histogram,
)

from analysis_code.timetrace_analysis import (
    histogram,
    calc_kurtosis,
)

from analysis_code.common import (
    CONVERTED_DATA_FOLDER,
    panel_labeller,
    prepare_data_for_plotting,
    add_subfig_label,
)
```
# Fig.S14


The script reproduces fig.S14 (low-field measurements).

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
ds_A = xr.open_dataset(CONVERTED_DATA_FOLDER / 'trivial_A'/ 'trivial_A_Cq.h5')
ds_B = xr.open_dataset(CONVERTED_DATA_FOLDER / 'trivial_B'/ 'trivial_B_Cq.h5')
```

```python
ds_A = prepare_data_for_plotting(ds_A)
ds_B = prepare_data_for_plotting(ds_B)
```

```python
ds_A["kurtosis"] = calc_kurtosis(ds_A["Cq"])
ds_B["kurtosis"] = calc_kurtosis(ds_B["Cq"])
```

```python
data_dict = {
    "device_A": {
        "ds": ds_A,
        "isel_kurtosis": (isel_A_kurtosis := {"V_wire": -5}),
        "isel_histogram": isel_A_kurtosis | {"V_lin_qd": 44},
        "plot_kwargs": {"Cq_tick_base": 1000, "Bx_tick_base": 5},
    },
    "device_B": {
        "ds": ds_B,
        "isel_kurtosis": (isel_B_kurtosis := {"V_wire": -6}),
        "isel_histogram": isel_B_kurtosis | {"V_lin_qd": 19},
        "plot_kwargs": {"Cq_tick_base": 250, "Bx_tick_base": 5},
    },
}
```

```python
label = panel_labeller()
fig = plt.figure(
    figsize=(5.7, 3.5),
)
gs = gridspec.GridSpec(2, 2)

pcm_kwargs = dict(linewidth=0, rasterized=True)

for i, device in enumerate(["A", "B"]):

    ds = data_dict[f"device_{device}"]["ds"]
    plot_kwargs = data_dict[f"device_{device}"]["plot_kwargs"]
    isel_kurtosis = data_dict[f"device_{device}"]["isel_kurtosis"]
    isel_histogram = data_dict[f"device_{device}"]["isel_histogram"]

    kurtosis_da = ds.isel(**isel_kurtosis)["kurtosis"]
    counts_da = histogram(
        ds.isel(**isel_histogram)['Cq']
    )

    ax0 = fig.add_subplot(gs[0, i])
    plot_kurtosis(ax0, kurtosis_da, i_V_lin_qd=isel_histogram['V_lin_qd'], show_xlabel=False)
    add_subfig_label(ax0, label.next(), description=f"Device {device}")

    ax1 = fig.add_subplot(gs[1, i], sharex=ax0)
    plot_histogram(ax1, counts_da, **plot_kwargs)
    add_subfig_label(
        ax1,
        label.next(),
    )


fig.subplots_adjust(wspace=0.75, hspace=0.3)
plt.savefig(
    "trivial_measurement.pdf", dpi=fig.dpi, bbox_inches="tight", pad_inches=0.01
)
```
