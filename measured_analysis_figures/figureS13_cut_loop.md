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
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from analysis_code.plotting_code import (
    plot_kurtosis,
    plot_timetrace,
)

from analysis_code.timetrace_analysis import calc_kurtosis

from analysis_code.common import (
    CONVERTED_DATA_FOLDER,
    panel_labeller,
    prepare_data_for_plotting,
    add_subfig_label,
)
```
# Fig.S13

The script reproduces fig.S13 (open-loop measurements).

The list of dataset variables:
| dataset variable      | variable in the paper        |
| --------------------- | ---------------------------- |
| `Cq`                  | $\mathrm{Re}\, \tilde C_{Q}$ |
| `iCq`                 | $\mathrm{Im}\, \tilde C_{Q}$ |
| `V_qd_1_plunger_gate` | $V_\mathrm{QD1}$             |
| `V_lin_qd`            | $V_\mathrm{QD2}$ *           |
| `V_qd_3_plunger_gate` | $V_\mathrm{QD3}$             |
| `V_wire`              | $V_{WP1}$                    |
| `bias`                | $V_3$                        |
| `B_perp`              | $B_x$                        |

*The offset is subtracted for `V_lin_qd` values

```python
ds_A = xr.open_dataset(CONVERTED_DATA_FOLDER / 'cut_loop_A'/ 'cut_loop_A_Cq.h5')
ds_B = xr.open_dataset(CONVERTED_DATA_FOLDER / 'cut_loop_B'/ 'cut_loop_B_Cq.h5')
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
        "isel_kurtosis": (isel_A_kurtosis := {"V_wire": 5}),
        "isel_trace": isel_A_kurtosis | {"V_lin_qd": 78, "B_perp":16},
        "plot_kwargs": {"linewidth": 0.5, "color": "tab:red", "Cq_tick_base" : 100},
    },
    "device_B": {
        "ds": ds_B,
        "isel_kurtosis": (isel_B_kurtosis := {"bias": 0}),
        "isel_trace": isel_B_kurtosis | {"V_lin_qd": 44, "B_perp": 33},
        "plot_kwargs": {"linewidth": 0.5, "color": "tab:blue"},
    },
}
```

```python
label = panel_labeller("bcde")
fig, axss = plt.subplots(ncols=2, nrows=2, figsize=(6, 5), width_ratios=(1.2, 1))

for device, axs in zip(["A", "B"], axss):

    ax = axs[0]
    ds = data_dict[f"device_{device}"]["ds"]
    plot_kwargs = data_dict[f"device_{device}"]["plot_kwargs"]
    isel_kurtosis = data_dict[f"device_{device}"]["isel_kurtosis"]
    isel_trace = data_dict[f"device_{device}"]["isel_trace"]

    kurtosis_da = ds.isel(**isel_kurtosis)["kurtosis"]
    timetrace_ds = ds.isel(**isel_trace)

    ax = axs[0]
    plot_kurtosis(ax, kurtosis_da, vmin=-1, Bx_tick_base=5)
    ax.scatter([timetrace_ds.B_perp], [timetrace_ds.V_lin_qd], marker = "o", s = 100, color=plot_kwargs['color'], facecolors='none', #ms=15
               )
    add_subfig_label(
        ax,
        label.next(),
        description=f"Device {device}",
    )

    ax = axs[1]
    plot_timetrace(timetrace_ds, ax, show_xlabel=True, **plot_kwargs)
    add_subfig_label(ax, label.next(), description="")


fig.tight_layout()
fig.savefig("cut_loop.pdf", dpi=fig.dpi, bbox_inches="tight", pad_inches=0.01)
plt.show()
```

```python

```
