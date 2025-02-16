---
jupyter:
  jupytext:
    custom_cell_magics: kql
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
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
from xarray_einstats.stats import kurtosis as kurtosis_func

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
```

```python
from analysis_code.plotting_code import (
    plot_kurtosis,
    plot_histogram,
    plot_up_down_dwells_bayesian,
    plot_B_arrows,
    plot_dwells_and_splitting,
    plot_timetrace,
    plot_IQ_blobs,
    plot_hist_with_fit,
)

from analysis_code.timetrace_analysis import (
    histogram,
    aggregate_dwells_from_frame,
    calc_kurtosis,
    calc_dwell_times_along_dim,
)

from analysis_code.common import (
    CONVERTED_DATA_FOLDER,
    panel_labeller,
    prepare_data_for_plotting,
)
```

```python
from pathlib import Path


```

# Fig.S10

The script reproduces fig.S10 (parity measurement for device B1).

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
run_id = "mpr_B1"

device = run_id[-2:]

V_wire = -1.6742

i_V_lin_qd = 57
i_B_perp_node = 28
i_B_perp_antinode = 34

Kurt_threshold = -0.5

bin_size = 20
```

```python
ds = (
    xr.load_dataset(
        CONVERTED_DATA_FOLDER / run_id / f"{run_id}_Cq.h5",
        engine="h5netcdf",
    )
)

ds = prepare_data_for_plotting(ds)

frame_ds = ds.sel(V_wire=V_wire, method='nearest')

linecut_ds = frame_ds.isel(V_lin_qd=i_V_lin_qd)

timetrace_ds_node = linecut_ds.isel(B_perp=i_B_perp_node)
timetrace_ds_antinode = linecut_ds.isel(B_perp=i_B_perp_antinode)
```

```python
kurtosis_da = calc_kurtosis(frame_ds["Cq"])
counts_da = histogram(linecut_ds["Cq"], bins=50)

kurtosis_linecut_da = kurtosis_da.isel(V_lin_qd=i_V_lin_qd)
bimodal_linecut_ds = linecut_ds.where(kurtosis_linecut_da < Kurt_threshold)

(dwells_up, dwells_down, dwells_up_fit, dwells_down_fit) = aggregate_dwells_from_frame(
    bimodal_linecut_ds,
)
```

```python
label = panel_labeller()

fig = plt.figure(figsize=(16, 5))
gs2 = gridspec.GridSpec(
    2,
    5,
    right=1 - 0.405,
    width_ratios=(0.53, 0.17, 0.15, 0.07, 0.32),
    height_ratios=(1, 2.2),
    wspace=0.09,
)
gs1 = gridspec.GridSpec(2, 1, left=1 - 0.35)


for i, timetrace_ds in enumerate([timetrace_ds_node, timetrace_ds_antinode]):

    n_components = [1, 2][i]
    show_xlabel = i == 1
    ax0 = fig.add_subplot(gs2[i, 0])
    plot_timetrace(timetrace_ds, ax0, show_xlabel=show_xlabel, panel_label=label.next())

    ax1 = fig.add_subplot(gs2[i, 1], sharey=ax0)
    plot_IQ_blobs(
        timetrace_ds,
        ax1,
        bin_size=bin_size,
        add_colorbar=n_components == 1,
        panel_label=label.next(),
    )

    ax2 = fig.add_subplot(gs2[i, 2], sharey=ax0)
    plot_hist_with_fit(
        timetrace_ds,
        ax2,
        bin_size=bin_size,
        n_components=n_components,
        plot_fit=True,
        show_xlabel=show_xlabel,
        panel_label=label.next(),
    )

ax = fig.add_subplot(gs2[:, 4])
plot_up_down_dwells_bayesian(
    dwells_up,
    dwells_up_fit,
    dwells_down,
    dwells_down_fit,
    ax=ax,
    panel_label=label.next(),
)

ax = fig.add_subplot(gs1[0, 0])
plot_histogram(ax=ax, counts_da=counts_da, show_xlabel=False, panel_label=label.next())
plot_B_arrows(
    labels=("a-c", "d-f"), pos_dss=(timetrace_ds_node, timetrace_ds_antinode), ax=ax
)

ax = fig.add_subplot(gs1[1, 0])
plot_kurtosis(ax, kurtosis_da, i_V_lin_qd=i_V_lin_qd, panel_label=label.next())

plt.tight_layout()
plt.savefig(
    f"device{device}_parity_measurements.pdf",
    dpi=fig.dpi,
    bbox_inches="tight",
    pad_inches=0.01,
)
```
