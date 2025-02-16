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
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from analysis_code.plotting_code import (
    plot_dwells_and_splitting,
    plot_timetrace,
    plot_hist_with_fit,
)

from analysis_code.timetrace_analysis import (
    calc_kurtosis,
    calc_dwell_times_along_dim,
)

from analysis_code.common import (
    CONVERTED_DATA_FOLDER,
    panel_labeller,
    prepare_data_for_plotting,
)

```
# Fig.S16


The script reproduces fig.S16 (quasiparticle injection measurements).

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
from pathlib import Path

```

```python
i_V_lin_qd = 13
i_B_perp = 13

i_bias_low = 0
i_bias_high = 4

Kurt_threshold = -0.5

bin_size = 30
```

```python
ds = xr.open_dataset(CONVERTED_DATA_FOLDER / 'injector'/ 'injector_Cq.h5')

ds = prepare_data_for_plotting(ds)

linecut_ds = ds.isel(V_lin_qd=i_V_lin_qd)

timetrace_ds = linecut_ds.isel(B_perp=i_B_perp)
```

```python
kurtosis_da = calc_kurtosis(ds["Cq"])
```

```python
dwells_ups, dwells_downs, splittings = calc_dwell_times_along_dim(
    linecut_ds["Cq"].copy(), param_name="bias", Kurt_threshold=Kurt_threshold
)
```

```python
label = panel_labeller()

fig = plt.figure(figsize=(5.5, 7.5), layout="tight")
gs2 = gridspec.GridSpec(
    2,
    2,
    bottom=0.46,
    width_ratios=(1, 0.25),
    height_ratios=(1, 1),
    wspace=0.05,
    hspace=0.17,
)
gs1 = gridspec.GridSpec(1, 1, top=0.38)


ax00 = fig.add_subplot(gs2[0, 0])
ax01 = fig.add_subplot(gs2[0, 1], sharey=ax00)
ax10 = fig.add_subplot(gs2[1, 0], sharex=ax00)
ax11 = fig.add_subplot(gs2[1, 1], sharex=ax01, sharey=ax10)


for i_bias, axs in zip((i_bias_low, i_bias_high), ((ax00, ax01), (ax10, ax11))):
    show_xlabel = i_bias == i_bias_high
    ds = timetrace_ds.isel(bias=i_bias)

    plot_timetrace(ds, axs[0], show_xlabel=show_xlabel, panel_label=label.next())

    plot_hist_with_fit(
        ds,
        axs[1],
        bin_size=bin_size,
        show_xlabel=show_xlabel,
        plot_fit=False,
        panel_label=label.next(),
    )

ax2 = fig.add_subplot(gs1[:])

plot_dwells_and_splitting(
    ax2, linecut_ds.bias, dwells_ups, dwells_downs, splittings, panel_label=label.next()
)


fig.tight_layout()
fig.savefig("deviceB_qpp_times.pdf", dpi=fig.dpi, bbox_inches="tight", pad_inches=0.01)
plt.show()
```
