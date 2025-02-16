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
import matplotlib.pyplot as plt

from analysis_code.plotting_code import (
    plot_eigenvalues,
    plot_timetrace,
    plot_CBP_histogram,
    plot_digital_timetrace,
    plot_dwells_bayesian,
)

from analysis_code.timetrace_analysis import (
    run_gmm,
    digitize_traces,
    extract_dwell_times,
    prepare_plot_dwells_bayesian,
    CPB_eigenvals,
)

from analysis_code.common import (
    CONVERTED_DATA_FOLDER,
    add_subfig_label,
    prepare_data_for_plotting,
    panel_labeller,
)
```

# Fig.S17


The script reproduces fig.S17 (quasiparticles in Cooper-pair box).

The list of dataset variables:
| dataset variable | variable in the paper        |
| ---------------- | ---------------------------- |
| `Cq`             | $\mathrm{Re}\, \tilde C_{Q}$ |
| `iCq`            | $\mathrm{Im}\, \tilde C_{Q}$ |

```python
ds = xr.load_dataset(CONVERTED_DATA_FOLDER / "qpp/qpp_Cq.h5")
ds = prepare_data_for_plotting(ds)
```

```python
da = ds["Cq"].copy()

_, mean1_da, mean2_da = run_gmm(da)
digital_traces_da = digitize_traces(
    da,
    mean1_da,
    mean2_da,
)

sample_rate = da.time.diff(dim = 'time')[0].values
trace_len = da.time.max(dim = 'time').values

dwells_up, dwells_down = extract_dwell_times(digital_traces_da)
dwells_fit = prepare_plot_dwells_bayesian(dwells=dwells_up, sample_rate=sample_rate, trace_len=trace_len)
```

```python
label = panel_labeller("cdef")

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4.8, 3.4))


plot_eigenvalues(ax, *CPB_eigenvals(Ec = 110.0, Ej = 23.0))
add_subfig_label(ax, label.next())

plt.tight_layout()
fig.savefig("qpp_levels.pdf", dpi=fig.dpi, bbox_inches='tight', pad_inches=0.01)
plt.show()


fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(4.05, 9.2625))

ax= axs[0]
plot_timetrace(ds.sel(time=slice(0, 50)), ax, Cq_tick_base = 1000,  color="tab:olive",  show_xlabel=True, zorder=50)
plot_digital_timetrace(digital_traces_da.sel(time=slice(0, 50)),
    mean1_da,
    mean2_da,
    ax)
add_subfig_label(ax, label.next())

ax= axs[1]
plot_CBP_histogram(ds = ds, bins = 64, ax = ax)
add_subfig_label(ax, label.next())

ax= axs[2]
plot_dwells_bayesian(ax, dwells = dwells_up, model_fit_res =dwells_fit, s = 30, bin_scale=6.7, color = 'red', plot_interval = True)
ax.set_xlim(-0.5, 10.1)
add_subfig_label(ax, label.next())


plt.tight_layout()
plt.subplots_adjust(hspace=0.32)
fig.savefig("qpp.pdf", dpi=fig.dpi, bbox_inches='tight', pad_inches=0.01)
plt.show()
```

```python

```
