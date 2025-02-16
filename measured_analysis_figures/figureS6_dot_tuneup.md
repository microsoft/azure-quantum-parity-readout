---
jupyter:
  jupytext:
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
from matplotlib import pyplot as plt
from analysis_code.common import (
    RAW_DATA_FOLDER,
    add_subfig_label,
)
```
# Fig.S6


The script reproduces fig.S6 (dot tune-up).

The list of dataset variables:
| dataset variable                | variable in the paper |
| ------------------------------- | --------------------- |
| `Vrf_0`                         |                       |
| `Vrf_1`                         |                       |
| `i_1`                           |                       |
| `i_2`                           |                       |
| `V_qd_1_plunger_gate`           | $V_\mathrm{QD1}$      |
| `V_linear_qd_left_plunger_gate` | $V_\mathrm{QD2}$      |
| `V_qd_2_plunger_gate`           | $V_\mathrm{QD3}$      |
| `bias`                          |                       |

```python
exp_labels = [
    "QD1_bias",
    "QD2_bias",
    "QD3_bias",
    "QD1_QD3",
]

datasets = [
    xr.load_dataset(
        RAW_DATA_FOLDER / f"dot_tuneup_A1/{elabel}.h5",
        engine="h5netcdf",
        invalid_netcdf=True,
    )
    for elabel in exp_labels
]

plunger_labels = ["$V_\mathrm{QD1}$", "$V_\mathrm{QD2}$", "$V_\mathrm{QD3}$"]
for plabel, ds in zip(plunger_labels, datasets[:3]):
    bias_dim, plunger_dim = list(ds.dims)
    ds['i'] -= np.median(ds["i"])
    ds['i'] *= 1e9
    ds[bias_dim] = ds[bias_dim]*1e3
    ds[bias_dim].attrs['long_name'] = 'Bias'
    ds[bias_dim].attrs['units'] = 'mV'
    ds[plunger_dim].attrs['long_name'] = plabel
    ds['i'].attrs['long_name'] = 'Current'
    ds['i'].attrs['units'] = 'nA'

ds = datasets[-1]
ds = ds.apply(np.angle)
for i in [0, 2]:
    ds[f"V_qd_{i+1}_plunger_gate"].attrs['long_name'] = plunger_labels[i]
    ds[f"V_qd_{i+1}_plunger_gate"].attrs['units'] = 'V'
ds['Vrf'].attrs['long_name'] = 'RF signal'
ds['Vrf'].attrs['units'] = 'rad.'
datasets[-1] = ds
```

```python
pcm_kwargs = dict(linewidth=0, rasterized=True)

fig, axs = plt.subplots(1, 4, figsize=(15, 3.0))

for plabel, ds, ax, label in zip(plunger_labels, datasets[:3], axs[:3], "abc"):
    pcm = ds["i"].real.plot.pcolormesh(ax=ax, **pcm_kwargs)
    pcm.set_edgecolor("face")

pcm = datasets[-1]["Vrf"].plot(ax=axs[-1], **pcm_kwargs)
pcm.set_edgecolor("face")

for ax, label in zip(axs, "abcd"):
    add_subfig_label(ax=ax, label=label)

plt.tight_layout()

plt.savefig("dot_tuneup.pdf", dpi=fig.dpi, bbox_inches='tight', pad_inches=0.01)
plt.show()
```
