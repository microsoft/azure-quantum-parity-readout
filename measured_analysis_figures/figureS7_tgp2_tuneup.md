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
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from analysis_code.common import (
    RAW_DATA_FOLDER,
    add_subfig_label
)
from analysis_code.tgp_analysis import (
    analyze_two,
    plot_stage2_diagram
)
```
# Fig. S7

The script reproduces Fig. S7 (TGP2 tuneup) for devices A1, A2, and B1.

```python
device_list = ["A1", "A2", "B1"]
runs = {
    device: {
        ds_type: xr.load_dataset(
            RAW_DATA_FOLDER / "tgp2_tuneup" / f"device{device}_tgp2_{ds_type}.h5",
            engine="h5netcdf",
            invalid_netcdf=True,
        )
        for ds_type in ["ds_left", "ds_right"]
    }
    for device in device_list
}
```

```python
for device, run in runs.items():
    run["zbp_ds"] = analyze_two(run["ds_left"], run["ds_right"]).zbp_ds
```

```python
infos = {
    "A1": dict(
        cutter_value=2,
        zbp_cluster_numbers=[1],
        figsize=(10.15, 6.16),
        pct_boundary_shifts={1: [0.48, -0.0005]},
        deco_yfactor=0.56,
        xlim=(1.5, 3.0),
        xticks=np.arange(1.6, 2.9, 0.2),
        yticks=np.arange(-1.827, -1.817+1e-6, 0.001),
    ),
    "A2": dict(
        cutter_value=1,
        zbp_cluster_numbers=[1],
        figsize=(10.15, 3.42),
        pct_boundary_shifts={1: [0.3, -0.0006]},
        deco_yfactor=1.0,
        xlim=(1.5, 3.0),
        xticks=np.arange(1.6, 2.9, 0.2),
        yticks=np.arange(-1.846, -1.841+1e-6, 0.001),
    ),
    "B1": dict(
        cutter_value=2,
        zbp_cluster_numbers=[1],
        figsize=(10.15, 3.42),
        pct_boundary_shifts={1: [0.40, 0.0007]},
        deco_yfactor=1.0,
        xlim=(1.35, 1.35+1.5),
        xticks=np.arange(1.4, 2.9, 0.2),
        yticks=np.arange(-1.677, -1.672+1e-6, 0.001),
    ),
}

for device, labels in zip(device_list, ["ab", "cd", "ef"]):
    run = runs[device]
    info = infos[device]
    fig, axs = plt.subplots(
        ncols=2,
        constrained_layout=True,
        figsize=info["figsize"],
        sharey=True,
    )
    plot_stage2_diagram(
        run["zbp_ds"],
        cutter_value=info["cutter_value"],
        zbp_cluster_numbers=info["zbp_cluster_numbers"],
        fig=fig,
        axs=axs,
        pct_boundary_shifts=info["pct_boundary_shifts"],
        description=f"TGP-{device}",
        deco_yfactor=info["deco_yfactor"],
    )
    for ax, label in zip(axs, labels):
        add_subfig_label(ax, label)
        ax.set_xticks(info["xticks"])
        ax.set_xlim(info["xlim"])
    axs[0].set_yticks(info["yticks"])
    plt.savefig(f"device{device}_tgp2_tuneup.pdf", dpi=fig.dpi, bbox_inches='tight', pad_inches=0.01)
    plt.show()
```

```python

```
