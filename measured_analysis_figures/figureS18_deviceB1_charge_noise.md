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
import xarray as xr
from uncertainties import unumpy

from matplotlib import pyplot as plt
import matplotlib as mpl

from analysis_code.common import (
    add_subfig_label,
    RAW_DATA_FOLDER,
)

from analysis_code.charge_noise import pt_full_analysis, VtoS
```

# Fig.S18


The script reproduces fig.S18 (charge noise measurement).

The list of dataset variables:
| dataset variable | variable in the paper |
| ---------------- | --------------------- |
| `Vrf`            |                       |
| `V_qd_1`         | $V_\mathrm{QD1}$      |


```python
ds = xr.load_dataset(
    RAW_DATA_FOLDER / "charge_noise/charge_noise_Vrf.h5",
    engine="h5netcdf",
    invalid_netcdf=True,
)
```

```python
pt_plunger_window = [502.5e-3, 503.5e-3] # Should be approximately n_g wide
invert = 1
time_window = [100, 700]

# Gate lever arms
alpha_1 = 0.46
alpha_2 = 0.45
alpha_3 = 0.48

# Dot charging energies in ueV
Ec_1 = 140
Ec_2 = 45
Ec_3 = 100
```

```python
alpha, alpha_err, plunger, time, pt_quad, peaks, freq_bins, psd, freq_bins_fit, psd_fit = pt_full_analysis(
    ds,
    pt_plunger_window,
    time_window,
    invert,
)

V0_1 = unumpy.uarray(alpha, alpha_err)
S0_1 = VtoS(alpha_1, Ec_1, Ec_2, V0_1)
```

```python
fig, ax = plt.subplots(1, 2, figsize=(6, 3), width_ratios=(1.2, 1))

pcm = ax[0].pcolormesh(
    1e3*plunger,
    time.values - time.values[0],
    1e6*pt_quad,
    cmap=mpl.colors.LinearSegmentedColormap.from_list("Vrf", ["w", "tab:orange"]),
    linewidth=0,
    rasterized=True,
)
pcm.set_edgecolor("face")
ax[0].plot(1e3*peaks, time.values - time.values[0], color="k")
ax[0].set_xlim(502.80, 503.11)
ax[0].set_ylabel("Time [s]")
ax[0].set_xlabel("$V_\\mathrm{QD1}$ [mV]")
cbar = plt.colorbar(pcm, ax = ax[0])
cbar.set_label("rf voltage [$\mu$V]")
add_subfig_label(ax[0], "a")

ax[1].loglog(freq_bins, psd, "o", alpha=0.4)
ax[1].loglog(freq_bins_fit, psd_fit, color="k")
ax[1].set_xlabel("Frequency [Hz]")
ax[1].set_ylabel("$S_{VV}(\omega)$ [V$^2$/Hz]")
add_subfig_label(ax[1], "b")

plt.tight_layout()
plt.savefig("charge_noise.pdf", dpi=fig.dpi, bbox_inches="tight", pad_inches=0.01)
```
