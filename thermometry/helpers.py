# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import xarray as xr

def _reshape(v, s):
    if isinstance(s, list):
        vnew = []
        istart = 0
        for _s in s:
            iend = istart + _s
            vnew.append(_reshape(v[istart:iend], _s))
            iend = istart
        return vnew
    return np.reshape(v, s)

def find_extent(ds, gate_x="V_DG2", gate_y="V_QD1"):
    x = ds[gate_x].to_numpy()
    y = ds[gate_y].to_numpy()

    extent = [np.amin(x), np.amax(x), np.amin(y), np.amax(y)]

    return (x, y, extent)
