# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import xarray as xr
import numpy as np

from scipy.interpolate import RegularGridInterpolator, interpnd

from thermometry.coulomb_diamond import CoulombDiamond

import sys
sys.path.insert(1, '..')
from paths import SIMULATION_DATA_FOLDER # noqa: E402

class ThermometryModel:

    def __init__(self, model_function, V0s):
        self.model_function = model_function
        self.V0s = V0s

    def curve(self, *args, **kwargs):
        return self.model_function(*args, **kwargs)

class ComplexOSDThermometryModel(ThermometryModel):

    def __init__(self,
                 dataset_loc=SIMULATION_DATA_FOLDER / "thermometry_reference_simulated_dataset.h5",
                 coulomb_diamond=CoulombDiamond(Ec=50e-6, alpha=0.5),
                 V0s=np.linspace(-0.5, 0.5, 101)
                ):
        self.V0s = V0s

        self.coulomb_diamond = coulomb_diamond

        Cq_ref = xr.load_dataset(dataset_loc, engine='h5netcdf')

        Cq_ref["iCq"] = np.imag(Cq_ref.CQ)
        Cq_ref["Cq"]  = np.real(Cq_ref.CQ)

        Cq_ref = Cq_ref.drop_vars(["CQ"]).squeeze().rename({
            'γng': 'gamma',
            'temperature': 'temp',
            'ng': 'Ng'
        })

        Cq_ref = Cq_ref.sortby("gamma").sortby("temp")

        Cq_ref["Ng"] = (Cq_ref.Ng - 0.5)
        Cq_ref["Cq"] = Cq_ref["Cq"] * 1000
        Cq_ref["iCq"] = Cq_ref["iCq"] * 1000
        self.Cq_ref = Cq_ref

        # Cq_ref["Cq"].dims returns a tuple of dims with the correct dim ordering
        # Cq_ref.dims doesn't, returns a frozen set
        coords = tuple(Cq_ref.coords[i].to_numpy() for i in Cq_ref["Cq"].dims)

        self.itp = RegularGridInterpolator(coords, Cq_ref["Cq"].to_numpy() + 1j*Cq_ref["iCq"].to_numpy(),
                                      bounds_error=False, fill_value=0)

    def model_function(self, t0, temp, Γ, A=None):
        if A is None:
            A = 0
        ri_Cq = self.itp(interpnd._ndim_coords_from_arrays((t0, self.V0s, Γ, temp)))
        return (ri_Cq)
