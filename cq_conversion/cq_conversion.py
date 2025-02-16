# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from collections.abc import Callable
from enum import Enum, unique
from functools import partial
from pathlib import Path
from typing import Collection, NamedTuple, Optional

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.interpolate import LinearNDInterpolator, make_interp_spline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree

CQ_NAME = "Cq"
ICQ_NAME = "iCq"


@dataclass(frozen=True, kw_only=True)
class CQConversionInput:
    name: str

    data_path: str | Path
    calib_path: str | Path
    target_path: Optional[str | Path] = None
    elec_delay: float
    L_ind: float

    gaussian_sigma: int
    fdelta: float
    drive_frequency: float
    resonance_frequency: float
    isel: dict[str, int|Collection[int]]
    coarsen_time: Optional[int]

    # If driven off resonance and the bg is not too jumpy, we can correct the off-resonance
    # drive by rotating the data around the center of the IQ-circle by phase(S11[fd] - S11[f0]).
    correct_off_resonance_drive: bool = True

    safe_freq_shift: float = 200e3
    interpolated_num_points: int = 2_001
    data_rf_parameter:str = "Vrf_1"
    amplitude_ratio: float = 1

    @property
    def f_expand(self):
        """
        Frequency at which to do the projection method
        Usually the drive frequency, if the analysis tried to correct for off-resonance drive,
        then use the resonance frequency
        """
        return self.fd if not self.correct_off_resonance_drive else self.f0

    @property
    def f0(self):
        return self.resonance_frequency

    @property
    def fd(self):
        return self.drive_frequency

    @property
    def target_name(self):
        return self.data_path.replace("Vrf", "Cq") if self.target_path is None else self.target_path

complex_dtype_map = {
    "complex128": np.float64,
    "complex64": np.float32,
}

## HELPERS ##

def interpolate_resonator(
    res_freq: np.ndarray,
    res_data: np.ndarray,
    drive_frequency: float,
    f_delta: float = 0.5e6,
    num_points: int = 2001,
    gaussian_sigma: float = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    returns:
        Function generating measured resonator response
    """

    # Create smoothed interpolation functions
    filtered_data_real = gaussian_filter1d(res_data.real, gaussian_sigma)
    filtered_data_imag = gaussian_filter1d(res_data.imag, gaussian_sigma)
    complex_interp = make_interp_spline(
        res_freq,
        np.c_[filtered_data_real, filtered_data_imag],
        k=1,
    )
    interpolated_freq = np.linspace(
        drive_frequency - f_delta, drive_frequency + f_delta, num_points
    )
    interp_data_real, interp_data_imag = complex_interp(interpolated_freq).T

    return interpolated_freq, interp_data_real + 1j * interp_data_imag

## CQ Projection Method ##

def Cq_core_transformation_projection(
    xr_array: xr.DataArray, transform_function: Callable, L: float, f0: float
) -> xr.Dataset:
    inverted_data: xr.DataArray = xr.apply_ufunc(
        transform_function,
        xr_array,
        dask="parallelized",
        output_dtypes=[xr_array.dtype],
    )
    Delta_f0_extracted = inverted_data.real
    Delta_ki_extracted = inverted_data.imag

    Cp = 1 / (L * (2 * np.pi * f0) ** 2)

    Cq = -1e18 * 2 * Cp * Delta_f0_extracted / f0
    iCq = 1e18 * Cp * Delta_ki_extracted / f0

    Cq.name = CQ_NAME
    Cq.attrs["unit"] = "aF"
    iCq.name = ICQ_NAME
    iCq.attrs["unit"] = "aF"

    return xr.Dataset({da.name: da for da in [Cq, iCq] if da is not None})


def complex_frequency_shift_transform(
    xr_array: npt.NDArray[np.complexfloating],
    interpolated_freq: npt.NDArray[np.floating],
    data_interpolator: KDTree,
    meas_freq: float,
    safe_freq_shift: float,
) -> np.ndarray:
    r"""
    Interpolator for complex projection method

    Args:
        xr_array : Data to convert
        interpolated_freq: Ref. freqs
        data_interpolator: Lookup tree to find the nearest points in the interpolator
        meas_freq: Measurement Frequency
        safe_freq_shift: The largest freq shift which is much less than a linewidth.  Should be about \kappa/50.
    Returns:
        Data array of the extracted frequency shifts
    """
    # Ensure we are working with np arrays so we can create views into the array
    if isinstance(xr_array, xr.DataArray):
        xr_array = xr_array.values
    # Extract the complex datatypes of the interpolated arrays
    xr_type = xr_array.dtype

    # rf signal of reference trace at the measurement frequency
    if interpolated_freq[0] > meas_freq or interpolated_freq[-1] < meas_freq:
        raise ValueError(
            "Interpolation range is too small.  Increase interpolated_f_delta kwarg."
        )

    meas_ind = np.searchsorted(interpolated_freq, meas_freq)
    S11_res = data_interpolator.data[meas_ind]

    interp_mean = np.mean(
        [data_interpolator.data[0], data_interpolator.data[-1]], axis=0
    )

    # Convert shift in radial direction to change in the loss rate
    safe_ind = np.searchsorted(interpolated_freq, meas_freq + safe_freq_shift)
    IQ2freq = safe_freq_shift / np.linalg.norm(
        data_interpolator.data[safe_ind] - S11_res
    )

    # Ensure the array is C-Contiguous in memory so that views will work
    xr_array = np.ascontiguousarray(xr_array)

    # Create a view into the dataset with floats instead of complex values
    xr_float_view = xr_array.view(complex_dtype_map[xr_type.name]).reshape(
        xr_array.shape + (2,)
    )
    # And create an output array, trying to minimize the number of intermediate views
    result = np.zeros_like(xr_float_view, dtype=complex_dtype_map[xr_type.name])

    # Calculate the nearest indices
    mask = np.isnan(xr_float_view)
    xr_float_view[mask] = 0

    _, nearest_ind = data_interpolator.query(xr_float_view)

    nearest_ref = data_interpolator.data[nearest_ind]
    nearest_ref[mask] = 0

    res = np.where(mask[..., 0], np.nan, interpolated_freq[nearest_ind])

    # This is the frequency shift
    result[..., 0] = meas_freq - res

    # To check if a radial shift has moved 'value' radially inwards or outwards,
    # we check whether the point is closer to the mean than the interpolated point.
    # If it is, multiply by -1
    mirror = 2 * nearest_ref - xr_float_view
    sign = np.sign(
        np.linalg.norm(mirror - interp_mean, axis=-1)
        - np.linalg.norm(xr_float_view - interp_mean, axis=-1)
    ).astype(np.int8)

    # Calculate the distance from the point to the reference curve,
    # and from that calculate the kappa_i change
    result[..., 1] = (
        2 * IQ2freq * np.linalg.norm(nearest_ref - xr_float_view, axis=-1) * sign
    )

    return result.view(xr_type)[..., 0]

def process_CqProjection(
    fs: np.ndarray,
    calib: np.ndarray,
    data: xr.Dataset,
    analysis_input: CQConversionInput,
) -> xr.Dataset:

    # Create interpolated dataset to normalize and filter the measured resonator response
    interpolated_freq, interpolated_data = interpolate_resonator(
        fs,
        calib * analysis_input.amplitude_ratio,
        analysis_input.f_expand,
        f_delta=analysis_input.fdelta,
        num_points=analysis_input.interpolated_num_points,
        gaussian_sigma=analysis_input.gaussian_sigma,
    )

    # Create a lookup tree for the resonator response. This returns the nearest index in the interpolated resonator
    interp_type = interpolated_data.dtype.name
    interpolated_data_float_view = interpolated_data.view(
        complex_dtype_map[interp_type]
    ).reshape(interpolated_data.shape + (2,))

    interpolator_lookup = KDTree(interpolated_data_float_view)

    interpolator = partial(
        complex_frequency_shift_transform,
        interpolated_freq=interpolated_freq,
        data_interpolator=interpolator_lookup,
        meas_freq=analysis_input.f_expand,
        safe_freq_shift=analysis_input.safe_freq_shift,
    )

    Cq = Cq_core_transformation_projection(
        data[analysis_input.data_rf_parameter],
        interpolator,
        analysis_input.L_ind,
        analysis_input.f_expand,
    )
    return Cq
