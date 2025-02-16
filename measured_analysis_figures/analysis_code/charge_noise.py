# +
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# -

import numpy as np
import xarray as xr
from scipy.constants import pi
from scipy.signal import savgol_filter, welch
from scipy.optimize import curve_fit


def decompose_signal(z):
    z = np.ravel(z)
    z = z[~np.isnan(z)]
    x = np.real(z)
    y = np.imag(z)
    cov = np.cov([x, y])
    w, v = np.linalg.eig(cov)
    main_i = np.argmax(w)
    other_i = 1 - main_i
    main_v = v[:, main_i]
    origin = np.array([np.median(x), np.median(y)])
    angle = np.arctan2(main_v[1], main_v[0])
    signal = np.sqrt(np.abs(w[main_i]))
    noise = np.sqrt(np.abs(w[other_i]))
    if np.abs(angle) >= pi / 2:  # Pick the rotation angle closest to zero:
        angle += pi
        angle = angle % (2 * pi)
    return x, y, main_v, origin, angle, signal, noise


def fit_quadrature_to_array(z):
    """Fit a line to a complex signal scattered in the plane.

    Args:
        z (array-like): Complex signal.

    Returns:
        angle: The angle of the signal in radians.
        snr: The signal to noise ratio of the signal. This is calculated as
        the ratio of the smaller covariance axis of the data (noise) to the
        larger covariance axis (signal)."""
    x, y, main_v, origin, angle, signal, noise = decompose_signal(z)
    snr = signal / noise
    return angle, signal, noise


def fit_quadrature_separate(dataset):
    """Fit a line to a complex signal scattered in the plane.
    This is a wrapper to fit_quadrature_to_array."""
    dims = list(dataset.dims)
    angle, signal, noise = xr.apply_ufunc(
        fit_quadrature_to_array,
        dataset,
        input_core_dims=[dims],
        output_core_dims=[[], [], []],
        exclude_dims=set(dims),
    )
    return angle, signal, noise


def fit_quadrature(dataset):
    """Fit a line to a complex signal scattered in the plane.
    This is a wrapper to fit_quadrature_to_array."""
    angle, signal, noise = fit_quadrature_separate(dataset)
    snr = signal / noise
    return angle, snr


def find_quadrature_and_project(dataset, snr_threshold=3.0):
    """Find the angle to maximize signal and project along that quadrature."""
    angle, snr = fit_quadrature(dataset)
    angle = angle.where(snr >= snr_threshold, 0.0)  # If fit is bad, do not rotate
    rotated_dataset = dataset / np.exp(1.0j * angle)
    projected_dataset = rotated_dataset.real
    projected_dataset.attrs = dataset.attrs
    return projected_dataset


def gaussian(x, amplitude, mean, stddev, offset):
    return amplitude * np.exp(-(((x - mean) / (2 * stddev)) ** 2)) + offset


def do_gaussian_fit(x_data, y_data):
    mean_guess = x_data[np.argmax(y_data)]
    amp_guess = np.max(np.abs(y_data)) - np.min(np.abs(y_data))
    stddev_guess = (x_data[-1] - x_data[0]) / 40
    offset_guess = np.mean(y_data)

    popt, _ = curve_fit(
        gaussian,
        x_data,
        y_data,
        p0=[amp_guess, mean_guess, stddev_guess, offset_guess],
    )
    return popt


def find_peaks_2d(array_2d, plunger):
    peak_locations = []
    for row in array_2d:
        popt = do_gaussian_fit(plunger, row)
        peak_locations.append(popt[1])
    return np.array(peak_locations)


def smoooth_and_max(array_2d, plunger):
    array_smoothed = savgol_filter(array_2d, 6, polyorder=2)
    peak_inds = np.argmax(array_smoothed, axis=1)
    return plunger[peak_inds], array_smoothed


def crop_plunger(plunger_window, plunger, rf, pt=True):
    if plunger_window is not None:
        plunger_mask = np.logical_and(
            plunger.values > plunger_window[0],
            plunger.values < plunger_window[1],
        )
        plunger_masked = plunger[plunger_mask].values
        if pt:
            rf_masked = rf[:, plunger_mask]
        else:
            rf_masked = rf[plunger_mask, :]
    else:
        rf_masked = rf
        plunger_masked = plunger.values
    return rf_masked, plunger_masked


def crop_time(rf, timevec, time_window):
    time_mask = np.logical_and(
        timevec.values > time_window[0],
        timevec.values < time_window[1],
    )
    time_masked = timevec[time_mask]
    rf_masked = rf[time_mask, :]
    return rf_masked, time_masked


def pt_analysis(ds, plunger_window, time_window=None, invert=1):
    """Peak tracking analysis.
    Finds the Coulomb peaks in a 2d map of plunger voltage vs time,
    creating a time-trace of plunger voltages at which the Coulomb peaks occur.
    Then Fourier transforms this trace to find the power spectral density of the voltage fluctuations on the plunger.
    """

    time = [ds[coord] for coord in ds.coords][0]
    plunger = [ds[coord] for coord in ds.coords][1]
    rf = [ds[data_var] for data_var in ds.data_vars][0]

    time = time - time.values[0]
    rf_masked, plunger_masked = crop_plunger(plunger_window, plunger, rf)
    rf_masked, time = crop_time(rf_masked, time, time_window)
    pt_quad = find_quadrature_and_project(rf_masked)

    peaks, pt_quad_smoothed = smoooth_and_max(invert * pt_quad, plunger_masked)
    peaks = find_peaks_2d(pt_quad_smoothed, plunger_masked)

    delta_t = time.values[1] - time.values[0]
    freq_bins_pt, PSD_pt = welch(peaks, 1 / delta_t, noverlap=None, nperseg=256)
    PSD_pt = 0.5 * PSD_pt
    jumps = count_jumps(plunger_masked, peaks)

    return plunger_masked, time, pt_quad, peaks, PSD_pt, freq_bins_pt, jumps


def model(x, alpha):  # for fitting 1/f noise
    """Model for fitting 1/f noise in log space."""
    return 2 * np.log(alpha) - x


def count_jumps(plunger, peaks, threshold=0.1):
    """Counts the number of plunger voltage jumps which are larger than a given threshold."""
    v_thresh = threshold * (plunger[-1] - plunger[0])
    return np.sum(abs(np.diff(peaks)) > v_thresh)


def pt_full_analysis(ds, pt_plunger_window, time_window, invert):
    """Perform full peak-tracking analysis.
    Run peak-tracking, and then fit the resulting power spectral density."""
    plunger_masked, time, pt_quad, peaks, PSD_pt, freq_bins_pt, jumps = pt_analysis(
        ds,
        pt_plunger_window,
        time_window,
        invert=invert,
    )

    freq_bins_fit = freq_bins_pt[freq_bins_pt > 0]
    y = PSD_pt[freq_bins_pt > 0]

    popt, pcov = curve_fit(
        model,
        np.log(2 * np.pi * freq_bins_fit),
        np.log(y),
        p0=[10 * 1e-12],
    )

    alpha = popt[0] * 1e6
    alpha_err = np.sqrt(pcov[0, 0]) * 1e6
    freq_bins = freq_bins_pt[1:]
    psd = 1e12 * PSD_pt[1:]

    psd_fit = 1e12 * np.exp(model(np.log(2 * np.pi * freq_bins_fit), popt))

    return (
        alpha,
        alpha_err,
        plunger_masked,
        time,
        pt_quad,
        peaks,
        freq_bins,
        psd,
        freq_bins_fit,
        psd_fit,
    )


def VtoS(alpha_s, Ec_1, Ec_2, V0):
    """Convert voltage fluctuations to energy fluctuations."""
    return (alpha_s**2 / ((Ec_1 / Ec_2) ** 0.5 + 1)) ** 0.5 * V0
