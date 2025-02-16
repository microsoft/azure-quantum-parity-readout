import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from circle_fit import taubinSVD
from scipy.optimize import curve_fit

from cq_conversion import CQConversionInput

from timeit import default_timer as timer

from datasets import CQ_CONV_INTERMEDIATE_FIGURE_FOLDER

from helpers import get_findex

def get_phase_accum(fs, calib, fd, ed, feval):
    cang = np.unwrap(np.angle(calib * np.exp(1j*ed*fs)))
    return cang[get_findex(fs, feval)] - cang[get_findex(fs, fd)]

def get_iq_center(calib, name):
    cutoff1, cutoff2 = 1000, 1200
    point_coordinates = [[*i] for i in zip(np.real(calib)[cutoff1:cutoff2], np.imag(calib)[cutoff1:cutoff2])]

    xc, yc, r, sigma = taubinSVD(point_coordinates)
    center = xc + 1j*yc

    plt.figure()
    _, ax = plt.subplots()
    plt.plot(np.real(calib), np.imag(calib))
    circle2 = plt.Circle((xc, yc), r, color='tab:orange', fill=False)
    ax.add_patch(circle2)

    plt.savefig(CQ_CONV_INTERMEDIATE_FIGURE_FOLDER / f"{name}_1b_fit_iq_center.png")
    print(f"Saved Center Fit {name}_1b_fit_iq_center.png")

    return center

def correct_off_resonance_drive(data, fs, calib, ed, f0, fd, name):
    """
        Rotate data around center of IQ circle drawn by calib by an amount equivalent to phase(S_11)[fd-f0].
    """

    start = timer()

    center = get_iq_center(calib, name)

    # correct for off-resonance drive
    fd_correction = get_phase_accum(fs, calib - center, fd, ed, f0)
    corrected_data = (data - center) * np.exp(1j*fd_correction) + center

    end = timer()

    print("Extracted Required Drive Frequency Correction of", fd_correction, ", time:", end-start)

    fig, ax = plt.subplots() ###
    plt.plot(np.real(calib), np.imag(calib))

    data_to_plot = data.reduce(np.mean, dim=set(data.dims) - {'time'}).to_array().compute()
    corrected_data_to_plot = corrected_data.reduce(np.mean, dim=set(corrected_data.dims) - {'time'}).to_array().compute()

    start, end = end, timer()
    print("Applied Correction to Data, time:", end-start)

    plt.scatter(np.real(data_to_plot), np.imag(data_to_plot), label="original data")
    plt.scatter(np.real(center), np.imag(center))
    plt.scatter(np.real(corrected_data_to_plot), np.imag(corrected_data_to_plot), label="corrected_data data")

    plt.legend()
    plt.savefig(CQ_CONV_INTERMEDIATE_FIGURE_FOLDER / f"{name}_1c_drive_freq_correction.png")
    start, end = end, timer()
    print(f"Saved Drive Frequency Correction Plot at {name}_1c_drive_freq_correction.png, time:", end-start)

    return corrected_data
