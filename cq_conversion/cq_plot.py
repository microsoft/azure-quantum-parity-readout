# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Sequence
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from helpers import create_frequency_mask, lin_to_dB

from cq_conversion import interpolate_resonator, CQConversionInput

FREQUENCY_NAME = "frequency"
SIGNAL_NAME = "signal"

def make_optional_figure(
    ax: Axes | None = None, figsize: tuple[float, float] | None = None
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        assert ax is not None
    elif not isinstance(ax, Axes):
        raise ValueError("ax is not a matplotlib Axes")
    else:
        _fig = ax.get_figure()
        assert _fig is not None
        fig = _fig
    return fig, ax

def plot_iq(
    signal: NDArray,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    **plot_kwargs,
) -> tuple[Figure, Axes]:
    fig, ax = make_optional_figure(ax, figsize)

    color = plot_kwargs.pop("color", "red")
    ax.plot(np.real(signal), np.imag(signal), color=color, **plot_kwargs)
    ax.set_xlabel(r"Re($S_{11}$)")
    ax.set_ylabel(r"Im($S_{11}$)")
    return fig, ax

def plot_cq_interpolators(
    fs,
    calib,
    data,
    analysis_input: CQConversionInput,
    fig: Figure | None = None,
    axes: tuple[Axes, Axes] | None = None,
) -> tuple[Figure, tuple[Axes, ...]]:

    interpolator_ds = make_projection_interpolator_ds(
        fs=fs,
        corrected_cal=calib * analysis_input.amplitude_ratio,
        f0=analysis_input.f_expand,
        analysis_input=analysis_input,
    )
    plot_title = f"Gen. Projection Interpolator\n{analysis_input.name}"

    return plot_interpolator_with_zoom(
        fs=fs,
        calib=calib * analysis_input.amplitude_ratio,
        measurement_ds=data,
        interpolator_ds=interpolator_ds,
        analysis_input=analysis_input,
        suptitle=plot_title,
        fig=fig,
        axes=axes,
    )


def make_projection_interpolator_ds(
    fs: NDArray,
    corrected_cal: NDArray,
    f0: float,
    analysis_input: CQConversionInput
):
    interpolated_frequency, interpolated_signal = interpolate_resonator(
        fs,
        corrected_cal,
        f0,
        f_delta=analysis_input.fdelta,
        num_points=analysis_input.interpolated_num_points,
        gaussian_sigma=analysis_input.gaussian_sigma,
    )

    interpolator_ds = xr.Dataset(
        data_vars={SIGNAL_NAME: (FREQUENCY_NAME, interpolated_signal)},
        coords={FREQUENCY_NAME: interpolated_frequency},
    )
    return interpolator_ds


def scaled_margin(xs: NDArray[np.float64], margin: float):
    """Scale an input array by a margin value, remaing centered at the
    original center point"""
    x_min = float(np.nanmin(xs))
    x_max = float(np.nanmax(xs))
    x_mid = (x_min + x_max) / 2
    x_delta = x_max - x_mid
    margin_min = x_mid - margin * x_delta
    margin_max = x_mid + margin * x_delta
    return margin_min, margin_max


def plot_interpolator(
    fs: NDArray,
    calib: NDArray,
    measurement_ds: xr.Dataset,
    interpolator_ds: xr.Dataset,
    analysis_input: CQConversionInput,
    zoom_margin: float | None = None,
    include_origin: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    interp_freq_name: str | None = None,
    interp_param_name: str | None = None,
    v_scale: float | None = 1e-3,
    include_legend: bool = False,
):
    interp_freq_name = interp_freq_name or str(list(interpolator_ds.coords)[0])
    interp_param_name = interp_param_name or str(list(interpolator_ds.data_vars)[0])

    fig, ax = make_optional_figure(ax, figsize)

    flat_meas_data = measurement_ds[analysis_input.data_rf_parameter].values.flatten()
    cal_data = calib
    interp_data = interpolator_ds[interp_param_name].values
    interp_drive_point = (
        interpolator_ds[interp_param_name]
        .sel({interp_freq_name: analysis_input.f0}, method="nearest")
        .values
    )

    plot_iq(
        flat_meas_data,
        ax=ax,
        marker=".",
        markersize=0.1,
        lw=0,
        alpha=0.5,
        label="meas data",
    )
    plot_iq(cal_data, ax=ax, color="green", lw=0.8, label="cal data")
    plot_iq(interp_data, ax=ax, color="blue", label="interpolator")
    plot_iq(
        interp_drive_point,
        ax=ax,
        marker="+",
        markersize=10,
        color="blue",
        label="drive frequency",
    )

    if zoom_margin is not None:
        zoom_data: NDArray[np.complexfloating] = np.concatenate(
            (interp_data, flat_meas_data)
        )
        ax.set_xlim(scaled_margin(zoom_data.real, zoom_margin))
        ax.set_ylim(scaled_margin(zoom_data.imag, zoom_margin))

    if v_scale is not None:
        scale_order = np.log10(v_scale)
        ax.ticklabel_format(axis="x", scilimits=(scale_order, scale_order))
        ax.ticklabel_format(axis="y", scilimits=(scale_order, scale_order))

    if include_origin:
        ax.axvline(0, lw=0.5, ls=":", color="gray")
        ax.axhline(0, lw=0.5, ls=":", color="gray")

    if include_legend:
        ax.legend(loc="lower right")
    return fig, ax


def plot_interpolator_with_zoom(
    fs: NDArray,
    calib: NDArray,
    measurement_ds: xr.Dataset,
    interpolator_ds: xr.Dataset,
    analysis_input: CQConversionInput,
    suptitle: str | None = None,
    zoom_margin: float | None = 1.2,
    include_origin: bool = True,
    figsize: tuple[float, float] | None = None,
    interp_freq_name: str | None = None,
    interp_param_name: str | None = None,
    v_scale: float | None = 1e-3,
    fig: Figure | None = None,
    axes: tuple[Axes, Axes] | None = None,
) -> tuple[Figure, tuple[Axes, ...]]:
    if fig is None and axes is None:
        figsize = figsize or (8, 4)
        fig, axs = plt.subplots(1, 2, figsize=figsize, tight_layout=True, squeeze=False)
        plot_axes: tuple[Axes, ...] = tuple(axs.flatten())
    elif fig is not None and axes is not None:
        plot_axes = axes
    elif not (fig is not None and axes is not None):
        raise ValueError("Either both fig and axes must be supplied or neither.")

    plot_interpolator(
        fs=fs,
        calib=calib,
        measurement_ds=measurement_ds,
        interpolator_ds=interpolator_ds,
        analysis_input=analysis_input,
        include_origin=include_origin,
        ax=plot_axes[0],
        interp_freq_name=interp_freq_name,
        interp_param_name=interp_param_name,
        v_scale=v_scale,
    )
    plot_interpolator(
        fs=fs,
        calib=calib,
        measurement_ds=measurement_ds,
        interpolator_ds=interpolator_ds,
        analysis_input=analysis_input,
        zoom_margin=zoom_margin,
        include_origin=False,
        ax=plot_axes[1],
        interp_freq_name=interp_freq_name,
        interp_param_name=interp_param_name,
        v_scale=v_scale,
    )
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize="medium")

    plot_handles = [
        Line2D([], [], lw=0, marker=".", color="red", label="meas data"),
        Line2D([], [], lw=1, color="green", label="cal data"),
        Line2D([], [], lw=1, color="blue", label="interpolator"),
        Line2D([], [], lw=0, color="blue", marker="+", label="drive frequency"),
    ]
    fig.legend(
        handles=plot_handles,
        loc="upper right",
        fontsize="small",
        frameon=False,
    )
    return fig, plot_axes

def plot_magnitude(
    frequency: NDArray,
    signal: NDArray,
    in_dB: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    **plot_kwargs,
) -> tuple[Figure, Axes]:
    fig, ax = make_optional_figure(ax, figsize)
    color = plot_kwargs.pop("color", "b")

    plot_signal = lin_to_dB(signal) if in_dB else signal
    plot_units = "dB" if in_dB else "V"

    ax.plot(frequency, plot_signal, color=color, **plot_kwargs)
    ax.set_ylabel(r"$S_{11}$ " + f"magnitude [{plot_units}]")

    ax.set_xlabel("Frequency [Hz]")
    ax.ticklabel_format(axis="x", scilimits=(6, 6))
    return fig, ax

def plot_phase(
    frequency: NDArray,
    signal: NDArray,
    in_pi: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    **plot_kwargs,
) -> tuple[Figure, Axes]:
    fig, ax = make_optional_figure(ax, figsize)
    color = plot_kwargs.pop("color", "red")

    plot_units = "\u03c0" if in_pi else "rads"
    phase_divisor = np.pi if in_pi else 1
    plot_phase = np.unwrap(np.angle(signal)) / phase_divisor

    ax.plot(frequency, plot_phase, color=color, **plot_kwargs)
    ax.set_ylabel(r"$S_{11}$ " + f"phase [{plot_units}]")

    ax.set_xlabel("Frequency [Hz]")
    ax.ticklabel_format(axis="x", scilimits=(6, 6))

    return fig, ax

def resonance_extraction_plotting(
    frequency,
    calib,
    itp_freqs,
    itp_calib,
    itp_dS11,
    orthogonality,
    central_crossing,
    zero_crossings,
    max_index,
    f0_guess,
    f_delta,
    gaussian_sigma,
):
    if f0_guess is not None:
        fmask = create_frequency_mask(frequency, f0_guess - f_delta, f0_guess + f_delta)
        calib = calib[fmask]
        frequency = frequency[fmask]

    fig, ax = plt.subplots(3, 1, figsize=(7, 10), squeeze=False)
    ax = ax.flatten()

    plt.axes(ax[0])
    plt.title("Interpolated $S_{11}$ Data")
    axp = ax[0].twinx()
    plot_magnitude(frequency, calib, color="k", ax=ax[0])
    plot_magnitude(itp_freqs, itp_calib, ax=ax[0], color="red", alpha=0.8)

    plt.axes(axp)
    plt.plot([], [], color="k", label="Raw Mag Data")
    plot_phase(frequency, calib, ax=axp, color="blue", label="Raw Phase Data")
    plot_phase(
        itp_freqs,
        itp_calib,
        ax=axp,
        color="red",
        alpha=0.8,
        label=f"Interpolation w/ $\\sigma_{{gauss}}={gaussian_sigma}$",
    )
    plt.legend()

    plt.axes(ax[1])
    plt.title("Derivative of $S_{11}$")
    plt.plot(
        itp_freqs / 1e6,
        abs(itp_dS11),
        color="k",
        label=f"Extracted $f_0$ = {itp_freqs[max_index]/1e6:.2f} MHz",
    )
    plt.scatter(itp_freqs[max_index] / 1e6, abs(itp_dS11[max_index]), color="tab:green")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel(r"abs$(dS_{11}/d\omega)$")
    plt.legend()

    plt.axes(ax[2])
    plt.title("Derivative Orthogonality")
    plt.plot(itp_freqs / 1e6, orthogonality, color="k")

    if len(zero_crossings) > 0:
        plt.scatter(
            itp_freqs[zero_crossings] / 1e6,
            0 * zero_crossings + 0.03 * np.max(orthogonality),
            marker="v",
            color="tab:blue",
            label="Zero Crossings",
        )
        plt.scatter(
            itp_freqs[central_crossing] / 1e6,
            -0.03 * np.max(orthogonality),
            marker="^",
            color="tab:green",
            zorder=2,
            label=f"Extracted $f_0$ = {itp_freqs[central_crossing]/1e6:.2f} MHz",
        )

    # y=0 line crossing => resonance
    plt.axhline(0, color="k", alpha=0.2)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel(
        "$\\Re \\left\\langle \\dfrac{dS_{11}}{d\\omega}, \\dfrac{d^2S_{11}}{d\\omega^2} \\right\\rangle$"
    )
    plt.legend()

    plt.tight_layout()

    plt.savefig("temp3.png")
