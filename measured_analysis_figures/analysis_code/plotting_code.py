# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Literal

import numpy as np
import sympy

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from analysis_code.common import (
    add_rectangle_horizontal,
    _set_panel_properties,
    add_subfig_label,
    add_field_arrow_text,
)
from analysis_code.timetrace_analysis import prepare_plot_dwells_bayesian
from scipy import stats

from lmfit import Model
import matplotlib.ticker as ticker


def plot_kurtosis(
    ax,
    kurtosis_da,
    x="B_perp",
    i_V_lin_qd=None,
    vmin=-2.0,
    vmax=0,
    V_arrow=None,
    V_arrow_height=0.14,
    V_arrow_wpad=0.17,
    V_arrow_color="tab:red",
    cmap="Greys_r",
    panel_label="",
    cbar_kwargs={"label": r"$K(C_\mathrm{Q})$"},
    Bx_tick_base=2,
    show_xlabel=True,
    **kwargs,
):
    """
    Plot kurtosis data on the specified axis.

    Parameters:
    ax : Axis
        The axis to plot on.
    kurtosis_da : DataArray
        The kurtosis data array.
    x : str
        The x-axis label.
    i_V_lin_qd : int, optional
        Index for V_lin_qd.
    vmin : float
        Minimum value for color scale.
    vmax : float
        Maximum value for color scale.
    V_arrow : float, optional
        V_lin_qd value to add a arrow at.
    V_arrow_height : float
        Height of V_lin_qd arrow.
    V_arrow_wpad : float
        Width padding for V_lin_qd arrow.
    V_arrow_color : str
        Color of the V_lin_qd arrow.
    cmap : str
        Colormap.
    panel_label : str
        Label for the panel.
    cbar_kwargs : dict
        Keyword arguments for colorbar.
    Bx_tick_base : int
        Tick base for x-axis.
    show_xlabel : bool
        Whether to show x-axis label.
    """

    pcm = kurtosis_da.plot.pcolormesh(
        ax=ax,
        x=x,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        cbar_kwargs={"label": kurtosis_da.attrs["long_name"]},
        linewidth=0,
        rasterized=True,
    )
    pcm.set_edgecolor("face")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=Bx_tick_base))

    if i_V_lin_qd is not None:
        V_arrow = kurtosis_da["V_lin_qd"][i_V_lin_qd]

        add_rectangle_horizontal(
            ax, V_arrow, height=V_arrow_height, color=V_arrow_color, wpad=V_arrow_wpad
        )
    plt.setp(ax.get_xticklabels(), visible=show_xlabel)
    ax.get_xaxis().get_label().set_visible(show_xlabel)

    add_subfig_label(ax, panel_label)


def plot_histogram(
    ax,
    counts_da,
    vmin=0,
    vmax=100,
    Cq_margin=0.2,
    panel_label="h",
    Bx_tick_base=2,
    Cq_tick_base=250,
    add_colorbar=True,
    show_xlabel=True,
    **kwargs,
):
    """
    Plot histogram data on the specified axis.

    Parameters:
    ax : Axis
        The axis to plot on.
    counts_da : DataArray
        The counts data array.
    vmin : float
        Minimum value for color scale.
    vmax : float
        Maximum value for color scale.
    Cq_margin : float
        Margin for Cq.
    panel_label : str
        Label for the panel.
    Bx_tick_base : int
        Tick base for x-axis.
    Cq_tick_base : int
        Tick base for y-axis.
    add_colorbar : bool
        Whether to add a colorbar.
    show_xlabel : bool
        Whether to show x-axis label.
    """

    pcm = counts_da.plot.pcolormesh(
        x="B_perp",
        cmap="Greys",
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=add_colorbar,
        linewidth=0,
        rasterized=True,
    )
    pcm.set_edgecolor("face")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=Bx_tick_base))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=Cq_tick_base))
    plt.setp(ax.get_xticklabels(), visible=show_xlabel)
    ax.get_xaxis().get_label().set_visible(show_xlabel)

    add_subfig_label(ax, panel_label)


def _calc_bins(data, bin_size):
    """
    Calculate bin edges based on data and bin size.

    Parameters:
    data : array-like
        Data to calculate bins for.
    bin_size : float
        The size of each bin.

    Returns:
    array
        Bin edges.
    """
    return np.arange(data.min(), data.max() + bin_size, bin_size)


def plot_timetrace(
    trace_da, ax, show_xlabel=False, panel_label="", Cq_tick_base=250, **kwargs
):
    """
    Plot Cq(t) time trace data on the specified axis.

    Parameters:
    trace_da : DataArray
        The time trace data array.
    ax : Axis
        The axis to plot on.
    show_xlabel : bool
        Whether to show x-axis label.
    panel_label : str
        Label for the panel.
    Cq_tick_base : int
        Tick base for y-axis.

    Returns:
    None
    """
    if not kwargs:
        kwargs = {"color": "tab:blue", "linewidth": 1}
    trace_da["Cq"].plot(ax=ax, x="time", **kwargs)
    plt.setp(ax.get_xticklabels(), visible=show_xlabel)
    ax.get_xaxis().get_label().set_visible(show_xlabel)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=Cq_tick_base))
    plt.setp(ax.get_xticklabels(), visible=show_xlabel)
    add_subfig_label(ax, panel_label)


def plot_digital_timetrace(digital_traces_da, mean1_da, mean2_da, ax):
    """
    Plot digital time trace data on the specified axis.

    Parameters:
    digital_traces_da : DataArray
        The digital traces data array.
    mean1_da : DataArray
        The mean data array for state 1.
    mean2_da : DataArray
        The mean data array for state 2.
    ax : Axis
        The axis to plot on.

    Returns:
    None
    """
    ax2 = ax.twinx()

    digital_traces_da.plot(ax=ax2, color="k", alpha=1, ls="-", lw=1, zorder=100)

    ax.axhline(mean1_da, color="tab:blue", ls="-.", alpha=1, zorder=150)
    ax.axhline(mean2_da, color="tab:red", ls="-.", alpha=1, zorder=150)

    ax2.set_yticks([])
    ax.text(
        1.025,
        0.02,
        "Odd",
        ha="left",
        va="bottom",
        rotation="vertical",
        transform=ax.transAxes,
    )
    ax.text(
        1.025,
        1.0,
        "Even",
        ha="left",
        va="top",
        rotation="vertical",
        transform=ax.transAxes,
    )


def plot_IQ_blobs(trace_da, ax, bin_size=30, add_colorbar=False, panel_label=""):
    """
    Plot 2d histogram of complex Cq data on the specified axis.

    Parameters:
    trace_da : DataArray
        The trace data array.
    ax : Axis
        The axis to plot on.
    bin_size : int
        The size of each bin.
    add_colorbar : bool
        Whether to add a colorbar.
    panel_label : str
        Label for the panel.

    """

    h = ax.hist2d(
        trace_da["iCq"],
        trace_da["Cq"],
        bins=(
            _calc_bins(trace_da["iCq"], bin_size),
            _calc_bins(trace_da["Cq"], bin_size),
        ),
        cmap="gist_heat_r",
    )

    ax.set_xlabel(r"$\mathrm{Im}\,{\tilde C}_\mathrm{Q}$ [aF]")

    plt.setp(ax.get_yticklabels(), visible=False)

    if add_colorbar:
        ax.set_xticklabels([])
        ax.set_xlabel("")

        box = ax.get_position()
        fig = ax.get_figure()
        cax = fig.add_axes(
            [
                box.x0 + 0.3 * box.width,
                box.y0 + box.height + 0.01,
                0.7 * box.width,
                0.02,
            ]
        )
        cb = fig.colorbar(h[3], cax=cax, orientation="horizontal", location="top")
        cb.set_ticks([0, h[0].max()], labels=["0", "Max"])
        for l in cax.xaxis.get_majorticklabels():
            l.set_y(0.5)
        cb.set_label("Count  ", labelpad=-4)
    add_subfig_label(ax, panel_label)


def _do_gaussian_fit(bin_centers, count, n_components):
    """
    Perform a Gaussian fit on the data.

    Parameters:
    bin_centers : array-like
        The centers of the bins.
    count : array-like
        The counts in each bin.
    n_components : int
        The number of Gaussian components.

    Returns:
    result : ModelResult
        The fit result.
    """

    def _gaussian(x, amp, center, sigma):
        return amp * np.exp(-((x - center) ** 2) / (2 * sigma**2))

    model = Model(_gaussian, prefix="p1_", nan_policy="omit")
    for i in range(1, n_components):
        model = model + Model(_gaussian, prefix=f"p{i+1}_", nan_policy="omit")

    params = model.make_params()
    bin_range = np.ptp(bin_centers)
    for i in range(0, n_components):
        prefix = f"p{i+1}"
        params[f"{prefix}_amp"].set(value=np.max(count), min=0)
        params[f"{prefix}_center"].set(
            value=np.mean(bin_centers) - (0.5 - i) * bin_range / 2,
            min=np.min(bin_centers),
            max=np.max(bin_centers),
        )
        params[f"{prefix}_sigma"].set(value=bin_range / 5, min=0, max=bin_range)

    result = model.fit(
        count,
        params,
        x=bin_centers,
        nan_policy="omit",
    )

    return result


def _plot_gaussian_fit(result, bin_centers, count, n_components, ax):
    """
    Plot the Gaussian fit on the data.

    Parameters:
    result : ModelResult
        The fit result.
    bin_centers : array-like
        The centers of the bins.
    count : array-like
        The counts in each bin.
    n_components : int
        The number of Gaussian components.
    ax : Axis
        The axis to plot on.
    """

    ax.plot(result.best_fit, bin_centers, c="k", ls="--")

    if n_components == 2:
        c_pos = 0.95 * np.max(count)
        p1_center = result.params[f"p1_center"].value
        p1_sigma = result.params[f"p1_sigma"].value
        p2_center = result.params[f"p2_center"].value
        p2_sigma = result.params[f"p2_sigma"].value

        delta = np.abs(p1_center - p2_center)
        middle = (p1_center + p2_center) / 2
        for center, sigma in zip((p1_center, p2_center), (p1_sigma, p2_sigma)):

            ax.add_patch(
                FancyArrowPatch(
                    (c_pos, center - sigma),
                    (c_pos, center + sigma),
                    arrowstyle="<|-|>",
                    mutation_scale=7,
                    color="k",
                    shrinkA=0,
                    shrinkB=0,
                )
            )
            shift, va, i = (
                (sigma, "bottom", 2) if center > middle else (-1 * sigma, "top", 1)
            )
            ax.text(
                c_pos + 2,
                center + shift,
                f"$2\sigma_{i}$",
                ha="right",
                va=va,
            )
        c_pos = 0.75 * np.max(count)
        ax.add_patch(
            FancyArrowPatch(
                (c_pos, p1_center),
                (c_pos, p2_center),
                arrowstyle="<|-|>",
                mutation_scale=7,
                color="k",
                shrinkA=0,
                shrinkB=0,
            )
        )
        ax.text(
            c_pos - 1,
            middle,
            "$\delta$",
            ha="right",
            va="center",
        )

        SNR = delta / (p1_sigma + p2_sigma)

        print(f"{delta= :1.0f} {p1_sigma= :1.0f} {p2_sigma= :1.0f} {SNR= :1.2f}")


def plot_hist_with_fit(
    trace_da,
    ax,
    bin_size=30,
    n_components=1,
    plot_fit=True,
    show_xlabel=True,
    panel_label="",
):
    """
    Plot histogram with a Gaussian fit.

    Parameters:
    trace_da : DataArray
        The trace data array.
    ax : Axis
        The axis to plot on.
    bin_size : int
        The size of each bin.
    n_components : int
        The number of Gaussian components.
    plot_fit : bool
        Whether to plot the Gaussian fit.
    show_xlabel : bool
        Whether to show x-axis label.
    panel_label : str
        Label for the panel.
    """

    count, bin_edges, _ = ax.hist(
        trace_da.Cq,
        orientation="horizontal",
        bins=_calc_bins(trace_da["Cq"], bin_size),
        color="tab:orange",
        ec="k",
        lw=0.3,
        alpha=0.9,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.set_xlim(0, np.max(count))
    ax.set_ylim(np.min(bin_centers) - 3 * bin_size, np.max(bin_centers) + 3 * bin_size)
    ax.set_xticks([0, np.max(count)], labels=["0", "Max"])
    ax.set_xlabel("Count" if show_xlabel else "")
    plt.setp(ax.get_xticklabels(), visible=show_xlabel)

    plt.setp(ax.get_yticklabels(), visible=False)

    add_subfig_label(ax, panel_label)
    if plot_fit:
        result = _do_gaussian_fit(bin_centers, count, n_components)
        _plot_gaussian_fit(result, bin_centers, count, n_components, ax)


def plot_dwells_and_splitting(
    ax,
    bias,
    dwells_ups,
    dwells_downs,
    splittings,
    panel_label="",
    ylim=(0, 1.5),
    **kwargs,
):
    """
    Plot dwell times and \Delta Cq splitting on the specified axis.

    Parameters:
    ax : Axis
        The axis to plot on.
    bias : array-like
        The bias values.
    dwells_ups : array-like
        The dwell times for upward transitions.
    dwells_downs : array-like
        The dwell times for downward transitions.
    splittings : array-like
        The splittings.
    panel_label : str
        Label for the panel.
    ylim : tuple
        Y-axis limits.
    """
    ax.scatter(
        bias * 1e6,
        dwells_ups,
        marker="^",
        ls="",
        alpha=(splittings / np.max(splittings)),
        label=r"$\tau_{\!\uparrow}$",
        color="tab:red",
    )
    ax.scatter(
        bias * 1e6,
        dwells_downs,
        marker="v",
        ls="",
        alpha=(splittings / np.max(splittings)),
        label=r"$\tau_{\!\downarrow}$",
        color="tab:blue",
    )
    ax.set_ylim(ylim)
    twax = ax.twinx()
    twax.plot(
        bias * 1e6,
        splittings,
        color="k",
        ls="",
        marker=".",
        label=r"$\Delta C_\mathrm{Q}$",
    )
    twax.set_ylabel(r"$\Delta C_\mathrm{Q}$ [aF]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twax.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2)
    kwargs.update(dict(xlabel="$V_3$ [μV]", ylabel="$\\tau_\\mathrm{RTS}$ [ms]"))
    add_subfig_label(ax, panel_label)
    _set_panel_properties(ax, **kwargs)


def plot_dwells_bayesian(
    ax,
    dwells,
    model_fit_res=None,
    bin_scale=15,
    color="k",
    s=3.0,
    label=r"\tau",
    plot_interval=False,
):
    """
    Plot dwell times distribution and results of fit to Bayesian model on the specified axis.

    Parameters:
    ax : Axis
        The axis to plot on.
    dwells : array-like
        The dwell times.
    model_fit_res : dict, optional
        The model fit results.
    bin_scale : int
        Scale for the bins.
    color : str
        Color for the plot.
    s : float
        Size of the scatter points.
    label : str
        Label for the dwells.
    plot_interval : bool
        Whether to plot the interval.

    Returns:
    mean : float
        The mean dwell time.
    """

    if model_fit_res is None:
        model_fit_res = prepare_plot_dwells_bayesian(dwells)

    n_dwells = len(dwells)
    counts, bins_e = np.histogram(
        dwells,
        bins=int(n_dwells / bin_scale),
        density=False,
    )
    bins = (bins_e[:-1] + bins_e[1:]) / 2

    mean = model_fit_res["mean"]

    w_plt = np.where(counts > 0.1)
    pdf_sample_points = stats.expon(scale=mean).pdf(bins[w_plt])
    pdf_scale = 1 / (np.sum(pdf_sample_points) / n_dwells)

    ax.scatter(bins, counts, label="", color=color, s=s)

    sig = (model_fit_res["hdi_plus"] - model_fit_res["hdi_minus"]) / 2

    ax.plot(
        bins[w_plt],
        stats.expon(scale=mean).pdf(bins[w_plt]) * pdf_scale,
        color=color,
        label=rf"${label} = {mean :1.2f}\pm{sig :1.2f}\,$ms",
    )

    if plot_interval:

        def sig_figs(val, sig_figs):
            digits = np.ceil(np.log10(val))
            factor = 10 ** (digits - sig_figs)
            return np.round(val / factor) * factor

        hdi_plus = model_fit_res["hdi_plus"]
        hdi_minus = model_fit_res["hdi_minus"]
        gamma = 1e3 / model_fit_res["mean"]
        rounded_gamma = int(sig_figs(gamma, 3))

        sig = (hdi_plus - hdi_minus) / 2

        ax.fill_between(
            bins,
            stats.expon(scale=hdi_minus).pdf(bins) * pdf_scale,
            stats.expon(scale=hdi_plus).pdf(bins) * pdf_scale,
            color="tab:blue",
            alpha=0.3,
        )

        gamma_low, gamma_high = 1 / hdi_plus - gamma, 1 / hdi_minus - gamma
        gamma_pm = (gamma_high - gamma_low) / 2
        gamma_pm = int(sig_figs(1e3 * gamma_pm, 1))

        ax.text(
            0.9,
            0.9,
            "$Γ_{EO} = " + str(rounded_gamma) + "\\pm" + str(gamma_pm) + "\,{\\rm Hz}$",
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Dwell time [ms]")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(0.5, 2 * np.max(counts))

    return mean


def plot_up_down_dwells_bayesian(
    dwells_up, dwells_up_fit, dwells_down, dwells_down_fit, ax, panel_label=""
):
    """
    Plot Bayesian fit results for up and down dwell times on the specified axis.

    Parameters:
    dwells_up : array-like
        The dwell times for upward transitions.
    dwells_up_fit : dict
        The fit results for upward transitions.
    dwells_down : array-like
        The dwell times for downward transitions.
    dwells_down_fit : dict
        The fit results for downward transitions.
    ax : Axis
        The axis to plot on.
    panel_label : str
        Label for the panel.
    """
    dwu = plot_dwells_bayesian(
        ax,
        dwells_up,
        model_fit_res=dwells_up_fit,
        label=r"\tau_{\!\uparrow}",
        color="tab:red",
        s=8,
    )
    dwd = plot_dwells_bayesian(
        ax,
        dwells_down,
        model_fit_res=dwells_down_fit,
        label=r"\tau_{\!\downarrow}",
        color="tab:blue",
        s=8,
    )

    ax.legend(fontsize=8, loc=1)
    ax.set_ylabel("Count", labelpad=-0.05)
    add_subfig_label(ax, panel_label)


def plot_B_arrows(labels, pos_dss, ax):
    """
    Plot field arrows with labels on the specified axis.

    Parameters:
    labels : list of str
        Labels for the arrows.
    pos_dss : list of DataArray
        Data arrays containing the positions for the arrows.
    ax : Axis
        The axis to plot on.
    """
    B_poss = [pos_ds.B_perp.values for pos_ds in pos_dss]

    sides = ["left", "right"]
    if B_poss[0] < B_poss[1]:
        sides = sides[::-1]
    for label, side, B_pos in zip(labels, sides, B_poss):
        add_field_arrow_text(
            ax,
            B_pos,
            label,
            color="tab:blue",
            text_B_shift=0.15,
            ha=side,
        )


def plot_qdmzm(
    frame, ax, N_qd, data_type: Literal["Measured", "Simulated"], Cq_range=(0, 1000)
):
    """
    Plot QD MZM data on the specified axis.

    Parameters:
    frame : DataFrame
        The data frame containing the data to plot.
    ax : Axis
        The axis to plot on.
    N_qd : int
        Quantum dot index.
    data_type : {"Measured", "Simulated"}
        Data type to plot.
    Cq_range : tuple of float
        Range for Cq values.

    Returns:
    None
    """
    cbar_kwargs = dict(aspect=12, pad=0.06)
    pcm_kwargs = dict(linewidth=0, rasterized=True)

    if data_type == "Measured":
        frame[f"V_qd_{N_qd}_plunger_gate"] = frame[f"V_qd_{N_qd}_plunger_gate_abs"]
        y = "V_lin_qd"
        da_to_plot = frame.Cq
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.1))
    elif data_type == "Simulated":
        da_to_plot = frame.CQ.real
        y = "ng2"
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])

    da_to_plot.plot(
        ax=ax,
        y=y,
        cmap="Blues",
        vmin=Cq_range[0],
        vmax=Cq_range[1],
        **pcm_kwargs,
        cbar_kwargs=cbar_kwargs,
    )


def mix_histograms(counts_even, counts_odd):
    """
    Mix even and odd histograms into a RGB matrix representing the mixture.

    Parameters:
    counts_even : DataArray
        Data array of even counts.
    counts_odd : DataArray
        Data array of odd counts.

    Returns:
    np.ndarray
        RGB matrix representing the mixed histograms.
    """
    counts_even = counts_even.T.squeeze()
    counts_odd = counts_odd.T.squeeze()

    blue_array = (counts_even / np.max(counts_even)).to_numpy()
    red_array = (counts_odd / np.max(counts_odd)).to_numpy()

    green_array = 1 - blue_array - red_array
    green_array[green_array < 0] = 0
    return np.dstack((1 - red_array, green_array, 1 - blue_array))[::]


def plot_sim_histogram(counts_even, counts_odd, ax):
    """
    Plot simulated histogram for even and odd counts.

    Parameters:
    counts_even : DataArray
        Data array of even counts.
    counts_odd : DataArray
        Data array of odd counts.
    ax : Axis
        The axis to plot on.

    Returns:
    None
    """
    Z = mix_histograms(counts_even, counts_odd)
    yv = counts_even.bins
    xv = counts_odd.phi
    extent = [np.min(xv), np.max(xv), np.min(yv), np.max(yv)]
    ax.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=extent,
    )

    ax.set_xlabel(r"$\Phi$ [$h/2e]$")
    ax.set_ylabel(r"$\mathrm{Re}\,C_\mathrm{Q}$ [aF]")


def plot_eigenvalues(ax, ng, even_energies, even_hamiltonian, odd_hamiltonian):
    """
    Plot the energy eigenvalues for even and odd states.

    Parameters:
    ax : Axis
        The axis to plot on.
    ng : Symbol
        The gate charge symbol.
    even_energies : list of Expr
        Symbolic expressions for even state energies.
    even_hamiltonian : Matrix
        Symbolic Hamiltonian for even states.
    odd_hamiltonian : Matrix
        Symbolic Hamiltonian for odd states.
    """
    # The states of the even energies can be calculated directly from the symbolic expression
    ng_vals = np.linspace(0, 2, 251)
    ax.plot(
        ng_vals,
        sympy.lambdify(ng, even_energies[0])(ng_vals),
        c="tab:red",
        label="Even",
    )
    ax.plot(
        ng_vals,
        sympy.lambdify(ng, even_energies[1])(ng_vals),
        linestyle="--",
        c="tab:red",
    )

    # Numerically solve for eigenvalues for the odd parity
    odd_eigenvals = np.zeros((251, 3))
    odd_ham_lambda = sympy.lambdify(ng, odd_hamiltonian)
    for i, ng_val in enumerate(ng_vals):
        odd_eigenvals[i, :] = sorted(np.linalg.eigvals(odd_ham_lambda(ng_val)))
    ax.plot(ng_vals, odd_eigenvals[:, 0], c="tab:blue", label="Odd")
    ax.plot(ng_vals, odd_eigenvals[:, 1], linestyle="--", c="tab:blue")

    # Label axes
    ax.set_xlabel("$N_\mathrm{g}$")
    ax.set_ylabel("$E$ [$\mu$eV]")
    ax.set_xlim(0, 2)
    ax.set_ylim(None, 200)
    ax.legend(loc="upper right")


def plot_CBP_histogram(ds, bins, ax):
    """
    Plot histogram of CBP data with Gaussian fit.

    Parameters:
    ds : DataArray
        The data array containing CBP data.
    bins : int
        The number of bins.
    ax : Axis
        The axis to plot on.
    """
    count, bin_edges = np.histogram(ds.Cq, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.plot(bin_centers, count, marker="o", ls="", color="tab:orange")

    result = _do_gaussian_fit(bin_centers, count, n_components=2)
    print(
        "SNR=",
        np.abs(result.params["p2_center"] - result.params["p1_center"])
        / (result.params["p1_sigma"] + result.params["p2_sigma"]),
    )

    ax.plot(bin_centers, result.best_fit, c="k", ls="--")

    centers = tuple(result.params[f"p{i}_center"].value for i in [1, 2])
    centers = centers[::-1] if centers[0] > centers[1] else centers

    ax.axvline(centers[1], color="tab:red", ls="-.", alpha=1)
    ax.axvline(centers[0], color="tab:blue", ls="-.", alpha=1)

    ax.set_ylim(None, 1.25 * np.max(count))

    shift = np.abs(centers[0] - centers[1]) / 5
    ax.text(
        centers[1] + shift,
        0.98 * ax.get_ylim()[1],
        "Even",
        ha="right",
        va="top",
        rotation="vertical",
    )
    ax.text(
        centers[0] + shift,
        0.98 * ax.get_ylim()[1],
        "Odd",
        ha="right",
        va="top",
        rotation="vertical",
    )
    ax.set_xlabel(r"$\mathrm{Re}\,{\tilde C}_\mathrm{Q}$ [aF]")
    ax.set_ylabel("PDF")
