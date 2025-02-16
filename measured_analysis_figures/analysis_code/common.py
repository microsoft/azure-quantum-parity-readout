# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from pathlib import Path
from string import ascii_lowercase
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import sys
sys.path.insert(1, '..')

# these imports are an interface that will be used by other classes
from paths import DATA_FOLDER, RAW_DATA_FOLDER, CONVERTED_DATA_FOLDER, SIMULATION_DATA_FOLDER  # noqa: E402

# Dictionary with data scaling and renaming scheme
PREPARE_DICT = {
    "time": {
        "scale": 1e3,
        "units": "ms",
        "long_name": "time",
    },
    "Cq": {
        "scale": 1,
        "units": "aF",
        "long_name": r"$\mathrm{Re}\,{\tilde C}_\mathrm{Q}$",
    },
    "CQ": {
        "scale": 1,
        "units": "aF",
        "long_name": r"$\mathrm{Re}\,C_\mathrm{Q}$",
    },
    "iCq": {
        "scale": 1,
        "units": "aF",
        "long_name": r"$\mathrm{Im}\,{\tilde C}_\mathrm{Q}$",
    },
    "V_lin_qd": {
        "scale": 1e3,
        "units": "mV",
        "long_name": r"$\Delta V_\mathrm{QD2}$",
    },
    "V_qd_1_plunger_gate": {
        "scale": 1e3,
        "units": "mV",
        "long_name": r"$V_\mathrm{QD1}$",
    },
    "V_qd_3_plunger_gate": {
        "scale": 1e3,
        "units": "mV",
        "long_name": r"$V_\mathrm{QD3}$",
    },
    "V_qd_1_plunger_gate_abs": {
        "scale": 1e3,
        "units": "mV",
        "long_name": r"$V_\mathrm{QD1}$",
    },
    "V_qd_3_plunger_gate_abs": {
        "scale": 1e3,
        "units": "mV",
        "long_name": r"$V_\mathrm{QD3}$",
    },
    "B_perp": {
        "scale": 1e3,
        "units": "mT",
        "long_name": r"$B_{x}$",
    },
    "phi": {
        "scale": 1,
        "units": r"$h/2e$",
        "long_name": r"$\Phi$",
    },
    "ng1": {
        "scale": 1,
        "units": None,
        "long_name": r"$N_\mathrm{g1}$",
    },
    "ng2": {
        "scale": 1,
        "units": None,
        "long_name": r"$N_\mathrm{g2}$",
    },
    "ng3": {
        "scale": 1,
        "units": None,
        "long_name": r"$N_\mathrm{g3}$",
    },
}


def prepare_data_for_plotting(ds, skip_scale=[]):
    """
    Prepare dataset for plotting by scaling and setting name and units attributes based on PREPARE_DICT.

    Parameters:
        ds (xarray.Dataset): Input dataset to be prepared.
        skip_scale (list of str): List of variable names to skip scaling.

    Returns:
        xarray.Dataset: Dataset with prepared data.
    """
    if "V_wire" in ds.dims:
        ds["V_wire"] = ds["V_wire"].round(4)

    for var, params in PREPARE_DICT.items():
        if var in ds.variables:
            if var not in skip_scale:
                ds[var] = ds[var] * params["scale"]
            if params["units"]:
                ds[var].attrs["units"] = params["units"]
            ds[var].attrs["long_name"] = params["long_name"]

    return ds


class panel_labeller:
    """
    Class for generating sequential labels for plot panels.
    """

    def __init__(self, sequence=ascii_lowercase):
        self.sequence = sequence
        self._label_index = 0

    def next(self):
        """
        Get the next label in the sequence.
        Returns:
            str: The next label.
        """
        label = self.sequence[self._label_index]
        self._label_index += 1
        return label


def add_subfig_label(ax, label, description=""):
    """
    Add a label to a subfigure.

    Parameters:
        ax (mpl.axes.Axes): The subplot axis to label.
        label (str): The label text.
        description (str, optional): Additional description text.
    """
    ax.set_title("")
    ax.set_title(
        rf"$\bf{{{label}}}$",
        loc="left",
    )
    ax.set_title(description, fontsize=10)


def _set_panel_properties(ax, xlabel, ylabel, title="", ylim=None, label=None):
    """
    Set properties for a given plot panel.

    Parameters:
        ax (mpl.axes.Axes): The subplot axis to modify.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str, optional): Title of the plot.
        ylim (tuple of float, optional): Y-axis limits.
        label (str, optional): Label for the subfigure.
    """
    ax.set_xlabel(xlabel)
    if not xlabel:
        plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel(ylabel)
    if not ylabel:
        plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    if label is not None:
        add_subfig_label(ax, label)


def add_rectangle(ax, xy, width, height, color="tab:red"):
    """
    Add a rectangular patch to a plot.

    Parameters:
        ax (mpl.axes.Axes): The subplot axis to modify.
        xy (tuple of float): (x, y) coordinates for the bottom-left corner of the rectangle.
        width (float): Width of the rectangle.
        height (float): Height of the rectangle.
        color (str, optional): Color of the rectangle.
    """
    ax.add_patch(
        mpl.patches.Rectangle(
            xy,
            width,
            height,
            linestyle="--",
            linewidth=1,
            edgecolor=color,
            facecolor="none",
            clip_on=False,
            zorder=1001,
        )
    )


def add_rectangle_horizontal(ax, y, height, color="tab:red", wpad=0.2e-3):
    """
    Add a plot-wide rectangle to a plot.

    Parameters:
        ax (mpl.axes.Axes): The subplot axis to modify.
        y (float): y-coordinate of the center of the rectangle.
        height (float): Height of the rectangle.
        color (str, optional): Color of the rectangle.
        wpad (float, optional): Width padding on each side of the rectangle.
    """
    add_rectangle(
        ax,
        (ax.get_xlim()[0] - wpad, y - height / 2),
        np.diff(ax.get_xlim())[0] + 2 * wpad,
        height,
        color,
    )


def add_field_arrow(ax, B, color="tab:blue"):
    """
    Add an arrow at a given field to a plot.

    Parameters:
        ax (mpl.axes.Axes): The subplot axis to modify.
        B (float): x-coordinate for the base of the arrow.
        color (str, optional): Color of the arrow.

    Returns:
        tuple: (y0, y1, arrow_len) where y0 and y1 are y-axis limits, and arrow_len is the length of the arrow.
    """
    y0, y1 = ax.get_ylim()
    arrow_len = 0.12 * (y1 - y0)
    ax.add_patch(
        FancyArrowPatch(
            (B, y0 + 0.5 * arrow_len),
            (B, y0 + 1.5 * arrow_len),
            arrowstyle="-|>",
            mutation_scale=7,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )
    return y0, y1, arrow_len


def add_field_arrow_text(ax, B, text, color="tab:blue", ha="left", text_B_shift=0.22):
    """
    Add an arrow at a given field value with accompanying text to a plot.

    Parameters:
        ax (mpl.axes.Axes): The subplot axis to modify.
        B (float): x-coordinate for the base of the arrow.
        text (str): Text to display near the arrow.
        color (str, optional): Color of the arrow.
        ha (str, optional): Horizontal alignment of the text.
        text_B_shift (float, optional): Horizontal shift of the text relative to B.
    """
    y0, y1, arrow_len = add_field_arrow(ax, B, color)
    if ha == "right":
        text_B_shift = -text_B_shift
    ax.text(
        B + text_B_shift,
        y0 + 0.8 * arrow_len,
        text,
        ha=ha,
        va="center",
        size=10,
        fontweight="bold",
    )
