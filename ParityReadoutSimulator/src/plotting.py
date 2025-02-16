# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import matplotlib as mpl
import numpy as np

def add_rectangle(ax, xy, width, height, color="tab:red"):
    ax.add_patch(mpl.patches.Rectangle(
        xy,
        width,
        height,
        linestyle="--",
        linewidth=1,
        edgecolor=color,
        facecolor="none",
        clip_on=False,
        zorder=1001,
    ))


def add_rectangle_horizontal(ax, y, height, color="tab:red", wpad=0.2e-3):
    add_rectangle(
        ax,
        (ax.get_xlim()[0]-wpad, y - height/2),
        np.diff(ax.get_xlim())[0]+2*wpad,
        height,
        color,
    )

def add_subfig_label(
    ax: mpl.axes.Axes,
    label: str,
    text_x_shift: float = 0,
    text_y_shift: float = 0,
    description: str = "",
    description_x_shift: float = 0,
    description_y_shift: float = 0,
) -> None:
    """Add a label to a subfigure."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + (x1 - x0) * text_x_shift
    y = y1 + (y1 - y0) * text_y_shift
    ax.text(
        x,
        y,
        label,
        c="k",
        fontweight="bold",
        ha="left",
        va="bottom",
        size=12,
        zorder=1001,
    )
    if description:
        ax.text(
            x + 0.2*(x1-x0) + description_x_shift,
            y + description_y_shift,
            description,
            c="k",
            ha="left",
            va="bottom",
            zorder=1001,
        )
