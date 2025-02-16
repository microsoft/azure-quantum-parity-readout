# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import xarray as xr
import numpy as np

import matplotlib.pyplot as plt

from thermometry.helpers import find_extent
from thermometry.hough_lines import HoughLine

def extract_Cqs_v_and_plot(
        datasets: dict[str, xr.Dataset],
        lines: list[list[HoughLine]],
        gate_x: str = "V_DG2",
        gate_y: str = "V_QD1",
        V_cutter: str = "V_QC1",
        cut_length: int = 160,
        plot: bool = False
    ):
    """
        A function that converts a dict of CQ-converted measured datasets to nested dicts of curves to be fit together.
            - The outer dict contains curves that are measured on different temperatures.
            - The inner dict contains curves measured in the same gate-gate map.
            - The first nested list contains curves that lie on the same line.
            - The inner-most list contains Cq(ng) values of a cut that will be fit.

        This function filters down the lines found in a gate-gate map to only the ones that
        are long - capturing multiple curves for one charge resonance to help with the fit.

        `datasets`: a dict mapping dataset paths (different temps) to xarrays
        `lines`: all hough lines found for the various datasets
        `gate_x`: the x-axis of CQ(ng) that we will be comparing to our model curves.
        `gate_y`: an axis that provides redundancy curves in the limit we are in such that we can fit all the curves on that axis togehter if there is no charge transition that breaks the line undetectably
        `V_cutter`: a value that is different for different gate-gate maps at the same temperature.
        `cut_length`: number of x-axis points to get from experimental data and use to fit.
        `plot`: plot the chosen lines for the various gate-gate maps
    """

    lineidx = -1

    Cqs_v: dict[str, dict[int, np.ndarray]] = {}

    dataset_names = list(datasets.keys())

    sizedataset = [len(datasets[dataset][V_cutter]) for dataset in dataset_names]
    dataset_space = [(dataset_names[k], j) for k in range(len(sizedataset)) for j in range(sizedataset[k])]

    for idx, dataset in enumerate(datasets):
        ids = dataset
        Cq = datasets[dataset]

        x, y, _ = find_extent(Cq, gate_x=gate_x, gate_y=gate_y)

        if plot:
            facetgrid = Cq["Cq"].plot(row=V_cutter, col_wrap=8)
            facetgrid.fig.set_dpi(60)
            plt.suptitle(ids)
            ax = facetgrid.axs.flatten()

        chosen_lines: dict[int, list[HoughLine]] = {}

        pane_line_lengths = []

        for paneidx in range(sizedataset[idx]):
            lineidx += 1
            chosen_lines[paneidx] = []

            assert dataset==dataset_space[lineidx][0]
            relative_length = 0.
            for linenum, L in enumerate(lines[lineidx]):

                if plot:
                    ax[paneidx].plot(
                        x[L.xs],
                        y[L.ys],
                        color="blue", linestyle="--")

                rr, cc = L.draw_line()

                chosen_lines[paneidx].append(L)
                if plot:
                    ax[paneidx].text(x[L.get_midpoint()[0]], y[L.get_midpoint()[1]], s=f"{linenum}", color="blue")

                relative_length += L.len

                if plot:
                    ax[paneidx].set_title(np.round(relative_length, 0))

            relative_length /= len(lines[lineidx])

            pane_line_lengths.append(relative_length)

        pane_line_lengths_cutoff = np.quantile(pane_line_lengths, 0.5)

        def is_within_line_length_cutoff(paneidx):
            # only select lines that are longer than the cutoff length
            # removes any high frequency short lines found when fitting DQD lines
            return pane_line_lengths[paneidx] > pane_line_lengths_cutoff

        Cqs_v[ids] = {}
        for paneidx in range(sizedataset[idx]):

            rZ = Cq["Cq"].isel (**{V_cutter: paneidx}).to_numpy()
            iZ = Cq["iCq"].isel (**{V_cutter: paneidx}).to_numpy()

            if is_within_line_length_cutoff(paneidx):

                temp_Cqs_v = []

                for L in chosen_lines[paneidx]:

                    rr, cc = L.draw_line()

                    if plot and len(np.unique(rr)) > 5:
                        ax[paneidx].plot(x[rr], y[cc], color="green", linestyle="--")
                        ax[paneidx].plot(
                            x[L.xs],
                            y[L.ys],
                            color="red", linestyle="--"
                        )

                    Cqs = []
                    collected_rs = set()
                    for r, c in zip(cc, rr):
                        if r in collected_rs:
                            continue
                        collected_rs.add(r)

                        c = min(c, rZ.shape[1]-cut_length)
                        c = max(c, cut_length)

                        _rZ = rZ[r, c-cut_length:c+cut_length]
                        c = c-cut_length + np.round(np.argmax(_rZ)).astype(np.int16)
                        c = min(c, rZ.shape[1]-cut_length)
                        c = max(c, cut_length)

                        _x, _rZ, _iZ = ((x[c-cut_length:c+cut_length] - x[c]) * 1e6,
                                            rZ[r,c-cut_length:c+cut_length],
                                            iZ[r,c-cut_length:c+cut_length])

                        Cqs.append( (_x, _rZ, _iZ) )

                    temp_Cqs_v.append(Cqs)
                Cqs_v[ids][paneidx] = temp_Cqs_v

            if plot:
                ax[paneidx].set_title(f"{'Will Fit' if is_within_line_length_cutoff(paneidx) else ''}{np.round(pane_line_lengths[paneidx], 2)}")
                if is_within_line_length_cutoff(paneidx):
                    for axloc in ["bottom", "top", "right", "left"]:
                        ax[paneidx].spines[axloc].set_color('tab:red')
                        ax[paneidx].spines[axloc].set_linewidth(3)

        if plot:
            plt.show()
            plt.close()

    return Cqs_v
