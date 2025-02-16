# +
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# -

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from types import SimpleNamespace
import tgp # package source lives in https://github.com/microsoft/azure-quantum-tgp


def analyze_two(
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    min_cluster_size: int = 7,
    zbp_average_over_cutter: bool = True,
    zbp_probability_threshold: float = 0.6,
    gap_threshold_high: float = 70e-3,
    gap_threshold_factor: float = 0.05,
    cluster_gap_threshold = None,
    cluster_volume_threshold = None,
    cluster_percentage_boundary_threshold = None,
):
    ds_left, ds_right = tgp.two.extract_gap(
        ds_left, ds_right, gap_threshold_factor=gap_threshold_factor
    )
    zbp_ds = tgp.two.zbp_dataset_derivative(
        ds_left,
        ds_right,
        average_over_cutter=zbp_average_over_cutter,
        zbp_probability_threshold=zbp_probability_threshold,
    )

    tgp.two.set_zbp_gap(zbp_ds, ds_left, ds_right)
    tgp.two.set_gap_threshold(zbp_ds, threshold_high=gap_threshold_high)

    zbp_ds = tgp.two.cluster_and_score(
        zbp_ds,
        min_cluster_size=min_cluster_size,
        cluster_gap_threshold=cluster_gap_threshold,
        cluster_volume_threshold=cluster_volume_threshold,
        cluster_percentage_boundary_threshold=cluster_percentage_boundary_threshold,
    )
    return SimpleNamespace(
        zbp_ds=zbp_ds,
        ds_left=ds_left,
        ds_right=ds_right,
    )


# +
def _list_diff(a, b):
    return [x for x in a if x not in set(b)]

def plot_stage2_diagram(
    ds: xr.Dataset,
    cutter_value,
    zbp_cluster_numbers,
    fig,
    axs,
    pct_boundary_shifts = None,
    gap_lim = 60.0,
    deco_yfactor = 1.0,
    description = None,
):
    """Plot the stage 2 diagram."""
    if "cutter_pair_index" in ds.dims:
        ds_sel = ds.sel(cutter_pair_index=cutter_value)
    else:
        ds_sel = ds

    gap_bool = 1.0 * ds_sel.gap_boolean.squeeze()
    cmap = mpl.colors.ListedColormap(["w", "tab:blue"])
    pl_kw = {
        "add_colorbar": False,
        "shading": "nearest",
        "infer_intervals": False,
        "linewidth": 0,
        "rasterized": True,
        "vmin": -0.5,
        "vmax": 1.5,
    }
    im = (
        gap_bool.squeeze()
        .transpose("V", "B")
        .plot.pcolormesh(ax=axs[0], cmap=cmap, zorder=1, **pl_kw)
    )
    cax = fig.add_axes([-0.13, 0.41*deco_yfactor, 0.03, 0.25*deco_yfactor])
    cbar = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[0, 1])
    cbar.ax.set_yticklabels(["Gapless", "Gapped"])

    zbp_bool = ds_sel.zbp.squeeze()
    cmap = mpl.colors.ListedColormap([np.array([255, 229, 82]) / 256, "tab:orange"])
    im = (
        gap_bool.where(zbp_bool, np.nan)
        .squeeze()
        .transpose("V", "B")
        .plot.pcolormesh(ax=axs[0], cmap=cmap, zorder=2, **pl_kw)
    )
    cax = fig.add_axes([-0.13, (0.1 - 0.004)*deco_yfactor, 0.03, 0.25*deco_yfactor])
    cbar = fig.colorbar(
        im,
        cax=cax,
        orientation="vertical",
        ticks=[0, 1],
    )
    cbar.ax.set_yticklabels(["Gapless & ZBP", "Gapped & ZBP"])

    if description is not None:
        cax = fig.add_axes([-0.13, 1-0.06*deco_yfactor, 0.03, 0.0])
        cax.axis("off")
        cax.text(0, 0, description, ha="left", va="top")

    axs[0].set_title("")

    reps = 20
    B = np.array(ds["B"])
    V = np.array(ds["V"])
    B1 = np.linspace(B.min(), B.max(), B.size * reps)
    V1 = np.linspace(V.min(), V.max(), V.size * reps)

    gp = np.abs(ds_sel.gap)
    clusters = tgp.common.expand_clusters(
        ds_sel.gapped_zbp_cluster,
        dim="zbp_cluster_number",
    )
    existing_zbp_cluster_numbers = np.array(
        clusters.zbp_cluster_number,
        dtype=int,
    ).tolist()
    if zbp_cluster_numbers is None:
        zbp_cluster_numbers = []
    elif zbp_cluster_numbers == "all":
        zbp_cluster_numbers = existing_zbp_cluster_numbers
    else:
        invalid_indices = _list_diff(zbp_cluster_numbers, existing_zbp_cluster_numbers)
        if invalid_indices:
            zbp_cluster_numbers = _list_diff(zbp_cluster_numbers, invalid_indices)
            print(
                "Warning: Clusters with indices",
                invalid_indices,
                "do not exist and are removed from zbp_cluster_numbers",
            )
    for i in zbp_cluster_numbers:
        gp = gp * (1.0 - 2.0 * clusters.sel(zbp_cluster_number=i))

    pcm = (1e3 * gp).plot.pcolormesh(
        ax=axs[1],
        vmin=-gap_lim,
        vmax=gap_lim,
        cmap="RdBu",
        linewidth=0,
        rasterized=True,
        add_colorbar=False,
        shading="nearest",
        infer_intervals=False,
    )
    pcm.set_edgecolor("face")
    cax = axs[1].inset_axes([1.03, 0, 0.03, 1], transform=axs[1].transAxes)
    exceed_min = int(np.min(1e3 * gp) < -gap_lim)
    exceed_max = int(np.max(1e3 * gp) > gap_lim)
    extend = {0: "neither", 1: "min", 2: "max", 3: "both"}[exceed_min + 2 * exceed_max]
    cb = fig.colorbar(
        pcm,
        ax=axs[1],
        cax=cax,
        ticks=np.arange(-gap_lim, gap_lim + 1e-5, 10),
        extend=extend,
        extendfrac=0.03*deco_yfactor,
    )
    cb.set_label(r"$q\Delta$ [$\mu$eV]")
    info = []
    for i in zbp_cluster_numbers:
        info.append(f"Cluster #{i}, cutter_pair_index={cutter_value}")

        cluster = clusters.sel(zbp_cluster_number=i)

        for ax in axs[:2]:
            cluster.astype(float).interp(B=B1, V=V1, method="nearest").plot.contour(
                x="B",
                y="V",
                ax=ax,
                levels=[0.5],
                colors="k",
                linewidths=1.5,
                linestyles="-",
                zorder=1000,
            )

        pct_boundary = 100 * float(
            ds_sel.percentage_boundary.sel(zbp_cluster_number=i).item(),
        )
        info.append(rf"    {pct_boundary:.0f}% gapless boundary")

        gp = np.asarray(np.abs(ds_sel.gap) * clusters.sel(zbp_cluster_number=i))
        top_gap = np.percentile(gp[np.nonzero(gp)], 80.0)
        info.append(rf"    Top 20% percentile gap = {1e3*top_gap:.2g} ueV")
        median_gap = np.median(gp[np.nonzero(gp)])
        info.append(rf"    Median gap = {1e3*median_gap:.2g} ueV")

        dB, dV = np.diff(B)[0], np.diff(V)[0]
        Vc, Bc = center_of_mass(np.array(cluster.transpose("V", "B")))
        Bc = B[0] + dB * Bc
        Vc = V[0] + dV * Vc
        pct_boundary_shifts = pct_boundary_shifts or {}
        shift_x, shift_y = pct_boundary_shifts.get(i, (0, 0))
        axs[0].text(
            Bc + shift_x,
            Vc + shift_y,
            f"{pct_boundary:.0f}% gapless boundary",
            c="k",
            ha="center",
            va="center",
            zorder=1100,
        )

        volume_px = np.sum(cluster)
        volume_mVT = dB * 1e3 * dV * volume_px
        info.append("    Volume = %d pixels = %.2g mV*T" % (volume_px, volume_mVT))
        info.append("    Center of mass B = %.1f T" % Bc)

    for ax in axs:
        ax.set_title("")
        ax.set_xlabel("$B_{\\!\\parallel}$ [T]")
    axs[0].set_ylabel("$V_\\mathrm{WP1}$ [V]")
    axs[1].set_ylabel("")

    print("\n".join(info))
# -
