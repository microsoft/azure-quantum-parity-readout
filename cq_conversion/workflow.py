# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import xarray as xr
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

from cq_plot import plot_cq_interpolators
from cq_conversion import process_CqProjection, CQConversionInput

from timeit import default_timer as timer

from datasets import datasets, RAW_DATA_FOLDER, CONVERTED_DATA_FOLDER, CQ_CONV_INTERMEDIATE_FIGURE_FOLDER

from corrections import correct_off_resonance_drive


def cq_convert_ds(cq_conv_input: CQConversionInput, target_dir: Path=CONVERTED_DATA_FOLDER):
    print(">"*20, f"\nBegin CQ Converting {cq_conv_input.name}")
    tn = timer()

    calib_ds: xr.Dataset = xr.load_dataset(RAW_DATA_FOLDER / cq_conv_input.calib_path, engine="h5netcdf", chunks='auto').squeeze()
    raw_ds = xr.load_dataset(RAW_DATA_FOLDER / cq_conv_input.data_path, engine="h5netcdf") # can open_datasets with chunks='auto' if too big to fit in memory w/ drawback of speed

    tp, tn = tn, timer()
    print(f"Loaded Dataset {cq_conv_input.name}, time:", tn-tp)
    print(raw_ds.data_vars)

    f0 = cq_conv_input.resonance_frequency
    fd = cq_conv_input.drive_frequency
    ed = cq_conv_input.elec_delay

    fs = calib_ds.frequency.to_numpy()
    calib = calib_ds.Vrf.to_numpy() * np.exp(-1j*2*np.pi*ed*fs)

    tp, tn = tn, timer()
    print("Processed Calibration Curve, time:", tn-tp)

    corrected_data = raw_ds.isel(**cq_conv_input.isel)
    if cq_conv_input.coarsen_time is not None and cq_conv_input.coarsen_time > 1:
        corrected_data = corrected_data.coarsen(time=cq_conv_input.coarsen_time, boundary='trim').mean()

        tp, tn = tn, timer()
        print("Coarsened dataset, time:", tn-tp)

    # correct electrical delay
    corrected_data *= np.exp(-1j*2*np.pi*ed*fd)

    tp, tn = tn, timer()
    print("Corrected data electrical delay", tn-tp)

    if cq_conv_input.correct_off_resonance_drive:
        corrected_data = correct_off_resonance_drive(corrected_data, fs=fs, calib=calib, ed=0, f0=f0, fd=fd, name=cq_conv_input.name)

        tp, tn = tn, timer()
        print("Corrected data for off-resonance drive", tn-tp)

    print(f"> {cq_conv_input.name} ready for CQ Conversion")

    ## Plot CQ interpolators and save

    plot_cq_interpolators(
        fs=fs,
        calib=calib,
        data=corrected_data,
        analysis_input=cq_conv_input
    )

    tp, tn = tn, timer()

    plt.savefig(CQ_CONV_INTERMEDIATE_FIGURE_FOLDER / f"{cq_conv_input.name}_2_cq_interpolators.png")
    print("Plot Data and Calib Interpolators in IQ space, time:", tn-tp)
    print(f"Saved Data and Calib Interpolators Figure {cq_conv_input.name}_2_cq_interpolators.png")

    cq_conv_ds: xr.Dataset = process_CqProjection(
        fs=fs,
        calib=calib,
        data=corrected_data,
        analysis_input=cq_conv_input,
    )

    target_path = target_dir / cq_conv_input.target_name
    target_path.parent.mkdir(exist_ok=True)

    # in case a dataset already exists at that location and has a lock, delete and re-write instead of
    # amending to not lose the conversion.
    if target_path.exists():
        target_path.unlink()

    # add all data vars (except for Vrf_1) in raw ds back
    cq_conv_ds_w_data_vars = xr.merge([cq_conv_ds] + [raw_ds.data_vars[i] for i in raw_ds.data_vars if i != "Vrf_1"])

    cq_conv_ds_w_data_vars.to_netcdf(target_path, engine="h5netcdf")
    print(f"Saved Final Dataset at {target_path}")


def main():

    from dask.distributed import Client

    _ = Client(threads_per_worker=32, n_workers=1)

    parser = argparse.ArgumentParser(description='Run workflow on specified datasets.')
    parser.add_argument('keys', nargs='*', help='Keys to run on')
    parser.add_argument('--all', action='store_true', help='Run on all datasets')

    args = parser.parse_args()

    if args.all:
        args.keys = datasets.keys()

    if len(args.keys)==0:
        raise ValueError(f"""

Enter the name of the dataset you want to CQ-convert or `--all` to convert all.
Available datasets:
{list(datasets.keys())}

""")

    for key in args.keys:
        if isinstance(datasets[key], CQConversionInput):
            cq_convert_ds(datasets[key])
        elif isinstance(datasets[key], list):
            for _input in datasets[key]:
                cq_convert_ds(_input)
        else:
            raise ValueError(f"datasets entry for {key} is of the wrong format")

if __name__ == '__main__':
    main()
