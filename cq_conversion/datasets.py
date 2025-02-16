# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from glob import glob
from cq_conversion import CQConversionInput

import sys
sys.path.insert(1, '.')
sys.path.insert(1, '..')

# these imports are an interface that will be used by other classes
from paths import RAW_DATA_FOLDER, CONVERTED_DATA_FOLDER, CQ_CONV_INTERMEDIATE_FIGURE_FOLDER  # noqa: E402

# The inductance of the readout resonator
deviceA_L_ind = 115e-9
deviceB_L_ind = 115e-9

datasets: dict[str, CQConversionInput | list[CQConversionInput]] = {
    'A1': CQConversionInput(
        name="A1",
        data_path="mpr_A1/mpr_A1_Vrf.h5",
        calib_path="mpr_A1/mpr_A1_calib.h5",
        elec_delay=-146.4e-9,
        L_ind=deviceA_L_ind,
        gaussian_sigma=10,
        fdelta=2e6,

        drive_frequency=625e6,
        resonance_frequency=622e6,

        isel=dict(),
        coarsen_time=20,
    ),

    'A2': CQConversionInput(
        name="A2",
        data_path="mpr_A2/mpr_A2_Vrf.h5",
        calib_path="mpr_A2/mpr_A2_calib.h5",
        elec_delay=-146.4e-9,
        L_ind=deviceA_L_ind,
        gaussian_sigma=10,
        fdelta=2e6,

        drive_frequency=619.18e6,
        resonance_frequency=619.18e6,

        correct_off_resonance_drive=False,

        isel=dict(),
        coarsen_time=20,
    ),

    'qd1mzm_A1_qd1': CQConversionInput(
        name="A1_qd1mzm",
        data_path="qdmzm_A1/qd1mzm_Vrf.h5",
        calib_path="qdmzm_A1/qdmzm_A1_calib.h5",
        elec_delay=-146.4e-9,
        L_ind=deviceA_L_ind,
        gaussian_sigma=10,
        fdelta=2e6,

        drive_frequency=625e6,
        resonance_frequency=619.5e6,

        isel=dict(),
        coarsen_time=None, # QDMZM data has no time axis to coarsen
    ),

    'qd3mzm_A1_qd1': CQConversionInput(
        name="A1_qd3mzm",
        data_path="qdmzm_A1/qd3mzm_Vrf.h5",
        calib_path="qdmzm_A1/qdmzm_A1_calib.h5",
        elec_delay=-146.4e-9,
        L_ind=deviceA_L_ind,
        gaussian_sigma=10,
        fdelta=2e6,

        drive_frequency=625e6,
        resonance_frequency=619.5e6,


        isel=dict(),
        coarsen_time=None, # QDMZM data has no time axis to coarsen
    ),

    'trivial_A': CQConversionInput(
        name="trivial_A",
        data_path='trivial_A/trivial_A_Vrf.h5',
        calib_path='trivial_A/trivial_A_calib.h5',
        elec_delay=-140.4e-9,
        L_ind=deviceA_L_ind,
        gaussian_sigma=10,
        fdelta=2e6,

        drive_frequency=619e6,
        resonance_frequency=621.6e6,

        correct_off_resonance_drive=False,

        isel=dict(),
        coarsen_time=20,
    ),

    'trivial_B': CQConversionInput(
        name="trivial_B",
        data_path='trivial_B/trivial_B_Vrf.h5',
        calib_path='trivial_B/trivial_B_calib.h5',
        elec_delay=-141.1e-9,
        L_ind=deviceB_L_ind,
        gaussian_sigma=10,
        fdelta=2e6,

        drive_frequency=653.3e6,
        resonance_frequency=654.3e6,

        correct_off_resonance_drive=False,

        isel=dict(),
        coarsen_time=20,
    ),

    'cut_loop_A': CQConversionInput(
        name="cut_loop_A",
        data_path='cut_loop_A/cut_loop_A_Vrf.h5',
        calib_path='cut_loop_A/cut_loop_A_calib.h5',
        elec_delay=-146e-9,
        L_ind=deviceA_L_ind,
        gaussian_sigma=8,
        fdelta=2e6,

        drive_frequency=619.5e6,
        resonance_frequency=621.9e6,

        isel=dict(),
        coarsen_time=20,
    ),

    'cut_loop_B': CQConversionInput(
        name="cut_loop_B",
        data_path='cut_loop_B/cut_loop_B_Vrf.h5',
        calib_path='cut_loop_B/cut_loop_B_calib.h5',
        elec_delay=-143.2e-9,
        L_ind=deviceB_L_ind,
        gaussian_sigma=8,
        fdelta=2e6,

        drive_frequency=649.6e6,
        resonance_frequency=651.3e6,

        isel=dict(),
        coarsen_time=20,
    ),

    'B1': CQConversionInput(
        name="B1",
        data_path="mpr_B1/mpr_B1_Vrf.h5",
        calib_path="mpr_B1/mpr_B1_calib.h5",

        elec_delay=-142e-9,
        L_ind=deviceB_L_ind,
        gaussian_sigma=10,
        fdelta=2e6,

        drive_frequency=651.6e6,
        resonance_frequency=651.7e6,


        isel=dict(),
        coarsen_time=20,
    ),

    'injector': CQConversionInput( # Device B
        name="injector",
        data_path="injector/injector_Vrf.h5",
        calib_path="injector/injector_calib.h5",

        elec_delay=-142e-9,
        L_ind=deviceB_L_ind,
        gaussian_sigma=10,
        fdelta=2e6,

        drive_frequency=651.6e6,
        resonance_frequency=651.4e6,

        correct_off_resonance_drive=False,

        isel=dict(),
        coarsen_time=20,

    ),

    'qpp': CQConversionInput( # Device B
        name="qpp",
        data_path="qpp/qpp_Vrf.h5",
        calib_path="qpp/qpp_calib.h5",

        elec_delay=-65.4e-9,
        L_ind=78e-9,
        gaussian_sigma=2,
        fdelta=6e6,

        drive_frequency=776.5e6,
        resonance_frequency=776.5e6,

        correct_off_resonance_drive = False,

        isel=dict(),
        coarsen_time=1,

    ),

    'thermometry': [],

}

for thermometry_name in glob(str(RAW_DATA_FOLDER / "thermometry/calib/*.h5")):

    thermometry_name = thermometry_name.split("_")[-1].split(".h5")[0]

    assert isinstance(datasets['thermometry'], list) # assertion to make the type checking happy

    datasets['thermometry'].append(
        CQConversionInput(
            name=f"thermometry_{thermometry_name}",
            data_path=f"thermometry/data/thermometry_data_{thermometry_name}.h5",
            calib_path=f"thermometry/calib/thermometry_calib_{thermometry_name}.h5",
            target_path=f"thermometry/thermometry_{thermometry_name}_Cq.h5",

            elec_delay = -860e-9/2/3.141592653589793238,
            L_ind = deviceB_L_ind,
            gaussian_sigma =  10,
            fdelta = 2e6,

            drive_frequency = 649.57e6,
            resonance_frequency = 649.57e6,

            correct_off_resonance_drive = False,

            isel=dict(),
            coarsen_time=None, # Electron Temp data has no time axis

        )
    )
