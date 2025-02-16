# Data and analysis scripts for paper "Interferometric Single-Shot Parity Measurement in an InAs-Al Hybrid Device"

This repository contains parity readout simulation code and scripts for generating data-driven figures for the published paper "Interferometric Single-Shot Parity Measurement in InAs-Al Hybrid Devices" by Microsoft Azure Quantum.
* Preprint: [arXiv:2401.09549](https://arxiv.org/abs/2401.09549)
* Peer-reviewed publication : To appear in Nature (February 2025)

All references to figure numbers in this repository refer to the peer-reviewed revision of the paper published in Nature which may differ from the preprint.


## Data

Measured and simulated data is available in the accompanying dataset on [Zenodo](https://zenodo.org/records/14804380). You can download it manually and point to the data
folder in [paths.py](paths.py), see [Data Paths](#data-paths) for more info.

For convenience, you can also run `./download_data.sh` in your shell, which will download the data from Zenodo
and put it in a folder called `data`.

### Data Paths
As an interface for all our code, all raw and generated data is organized into the following folder variables:

- `DATA_FOLDER`: Variable for a folder containing four subdirectories outlined below
- `RAW_DATA_FOLDER`: Variable for a folder containing all measured data to be used for analysis and plotting.
- `CONVERTED_DATA_FOLDER`: Variable for a target folder to contain all CQ Converted datasets generated from running `cq_conversion/workflow.py` on the data in `RAW_DATA_FOLDER`
- `CQ_CONV_INTERMEDIATE_FIGURE_FOLDER`: Variable for a target folder to store intermediate figures for debugging the output of various stages of CQ Conversion
- `SIMULATION_DATA_FOLDER`: Variable for a folder containing all the simulated data.

The paths to these folder variables are defined in [`paths.py`](paths.py). By default, `DATA_FOLDER` points to
a folder named `data` in the root of this repository. These variables are propagated through the various
scripts that rely on measured, simulated or CQ converted data.

We provide an `unzip.py` python script to automatically generate this target directory structure from
the downloaded zip files. In case this fails (e.g. due to a different OS), the zip outputs on the
zenodo should correspond to the following directories:

- `simulated.zip` $\to$ `data/simulated/...`
- `converted_data.zip` $\to$ `data/converted_data/...`
- `dot_tuneup_A1.zip` $\to$ `data/raw_data/dot_tuneup_A1/...`
- `injector.zip` $\to$ `data/raw_data/injector/...`
- `mpr_A1.zip` $\to$ `data/raw_data/mpr_A1/...`
- `mpr_A2.zip` $\to$ `data/raw_data/mpr_A2/...`
- `mpr_B1.zip` $\to$ `data/raw_data/mpr_B1/...`
- `charge_noise.zip` $\to$ `data/raw_data/charge_noise/...`
- `cut_loop_A.zip` $\to$ `data/raw_data/cut_loop_A/...`
- `cut_loop_B.zip` $\to$ `data/raw_data/cut_loop_B/...`
- `qdmzm_A1.zip` $\to$ `data/raw_data/qdmzm_A1/...`
- `qpp.zip` $\to$ `data/raw_data/qpp/...`
- `tgp2_tuneup.zip` $\to$ `data/raw_data/tgp2_tuneup/...`
- `thermometry.zip` $\to$ `data/raw_data/thermometry/...`
- `trivial_A.zip` $\to$ `data/raw_data/trivial_A/...`
- `trivial_B.zip` $\to$ `data/raw_data/trivial_B/...`

Note: All the analysis scripts in `measured_analysis_figures` use only the files in `data/converted_data` and `data/simulated`
except for `figureS6_dot_tuneup.md`, `figureS7_tgp2_tuneup.md` and `figureS18_deviceB1_charge_noise.md` which only use raw measured data.

### Minimal Data to Reproduce Figures

Some of the raw datasets (all datasets labelled `run_name.zip`) contain timetraces that were left uncoarsened and as a result, are really large. If you are only interested
in using the coarsened data to regenerate a figure, you can download the `converted_data.zip` file from zenodo. This zip contains all the CQ converted RF measurements used to
generate the figures (see note in previous subsection).

As a result, the minimal datasets you need to download from [Zenodo](https://zenodo.org/records/14804380) to reproduce all figures in the paper are:
- `simulated.zip` (1.0 GB)
- `converted_data.zip` (4.8 GB)
- `charge_noise.zip` (3.0 MB)
- `dot_tuneup_A1.zip` (0.8 MB)
- `tgp2_tuneup.zip` (20.1 MB)

You should put these files in a directory named `data` and then run `unzip.py` to extract the files into their respective locations automatically.

## Requirements

The data analysis and figure generating scripts in this repository were run using Python 3.10.9 and the environment specified in [`uv.lock`](uv.lock) (we also provide a [`requirements.txt`](requirements.txt) for advanced users
who do not wish to use `uv`).

The simulation data and figures were generated using Julia `v1.10.8`. See [ParityReadoutSimulator/README.md](ParityReadoutSimulator/README.md) for additional information on the `ParityReadoutSimulator` and Julia environment requirements.

The quickest way to reproduce the environment used for data analysis and figure generation is to use the
tool [`uv`](https://docs.astral.sh/uv/), which can be installed by following [these instructions](https://docs.astral.sh/uv/getting-started/installation/).

Once `uv` is installed, running the following commands will install the necessary software into an isolated
environment, and activate that environment for future use:
```bash
uv sync # download Python 3.10.9, and all dependencies, creating a virtual environment with everything installed
source .venv/bin/activate  # activate the virtual environment
```

## Modules

This package has 4 main folders:
1. `cq_conversion`: A python module responsible for converting measured data from units of voltage to units of (quantum) capacitance ($C_{\rm Q}$)provided information about the resonator and RF background.
2. `measured_analysis_figures`: A folder of python scripts and notebooks that are used to generate all data analysis figures from measured data. Requires data that has already been $C_{\rm Q}$-converted, either by using the $C_{\rm Q}$-converted data from the accompanying dataset or by running the $C_{\rm Q}$-conversion code as outlined in the [CQ Conversion](#cq_conversion) section of this document on the raw measured data.
3. `ParityReadoutSimulator`: A Julia package for computing the steady-state capacitive response of small mesoscopic devices composed of single-level quantum dots and Majorana zero-modes. Also contains the notebooks used to generate reference simulated data and figures.
4. `thermometry`: A python modules responsible for extracting the electron temperature from DQD datasets acquired at various different temperatures by curve_fitting correlated parameters together.

### 1. `cq_conversion`

This module contains all the datasets that need to be CQ converted in `datasets.py` along with information about the resonator and background.

To run CQ conversion on a specific datasets (for example the `A1` and `thermometry` datasets), run `python cq_conversion/workflow.py A1 thermometry` in your terminal. To run CQ conversion on all
datasets, use `python cq_conversion/workflow.py --all`. This takes long for datasets with timetraces since they are large hypercubes - this could easily take over an hour on slower machines due to the size of the timetrace datasets.

The workflow file reads netcdfs from `RAW_DATA_FOLDER` and puts them in `CONVERTED_DATA_FOLDER` for use in later analyses and plotting.

Other files:
- `corrections.py`: Applies required correction on the measured data for off-resonance drive, required for the small parameter expansion used to cq convert, see the CQ conversion supplementary for more details.
- `cq_conversion.py`: The logic for interpolating a calibration curve and assigning the capacitance value based on the proximity of the measured data to the calibration curve.
- `cq_plot.py`: Plotting code required to generate intermediate cq conversion figures. These figures aren't included the paper, they are meant as useful sanity checks when performing CQ conversion.
- `extract_parameters.py`: A script that spits out input parameters to use for CQ conversion if there is no prior for this information - can be run in the same manner as `workflow.py`. E.g. `python extract_parameters.py --all`.
- `helpers.py`: Helper functions that are useful when dealing with calibration curves.

### 2. `measured_analysis_figures`

Code that is used to generate figures 3, S5, S6, S7, S9, S10, S11, S12, S13, S14, S16, S17, S18 and S19
in the Nature paper "Interferometric Single-Shot Parity Measurement in InAs-Al Hybrid Devices".

- `analysis_code.timetrace_analysis.py`: Defines functions used for time trace analysis and metric extraction (kurtosis, dwell times).
- `analysis_code.plotting_code.py`: Plotting functions called from `measurement_analysis_figures` notebooks.
- `analysis_code/common.py`: Helper plotting tools.

To regenerate all figures in the paper with measured data, you can run `jupytext --execute measured_analysis_figures/*.md`. You can alternatively convert a script to a python notebook you can run using `jupytext measured_analysis_figures/[NAME].md --to ipynb`, then run through it as you normally would in Jupyter Notebook or VSCode. See the [jupytext documentation](https://jupytext.readthedocs.io/en/latest/) for more details on how to work with `md` files.

#### Additional Requirements

The notebooks `figureS3_deviceA1_parity_measurements.md`, `figureS9_deviceA2_parity_measurements.md`, `figureS10_deviceB1_parity_measurements.md`,
`figureS11_qd_mzm.md`, `figureS12_scenarios.md`, `figureS13_cut_loop.md`, `figureS14_trivial_measurement.md`,
`figureS16_deviceB_qpp_times.md`, `figureS17_qpp.md`, `figureS19_deviceB_thermometry.md` additionally requires CQ Converted data be generated into the `CONVERTED_DATA_FOLDER`.
You can either use the cq converted data from zenodo or run the CQ conversion code as outlined in the [CQ Conversion](#cq_conversion) section of this document on the raw data from zenodo.

The notebook `figureS19_deviceB_thermometry.md` requires a simulation reference dataset generated from `ParityReadoutSimulator` in addition to the cq converted measurement data. You can either use the reference dataset provided in the zenodo upload or regenerate the simulation data by following the details in the `ParityReadoutSimulator` section.

### 3. `ParityReadoutSimulator`
A Julia package for computing the steady-state capacitive response of small mesoscopic devices composed of single-level quantum dots and a few Majorana zero-modes. This simulation code follows the methods and assumptions outlined in section "S2.4. Dynamical $C_{\rm Q}$ calculation using open systems dynamics" of the supplementary materila accompanying the published paper. See [ParityReadouSimulator/README.md](./ParityReadouSimulator/README.md) for additional details on the simulation code and the Julia environment.

`ParityReadouSimulator/examples` contains notebooks for generating figures based on the accompanying simulated data. Each notebook also contains the required code to regenerate the simulation data using the `ParityReadoutSimulator` package.

### 4. `thermometry`

A module used in `figureS19_deviceB_thermometry.md` to perform simultaneous curve fits of double quantum dots (DQD) measurements acquired at the same temperature. Compares open systems dynamics simulations from
`ParityReadoutSimulator` to CQ converted measured data to identify and extract the broadening caused by temperature.

This module relies on a modified variant of the hough lines algorithm to detect charge resonances in a DQD, to extract a group of correlated peaks that lie on the same charge transition.
We assume that these curves share a dot-dot coupling and broadening term.

These correlated fits allows to fit one temperature for every gate-gate map measured
(acquired at the same puck temperature), multiple broadening terms and dot-dot couplings, one per extracted line and mutliple shifts, 3 per each curve.
The shifts are one $n_g$ shift (x-axis translation of the curve) due to dot stability and 2 real (1 complex) offsets for the real and imaginary part of $C_Q$.

See Supplementary "S10: Electron temperature" for more details on this analysis.

## Microsoft Open Source Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

Resources:

- [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)
- [Microsoft Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
- Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions or concerns
- Employees can reach out at [aka.ms/opensource/moderation-support](https://aka.ms/opensource/moderation-support)

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
