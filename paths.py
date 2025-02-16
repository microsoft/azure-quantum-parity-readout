from pathlib import Path

DATA_FOLDER = Path(__file__).absolute().parent / "data"

assert DATA_FOLDER.exists(), "The data directory doesn't exist, make sure to download and prepare the data as described in the repository README.md"

RAW_DATA_FOLDER = DATA_FOLDER / "raw_data"
CONVERTED_DATA_FOLDER = DATA_FOLDER / "converted_data"
SIMULATION_DATA_FOLDER = DATA_FOLDER / "simulated"
CQ_CONV_INTERMEDIATE_FIGURE_FOLDER = DATA_FOLDER / "cq_conv_figs"

# The raw data must be downloaded from zenodo to be able to generate the plots
if not RAW_DATA_FOLDER.exists():
    print("The raw data directory doesn't exist, make sure to download and prepare the data as described in the repository README.md")

# The converted_data and simulated folders are included in the zenodo files but can
# also be regenerated if one wishes to reproduce them.
CONVERTED_DATA_FOLDER.mkdir(exist_ok=True)
SIMULATION_DATA_FOLDER.mkdir(exist_ok=True)

# This directory is for debugging plots while running CQ conversion, this is
# not included in the zenodo download and is here optionally if you want to
# re-run the CQ conversion.
CQ_CONV_INTERMEDIATE_FIGURE_FOLDER.mkdir(exist_ok=True)
