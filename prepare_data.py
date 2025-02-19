import sys, os
import subprocess
from pathlib import Path

sys.path.append(".")
from paths import DATA_FOLDER

try:
    import wget
except:
    raise Exception("To use `prepare_data.py` you need to pip install wget")

# if len(sys.argv) == 1:
basepath = str(DATA_FOLDER)
# else:
#     basepath = sys.argv[1]

print("Looking for files in", basepath, "\n")

zenodo_doi = "14804380"

zenodo_record = f"https://zenodo.org/records/{zenodo_doi}"
zenodo_file_api = f"https://zenodo.org/api/records/{zenodo_doi}/files-archive"
zenodo_individual_files_api = f"https://zenodo.org/records/{zenodo_doi}/files"

from zipfile import ZipFile, ZIP_DEFLATED, is_zipfile
from concurrent.futures import ProcessPoolExecutor

zipfiles = [
    "charge_noise.zip",
    "converted_data.zip",
    "cut_loop_A.zip",
    "cut_loop_B.zip",
    "dot_tuneup_A1.zip",
    "injector.zip",
    "mpr_A1.zip",
    "mpr_A2.zip",
    "mpr_B1.zip",
    "qdmzm_A1.zip",
    "qpp.zip",
    "simulated.zip",
    "tgp2_tuneup.zip",
    "thermometry.zip",
    "trivial_A.zip",
    "trivial_B.zip",
]

minimal_files = [
    "converted_data.zip",
    "simulated.zip",
    "tgp2_tuneup.zip",
    "charge_noise.zip",
    "dot_tuneup_A1.zip",
]


def process_file(filename):
    full_path = os.path.join(basepath, filename)
    if not is_zipfile(full_path):
        print(
            f"Cannot find {full_path}, some files may be missing for the full analysis."
        )
    else:
        print(f"Unpacking {full_path}")
        with ZipFile(full_path, "r") as myzip:
            myzip.extractall(path=basepath)


if __name__ == "__main__":

    download, minimal_dataset = None, None
    if "--download-all" in sys.argv:
        download = True
        minimal_dataset = "n"
    if "--download-minimal" in sys.argv:
        download = True
        minimal_dataset = "Y"

    if download is None:
        to_download = input(
            f"""To run the code in this repo and reproduce the paper figures, you
will need access to the measurement and simulation data. You can do so
by downloading it from {zenodo_record} and placing
it in a folder named data (total path should be parity-readout/data).

Do you want to download the data using this script (this will query
{zenodo_file_api})? If you have
already downloaded the data, you can skip this step.
[Y/n] """
        )

        if to_download == "Y":
            download = True
        elif to_download == "n":
            download = False
        else:
            raise ValueError("Answer to download can be either Y or n.")

    if download:

        Path(basepath).mkdir(exist_ok=True)

        if minimal_dataset is None:
            minimal_dataset = input(
                """\n\nDo you want to download the minimal datasets required reproduce the paper (~6GB)?
Answer yes to download {converted_data.zip, simulated.zip, tgp2_tuneup.zip} and no
to download all datasets (~150Gb+).
[Y/n] """
            )

        if minimal_dataset == "Y":

            for file_name in minimal_files:
                print("\nDownloading", file_name)
                _ = wget.download(
                    f"{zenodo_individual_files_api}/{file_name}",
                    out=f"{basepath}/{file_name}",
                )

        elif minimal_dataset == "n":

            for file_name in zipfiles:
                if not is_zipfile(f"{basepath}/{file_name}"):
                    print("\nDownloading", file_name)
                    _ = wget.download(
                        f"{zenodo_individual_files_api}/{file_name}",
                        out=f"{basepath}/{file_name}",
                    )
                else:
                    print(f"\nFound {file_name}, skipping download")

        else:
            raise ValueError("Answer to download minimal dataset can be either Y or n.")

    print("\n\nExtracting files and setting up directory structure")

    with ProcessPoolExecutor() as executor:
        executor.map(process_file, zipfiles)
