# ParityReadoutSimulator

`ParityReadoutSimulator` is a Julia package for computing the steady-state capacitive response of small mesoscopic devices composed of single-level quantum dots and a few Majorana zero-modes. This simulation code follows the methods and assumptions outlined in section "S2.4. Dynamical $C_{\rm Q}$ calculation using open systems dynamics" of the supplementary materila accompanying the published paper.

The `examples` directory contains notebooks to reproduce the figures involving simulation data of the published manuscripts. Each notebook is separate in two parts (i) Generating figures based on precomputed data (available in the accompanying Zenodo dataset), and (ii) the code required to regenerated the simulation datasets.

## Installation
The simulation code was tested on the long-term support (LTS) release of Julia (version 1.10.8).
1. Julia install
    * If not already installed, visit https://julialang.org/downloads/, to download and install Julia.

2. Jupyter kernel
    * Example codes are provided as jupyter notebooks in markdown format in the `examples` directory. Opening these require the `jupytext` package to be installed.
    * In addition to a standard jupyter installation, opening notebooks require the julia package [`IJulia`](https://github.com/JuliaLang/IJulia.jl), which can be installed with
    ```julia
    using Pkg
    Pkg.add("IJulia")
    ```

3. Environment and required packages.
    * An project environment is provided in [ParityReadoutSimulator/Project.toml](ParityReadoutSimulator/Project.toml). This environment can be activated using
    ```julia
    using Pkg
    Pkg.activate("/path/to/.../azure-quantum-parity-readout/ParityReadoutSimulator")
    ```
    * A full manifest of the packages used is also available in [ParityReadoutSimulator/Manifest.toml](ParityReadoutSimulator/Manifest.toml).
