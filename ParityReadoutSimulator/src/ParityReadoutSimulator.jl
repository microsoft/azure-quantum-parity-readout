# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

module ParityReadoutSimulator

##### External packages #####
using Interpolations
using StatsBase
using LinearAlgebra
using SparseArrays
using Arpack
using OrderedCollections
using PyCall

const xr = PyNULL()
const np = PyNULL()
const SIMULATED_DATA_PATH = Ref{String}()

function set_default_simulated_data_path()
    # To correctly load paths.py and posible relative paths, we move to the root of the
    # repository before loading the python file with path definitions.
    current_dir = pwd()
    cd(dirname(dirname(@__DIR__)))
    py""" # @pyinclude does not know __file__
    __file__ = "paths.py"
    """
    @pyinclude(py"__file__")
    cd(current_dir)
    SIMULATED_DATA_PATH[] = py"str(SIMULATION_DATA_FOLDER)"
    return SIMULATED_DATA_PATH[]
end

function get_simulated_data_path()
    return SIMULATED_DATA_PATH[]
end

function __init__()
    copy!(xr, pyimport_conda("xarray", "xarray"))
    copy!(np, pyimport_conda("numpy", "numpy"))
    pyimport_conda("h5netcdf", "h5netcdf")
    set_default_simulated_data_path()
end

##### Exports #####
include("exports.jl")

##### Modules #####
include("Units.jl")
using .Units
include("Noise.jl")

##### Core #####
include("Hamiltonians.jl")
include("SpectralFunctions.jl")
include("BathCorrelationFunctions.jl")
include("BathCouplings.jl")
include("LindbladOperators.jl")
include("QuantumCapacitance.jl")
include("TimeTraces.jl")
include("Plotting.jl")

#### Models #####
include("models/Fermions.jl")
include("models/MPR_TQD.jl")
include("models/DQD.jl")

include("Sweeps.jl")

end # module ParityReadoutSimulator
