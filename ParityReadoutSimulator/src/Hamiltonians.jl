# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

abstract type AbstractParameterizedHamiltonian{T <: Number} end

struct DrivenHamiltonian
    H0::Matrix{ComplexF64}
    V::Matrix{ComplexF64}
    ωd::Float64
end

function (Ht::DrivenHamiltonian)(t::Float64)
    return Ht.H0 + 2Ht.V*sin(Ht.ωd*t)
end

function get_Lindblad_time_grid(Hd::DrivenHamiltonian, δt_factor)
    period = 2pi/Hd.ωd
    N = round(Int, 1/δt_factor)
    return LinRange(0,period,N+1)
end
