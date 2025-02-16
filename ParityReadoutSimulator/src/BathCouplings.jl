# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

abstract type AbstractBathCoupling end

struct StaticBathCoupling{Stype <: AbstractSpectralFunction} <: AbstractBathCoupling
    Xop::Matrix{ComplexF64}
    corr::BathCorrelationFunction{Stype}
end

function (X::StaticBathCoupling)(t)
    return X.Xop
end

function (X::StaticBathCoupling)(Xmat::Matrix{ComplexF64}, t)
    copyto!(Xmat, X.Xop)
end

struct PeriodicChargeNoiseBathCoupling{Stype <: AbstractSpectralFunction} <: AbstractBathCoupling
    """
    Given a charge operator `Nop` and a time profile of the gate charge,
    evaluate the bath coupling operator X(t) = A0*sin(ωd*t)*I - X0
    where X0 = ng0*I - N_op
    """
    X0::Matrix{ComplexF64}
    ωd::Float64
    A0::Float64
    corr::BathCorrelationFunction{Stype}
end

function (X::PeriodicChargeNoiseBathCoupling)(t::Float64)
    Xmat = similar(X.X0)
    return X(Xmat, t)
end

function _add_constant_diagonal!(a::T1, b::Matrix{T2}) where {T1,T2}
    N::Int = size(b,1)
    @inbounds for i in 1:N
        b[i,i] += a
    end
end

function (X::PeriodicChargeNoiseBathCoupling)(Xmat::Matrix{ComplexF64}, t::Float64)
    copyto!(Xmat, X.X0)
    _add_constant_diagonal!(X.A0*sin(X.ωd*t), Xmat)
    return Xmat
end
