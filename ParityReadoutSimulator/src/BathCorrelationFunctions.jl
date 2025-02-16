# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

using Interpolations
using QuadGK, Roots, FastGaussQuadrature

abstract type AbstractBathCorrelationFunction end

function _get_ω_grid(
    Sω::AbstractSpectralFunction,
    ;
    t_max_corr::Float64,
    ω_max_zero_search::Float64,
    ω_zero_threshold::Float64=1e-12,
    gauss_legendre_factor::Float64=10.,
)
    ω_min = find_zero(ω -> Sω(ω) - ω_zero_threshold, (-ω_max_zero_search, 0.), Roots.Bisection())
    ω_max = find_zero(ω -> Sω(ω) - ω_zero_threshold, (0., ω_max_zero_search), Roots.Bisection())

    ω, weights = gausslegendre(ceil(Int, gauss_legendre_factor * t_max_corr * max(ω_max, abs(ω_min))))

    ω = (ω_max - ω_min)/2 * ω .+ (ω_max + ω_min)/2
    weights = (ω_max - ω_min)/2 * weights

    return ω, weights
end

function _get_time_dependent_correlation_function(
    fω::Function,
    ω,
    weights,
    ;
    t_max_corr::Float64,
    num_t_points::Int=10_000,
)::ScaledInterpolation
    t_grid = range(-t_max_corr, t_max_corr, length=Int(num_t_points))

    w_fω = weights .* fω.(ω)

    iωt  = exp.(im * ω * minimum(t_grid))
    iωdt = exp.(im * ω * (t_grid[2] - t_grid[1]))

    fω_eval = zeros(ComplexF64, length(t_grid))
    for it in eachindex(fω_eval)
        fω_eval[it] = sum(k -> iωt[k] * w_fω[k], 1:length(ω)) / (2π)
        iωt .*= iωdt
    end

    fω_ = interpolate(fω_eval, BSpline(Quadratic(Line(OnGrid()))))
    fω_ = extrapolate(fω_, zero(ComplexF64))

    return scale(fω_, t_grid)
end

"""
A `BathCorrelationFunction` object is constructed from a single `Sω::AbstractSpectralFunction`
representing a bath spectral function S(ω). A single inner constructor is provided
with the following signature:

    function BathCorrelationFunction(
        Sω::Stype,
        ;
        τ_guess::Float64,
        t_max_corr_factor::Float64=15.,
        ω_max_zero_search_factor::Float64=10.,
        ω_zero_threshold::Float64=1e-12,
        gauss_legendre_factor::Float64=10.,
        num_t_points::Int=10_000,
    ) where {Stype <: AbstractSpectralFunction}

This inner constructor computes the time domain correlation functions
g(t) = FT[ sqrt(S(ω)) ] and store ot
as struct field `gt` (`ScaledInterpolation` object). Additionally,
it computes the bath noise correlation time `τ` based on g(t),
as defined in Nathan and Rudner, and stores this value in the field `τ`.

There are a multitude of kwargs in the constructor which mainly relate to somewhat mundane
implementation details of the Fourier transforms etc. We provide reasonable defaults
when applicable. Their descriptions are as follows:

- `τ_guess::Float64`: Order of magnitude estimate for τ.
- `t_max_corr_factor::Float64=15.`: g(t) and C(t) will be set to zero identically in the ScaledInterpolation
        object for times outside of the domain [-`t_max_corr_factor*τ_`, `t_max_corr_factor*τ_`]
        where `τ_ ≈ τ` (see implementation for details).
        Also used to determine the number of sample points in the Gauss-Legendre expansion
        (see `gauss_legendre_factor`).
- `ω_max_zero_search_factor::Float64=10.`: Search for zeros of S(ω) - `ω_zero_threshold` from
        `-ω_max_zero_search_factor * high_frequency_cutoff(Sω)` to
        `ω_max_zero_search_factor * high_frequency_cutoff(Sω)`.
- `ω_zero_threshold::Float64=1e-12`: See `ω_max_zero_search_factor`.
- `gauss_legendre_factor::Float64=10.`: The number of samples used in the Gauss-Legendre
        quadrature is given by `ceil(Int, gauss_legendre_factor * t_max_corr * max(ω_max, abs(ω_min)))`,
        where `t_max_corr = t_max_corr_factor*τ_` with `τ_ ≈ τ` and `ω_max`, `ω_min` are the obtained
        zeros of S(ω) - `ω_zero_threshold`.
- `num_t_points::Int=10_000`: Number of time points in [-`t_max_corr`, `t_max_corr`] at which
        we evaluate (and interpolate between in the final ScaledInterpolation object).
"""
struct BathCorrelationFunction{Stype <: AbstractSpectralFunction} <: AbstractBathCorrelationFunction
    Sω::Stype  # Just the functional form (excluding a prefactor γ_prefactor) -- shall it be called smth else other than Sω?
    gt::ScaledInterpolation
    τ::Float64

    function BathCorrelationFunction(
        Sω::Stype,
        ;
        τ_guess::Float64,
        t_max_corr_factor::Float64=15.,
        ω_max_zero_search_factor::Float64=10.,
        ω_zero_threshold::Float64=1e-12,
        gauss_legendre_factor::Float64=10.,
        num_t_points::Int=10_000
    ) where {Stype <: AbstractSpectralFunction}
        ω_max_zero_search = ω_max_zero_search_factor * high_frequency_cutoff(Sω)
        ω, weights = _get_ω_grid(Sω, t_max_corr=t_max_corr_factor*τ_guess, ω_max_zero_search=ω_max_zero_search,
                                 ω_zero_threshold=ω_zero_threshold, gauss_legendre_factor=gauss_legendre_factor)

        gt_ = _get_time_dependent_correlation_function(ω_ -> evaluate_gω(Sω, ω_), ω, weights,
                                                       t_max_corr=t_max_corr_factor*τ_guess,
                                                       num_t_points=num_t_points)
        τ_ = _evaluate_τ(gt_)

        ω, weights = _get_ω_grid(Sω, t_max_corr=t_max_corr_factor*τ_, ω_max_zero_search=ω_max_zero_search,
                                 ω_zero_threshold=ω_zero_threshold, gauss_legendre_factor=gauss_legendre_factor)

        gt = _get_time_dependent_correlation_function(ω_ -> evaluate_gω(Sω, ω_), ω, weights,
                                                      t_max_corr=t_max_corr_factor*τ_,
                                                      num_t_points=num_t_points)
        τ = _evaluate_τ(gt)

        return new{Stype}(Sω, gt, τ)
    end
end

evaluate_gω(bcf::BathCorrelationFunction, ω) = evaluate_gω(bcf.Sω, ω)

function _evaluate_Γ(gt::ScaledInterpolation, γ_prefactor::Float64; t_max::Float64=Inf)
    return 4*γ_prefactor*(quadgk((t) -> abs(gt(t)), -t_max, t_max)[1])^2
end

function _evaluate_τ(gt::ScaledInterpolation; t_max::Float64=Inf)
    return quadgk((t) -> abs(t*gt(t)), -t_max, t_max)[1] / quadgk((t) -> abs(gt(t)), -t_max, t_max)[1]
end

evaluate_Γ(bcf::BathCorrelationFunction, γ_prefactor::Float64; kwargs...) = _evaluate_Γ(bcf.gt, γ_prefactor; kwargs...)
evaluate_τ(bcf::BathCorrelationFunction; kwargs...) = _evaluate_τ(bcf.gt; kwargs...)
