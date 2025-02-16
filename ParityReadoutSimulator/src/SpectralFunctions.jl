# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

abstract type AbstractSpectralFunction end

evaluate_gω(Sω::AbstractSpectralFunction, ω::Float64) = sqrt(Sω(ω))

# Phenomenological spectral function, corresponding to constant relaxation rate γ and finite temp
struct PhenomenologicalRelaxation <: AbstractSpectralFunction
    γ::Float64
    temperature::Float64
    ωc::Float64
    function PhenomenologicalRelaxation(;
        γ,
        temperature,
        ωc,
    )
        return new(γ, temperature, ωc)
    end
end

function (Sω::PhenomenologicalRelaxation)(ω::Float64)
    return Sω.γ * exp( -(ω / Sω.ωc)^2 ) / ( 1 + exp(-ω / Sω.temperature) )
end

high_frequency_cutoff(Sω::PhenomenologicalRelaxation) = Sω.ωc
