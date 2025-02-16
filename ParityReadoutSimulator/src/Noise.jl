# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import Distributions
using FFTW

struct TelegraphTrajectory
    duration::Float64
    jumps::Vector{Float64}

    function TelegraphTrajectory(duration::Float64, τ_qpp::Float64)
        jumps = Float64[]
        distr = Distributions.Exponential(τ_qpp)

        push!(jumps, rand(distr))
        while jumps[end] < duration
            push!(jumps, jumps[end] + rand(distr))
        end

        return new(duration, jumps)
    end
end

function (tj::TelegraphTrajectory)(t::Float64)::Float64
    p = searchsortedfirst(tj.jumps, t)
    return (-1.)^p
end
duration(tj::TelegraphTrajectory) = tj.duration

function generate_noise_trajectory(
    S, T::Float64, ωmax::Float64
)::Tuple{Vector{Float64}, Vector{Float64}}
    Δω = π/T

    # factor of 2 here is related to some convention in FFTW.
    ω_grid = (Δω/2:Δω:ωmax)
    fourier_amp = sqrt.(S.(ω_grid))
    rand_phases = 2π*rand(length(ω_grid))

    yω = fourier_amp .* cos.(rand_phases) / sqrt(T)
    # FFTW.REDFT10 has the right symmetries, given that we're not providing a zero-frequency term:
    # http://www.fftw.org/fftw3_doc/Real-even_002fodd-DFTs-_0028cosine_002fsine-transforms_0029.html
    # http://www.fftw.org/fftw3_doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html
    yt = FFTW.r2r(yω, FFTW.REDFT10)

    t_grid = (0:length(yt)-1) * (π/ωmax)

    return t_grid, yt
end
