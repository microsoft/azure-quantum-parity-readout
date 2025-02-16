# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


function calculate_quasistatic_Lindblad_matrix(
    t::Float64,
    Ht,
    bath_coupling::AbstractBathCoupling,
)
    U0 = eigen(Hermitian(Ht(t)))
    ΔEmat = [En - Em for Em in U0.values, En in U0.values]
    return U0.vectors * (
        evaluate_gω.([bath_coupling.corr], ΔEmat) .* (U0.vectors'*bath_coupling(t)*U0.vectors)
    ) * U0.vectors'
end

function build_Lindblad_operator_fourier_component(
    ts::LinRange{Float64, Int64}, Lmats, n::Int
)
    period = ts[end]-ts[1]
    w = 2pi/period
    # We use a simple trapezoidal integration, assuming periodic boundaries.
    cs = step(ts)*exp.(1im*n*w*ts[1:end-1])
    Ln = sum([c*L for (c, L) in zip(cs, Lmats[1:end-1])])
    return Ln./period
end

function Lindblad_Fourier_components(ts, L, n_freq)
    return Dict(n=>build_Lindblad_operator_fourier_component(ts, L,n) for n in -n_freq:n_freq)
end
