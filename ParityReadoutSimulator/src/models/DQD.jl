# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Basis convention :
* All majoranas are first then the QD levels.
* Arbitrary number of Majorana modes with static coupling between them
  (number set by size of input matrix A)
"""
struct DQD_Hamiltonian <: AbstractParameterizedHamiltonian{ComplexF64}
    H0::Matrix{ComplexF64}
    EC::Float64
    Dot_number_op::Matrix{ComplexF64}
    Exchange_op::Matrix{ComplexF64}

    function DQD_Hamiltonian(EC::Float64, ng::Float64, t0::Float64)
        Nfermions = 2
        selector = buildParitySelector(-1)
        fermion_basis = FermionBasis(Nfermions, selector)
        dot_number_op = fermion_site_number_op(fermion_basis, 2)
        exchange_op = fermion_exchange(fermion_basis, 1,2)

        H0 = t0*exchange_op
        for k in 1:size(H0, 1)
            H0[k,k] += EC*(dot_number_op[k,k] - ng)^2
        end

        return new(H0, EC, dot_number_op, exchange_op)
    end
end

function number_operator(H::DQD_Hamiltonian, ng_name::Symbol)
    @assert ng_name == :ng
    return H.Dot_number_op
end

function (H::DQD_Hamiltonian)()::Matrix{ComplexF64}
    return Matrix(H.H0)
end

function evaluate_CQ_DQD_model(;kwargs... )
    A::Float64 = kwargs[:A]
    fd::Float64 = kwargs[:fd]
    h0 = DQD_Hamiltonian(kwargs[:EC]::Float64, kwargs[:ng]::Float64, kwargs[:t0]::Float64)
    bath_correlation = get_bath_correlations_from_cache(;kwargs...)
    Hd = DrivenHamiltonian(h0(), -h0.EC*A* number_operator(h0, :ng), 2pi*fd)
    coupling_op = Dict(:ng=>kwargs[:ng]::Float64*I - number_operator(h0, :ng))
    ts, Ls = prep_Lindblad(
        Hd, coupling_op, bath_correlation, :ng
        ;
        δt_factor=kwargs[:δt_factor]::Float64, A=A
    )

    prefactor = calculate_CQ_prefactor(kwargs[:EC], kwargs[:lever_arm])/A
    Lns = [Lindblad_Fourier_components(ts, L, kwargs[:n_freq_lindblad]::Int) for L in Ls]

    CQ = periodic_steady_CQ(
        Hd,Lns,number_operator(h0,:ng);
        CQ_prefactor=prefactor,
        n_freq=kwargs[:n_freq],
    )
    return Dict(:CQ=>CQ)
end
