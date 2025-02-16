# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Basis convention :
* All majoranas are first then the QD levels.
* Arbitrary number of Majorana modes with static coupling between them
  (number set by size of input matrix A)
"""
struct MPR_TQD_Hamiltonian <: AbstractParameterizedHamiltonian{ComplexF64}
    H0::Matrix{ComplexF64}
    EC::Vector{Float64}
    Eshift::Ref{Float64}
    Dot_number_ops::Vector{Matrix{ComplexF64}}
    Majorana_ops::Matrix{Matrix{ComplexF64}}

    function MPR_TQD_Hamiltonian(
        ECs::Vector{Float64}, A::AbstractMatrix{Float64}, P::Int;
        t12::Float64, t23::Float64, tm1::Float64, tm2::Float64, tm1_phase::Float64, ng1::Float64, ng2::Float64, ng3::Float64
    )
        Nmaj = size(A,1)
        @assert Nmaj % 2 == 0
        @assert length(ECs)==3
        @assert norm(A+A')<1e-14

        Nfermions = 3 + div(Nmaj,2)
        selector = buildParitySelector(P)
        fermion_basis = FermionBasis(Nfermions, selector)
        dot_indices = collect(div(Nmaj,2)+1:Nfermions) # Last three fermions
        dot_number_ops = [fermion_site_number_op(fermion_basis, k) for k in dot_indices]
        Majorana_ops = [
            majorana_hopping(fermion_basis, p1, p2) for p1 in 1:Nmaj, p2 in 1:Nmaj
        ]

        H0 = sum(
            A[p1, p2] * majorana_hopping(fermion_basis, p1, p2) for p1 in 1:Nmaj, p2 in 1:Nmaj
        )

        Eshift = Ref{Float64}()
        Eshift[] = 0.0

        # QD-QD couplings
        dot_coupling_ops = [fermion_exchange(fermion_basis, i,j) for (i,j) in zip(dot_indices[1:end-1], dot_indices[2:end])]
        axpy!(t12, dot_coupling_ops[1], H0)
        axpy!(t23, dot_coupling_ops[2], H0)

        # QD MZM couplings
        Majorana_coupling_ops = [
            majorana_dot_coupling_op(fermion_basis, 1, 1, Nmaj),
            majorana_dot_coupling_op(fermion_basis, 2, 3, Nmaj)
        ]
        t_img = (tm1 * exp(im*π*tm1_phase), tm2)
        for (j, t) in enumerate(t_img)
            axpy!(t, Majorana_coupling_ops[j], H0)
            axpy!(conj(t), Majorana_coupling_ops[j]', H0)
        end
        for k in 1:size(H0, 1)
            H0[k,k] += ECs[1]*(dot_number_ops[1][k,k] - ng1)^2
            H0[k,k] += ECs[2]*(dot_number_ops[2][k,k] - ng2)^2
            H0[k,k] += ECs[3]*(dot_number_ops[3][k,k] - ng3)^2
        end

        return new(
            H0, ECs, Eshift, dot_number_ops,  Majorana_ops
        )
    end
end

function get_Eshift(H::MPR_TQD_Hamiltonian)
    return H.Eshift[]
end
function update_Eshift!(H::MPR_TQD_Hamiltonian, new_shift::Float64)
    H.Eshift[] = new_shift
end

function number_operator(H::MPR_TQD_Hamiltonian, ng_name::Symbol)
    index = (;ng1=1, ng2=2, ng3=3)[ng_name]
    return H.Dot_number_ops[index]
end

function (H::MPR_TQD_Hamiltonian)(Hmat::Matrix{ComplexF64})::Nothing
    # Static quadratic Majorana terms
    copyto!(Hmat, H.H0)

    # QDs charging energies and global energy shift
    Eshift = get_Eshift(H)
    for k in 1:size(Hmat, 1)
        Hmat[k,k] += Eshift
    end

    return nothing
end

function (H::MPR_TQD_Hamiltonian)()::Matrix{ComplexF64}
    Hmat = similar(H.H0)
    H(Hmat)
    return Hmat
end

function calculate_bcfs(; temperature, ωc, γng)
    S_ng = PhenomenologicalRelaxation(temperature=temperature, ωc=ωc, γ=γng)
    return BathCorrelationFunction(S_ng, τ_guess=1.)
end

function prep_Lindblad(Hd, coupling_ops, bcf_ng, driven_gate_charge, ;δt_factor=0.05, A)
    X_ops = AbstractBathCoupling[
        PeriodicChargeNoiseBathCoupling(coupling_ops[driven_gate_charge], Hd.ωd, A, bcf_ng)
    ]
    other_ngs = setdiff(collect(keys(coupling_ops)), [driven_gate_charge])
    for ng in other_ngs
        push!(X_ops, StaticBathCoupling(coupling_ops[ng], bcf_ng))
    end
    ts = get_Lindblad_time_grid(Hd, δt_factor)
    Lmats = [[calculate_quasistatic_Lindblad_matrix(t, Hd, X) for t in ts] for X in X_ops]

    return ts, Lmats
end

function _skew_sym_matrix(v)
    return [0 v; -v 0]
end

function _build_A_matrix(; kwargs...)
    E_M = kwargs[:E_M]::Float64
    if :E_0 in keys(kwargs)
        E_0 = kwargs[:E_0]::Float64
        u1::Float64 = kwargs[:u1]
        u2::Float64 = kwargs[:u2]
        b12 = _skew_sym_matrix(E_M)
        b34 = _skew_sym_matrix(E_0)
        off_block = [u1 0; 0 u2]
        return 0.5.*[b12 off_block; -off_block' b34]
    else
        return 0.5.*_skew_sym_matrix(E_M)
    end
end

function build_ZMPR_TQD(; kwargs...)
    ECs = [kwargs[Symbol("EC$i")]::Float64 for i in 1:3]
    A = _build_A_matrix(; kwargs...)
    fixed_model_parameters=Dict(k=>kwargs[k]::Float64 for k in [:tm1, :tm2, :t12,:t23, :ng1, :ng2, :ng3, :tm1_phase])
    H_ZMPR = MPR_TQD_Hamiltonian(
        ECs, A, kwargs[:parity]::Int;
        fixed_model_parameters...
    );
    return H_ZMPR
end

function get_bath_correlations_from_cache(;kwargs...)
    bath_parameters=Dict(k=>kwargs[k]::Float64 for k in [:temperature,:ωc, :γng])
    if !(:bcfs in keys(kwargs)) || isnothing(kwargs[:bcfs])
        println("Calculating bcf. Consider caching it.")
        bath_correlation = calculate_bcfs(; bath_parameters...)
    else
        bath_correlation = kwargs[:bcfs]
        @assert all([kwargs[:bath_parameters][k] ≈ v for (k,v) in bath_parameters])
    end
    return bath_correlation
end

function evaluate_CQ_TQD_model(;phi, kwargs... )
    A::Float64 = kwargs[:A]
    fd::Float64 = kwargs[:fd]
    h0 = build_ZMPR_TQD(;kwargs..., tm1_phase=phi)
    bath_correlation = get_bath_correlations_from_cache(;kwargs...)
    Hd = DrivenHamiltonian(h0(), -h0.EC[2]*A* number_operator(h0, :ng2), 2pi*fd)
    coupling_ops = Dict(ng=>kwargs[ng]::Float64*I - number_operator(h0, ng) for ng in  [:ng1, :ng2,:ng3])
    ts, Ls = prep_Lindblad(
        Hd, coupling_ops, bath_correlation, :ng2
        ;
        δt_factor=kwargs[:δt_factor]::Float64, A=A
    )

    prefactor = calculate_CQ_prefactor(kwargs[:EC2], kwargs[:lever_arm])/A
    Lns = [Lindblad_Fourier_components(ts, L, kwargs[:n_freq_lindblad]::Int) for L in Ls]

    CQ = periodic_steady_CQ(
        Hd,Lns,number_operator(h0,:ng2);
        CQ_prefactor=prefactor,
        n_freq=kwargs[:n_freq],
    )
    return Dict(:CQ=>CQ)
end

function cache_bath_correlation_function(params)
    bath_parameters=OrderedDict(k=>params[k]::Float64 for k in [:temperature,:ωc, :γng])
    bath_correlation = calculate_bcfs(; bath_parameters...)
    params[:bcfs] = bath_correlation
    params[:bath_parameters] = bath_parameters
end
