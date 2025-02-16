# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

function calculate_CQ_prefactor(Ec, lever_arm; units=:two_pi_GHz)
    return lever_arm^2*Units.CΣ_from_Ec(Ec; units=units)
end

function build_Floquet_superop(H0, V1, _Ls::AbstractVector{T}, n_freq, ω) where {T<: AbstractMatrix}
    Ls = [Dict(0=>L) for L in _Ls]
    return build_Floquet_superop(H0, V1, Ls, n_freq, ω)
end

function _AρB(A,B)
    kron(transpose(B),A)
end

function build_Floquet_superop(H0, V1, Ls::AbstractVector{T}, n_freq, ω) where {T<: AbstractDict}
    # Let's allow for multiple Fourier components, Ls is a collection of Lindblad operators
    # each Lindblad operator is described by a dict of fourier components n=>Ln
    nLs = [sort(collect(keys(L))) for L in Ls]
    dim = size(H0,1)
    OD1 = spdiagm(-1=>ones(2n_freq))
    I_dim = spdiagm(0=>ones(dim))
    I_dim2 = spdiagm(0=>ones(dim^2))
    I_freq = spdiagm(0=>ones(2n_freq+1))
    nω = ω*spdiagm(0=>-n_freq:n_freq)

    # Loop over diagonals:
    rhs = - kron(nω, I_dim2)
    for d in -2n_freq:2n_freq
        K = zeros(ComplexF64, size(H0))
        for (ns, L) in zip(nLs, Ls)
            for n in ns, l in ns
                if n-l == d
                    # @show d, n, l
                    K += -0.5im*(L[n]'*L[l])
                end
            end
        end
        if norm(K)==0
            rhs_d = spzeros(ComplexF64, dim^2, dim^2)
        else
            rhs_d = _AρB(K,I_dim) + _AρB(I_dim,K)
        end
        for (ns, L) in zip(nLs, Ls)
            for n in ns, l in ns
                if n-l == d
                    rhs_d += 1im*_AρB(L[l],L[n]')
                end
            end
        end
        D = spdiagm(d=>ones(2n_freq+1-abs(d)))
        if d==0
            rhs_d += _AρB(H0,I_dim) - _AρB(I_dim,H0)
        end
        rhs += kron(D, rhs_d)
    end
    OV1 = 1im*kron(OD1, _AρB(V1,I_dim) - _AρB(I_dim,V1))
    rhs += OV1 + OV1'
end

function periodic_steady_CQ(
    Hd::DrivenHamiltonian, Lns, N_op;
    CQ_prefactor, # Cg^2/Csum
    n_freq=10, #number of frequency components
    verbose=false,
    sparse_solve=true,
)
    dim = size(Hd.H0,1)
    rhs = build_Floquet_superop(Hd.H0, Hd.V, Lns, n_freq, Hd.ωd)
    if sparse_solve
        vals, vecs = eigs(rhs,which=:LM,sigma=1e-6,nev=1)
        rho = reshape(vecs,(dim,dim,2*n_freq+1))
        if abs.(vals)[] > 1e-12
            @show vals
        end
    else
        rho = reshape(nullspace(Matrix(rhs)),(dim,dim,2*n_freq+1))
    end
    rho0 = @view(rho[:,:,n_freq+1])
    rho ./= tr(rho0)
    rho1 = @view rho[:,:,n_freq+2]
    return -1im*2*CQ_prefactor*tr(N_op*rho1)
end
