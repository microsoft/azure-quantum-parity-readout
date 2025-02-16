# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Generic ED code allowing for parity or number conservation
abstract type AbstractSelector end

struct ParitySelector <: AbstractSelector
    P::Int
end
function (PS::ParitySelector)(v::AbstractVector)
    return (-1)^sum(v) == PS.P
end

struct EverythingSelector <: AbstractSelector end
function (ES::EverythingSelector)(v::AbstractVector)
    return true
end

function buildParitySelector(P::Int)
    if P == 0
        selector = EverythingSelector()
    else
        selector = ParitySelector(P)
    end
    return selector
end

# a chain of fermions
struct FermionBasis
    Nsites::Int

    basis::OrderedDict{Vector{Int},Int}

    function FermionBasis(Nsites, basis_selector)
        full_basis = Iterators.product(fill(0:1, Nsites)...)

        basis = OrderedDict{Vector{Int},Int}()
        for basis_vec_ in full_basis
            basis_vec = collect(basis_vec_)
            if basis_selector(basis_vec)
                basis[basis_vec] = length(basis) + 1
            end
        end

        return new(Nsites, basis)
    end
end

Base.length(basis::FermionBasis) = length(basis.basis)
Base.size(basis::FermionBasis) = length(basis.basis)

function fermion_site_number_op(basis::FermionBasis, m::Int)
    @assert 1 <= m <= basis.Nsites
    return diagm(0 => [v[m] for (v,_) in basis.basis])
end

function fermion_total_number_op(basis::FermionBasis)
    return diagm(0 => [sum(v) for (v,_) in basis.basis])
end

function fermion_site_parity_op(basis::FermionBasis, m::Int)
    @assert 1 <= m <= basis.Nsites
    return diagm(0 => [(-1)^v[m] for (v,_) in basis.basis])
end

function fermion_total_parity_op(basis::FermionBasis)
    return diagm(0 => [(-1)^sum(v) for (v,_) in basis.basis])
end

function _single_fermion_op(basis::FermionBasis, pos::Int, op_type::Symbol)
    @assert 1 <= pos <= basis.Nsites
    @assert op_type == :create || op_type == :annihilate

    R = zeros(length(basis), length(basis))
    for (state_vec, state_idx) in basis.basis
        if state_vec[pos] == (op_type == :create ? 0 : 1)
            new_state_vec = copy(state_vec)
            new_state_vec[pos] = (op_type == :create ? 1 : 0)

            new_state_idx = get(basis.basis, new_state_vec, nothing)

            parity = (-1)^sum(state_vec[1:pos-1])

            if new_state_idx !== nothing
                R[new_state_idx, state_idx] = parity
            end
        end
    end

    return R
end

fermion_create(basis::FermionBasis, pos::Int) = _single_fermion_op(basis, pos, :create)
fermion_annihilate(basis::FermionBasis, pos::Int) = _single_fermion_op(basis, pos, :annihilate)

function fermion_hop(basis::FermionBasis, to::Int, from::Int)
    @assert 1 <= from <= basis.Nsites
    @assert 1 <= to <= basis.Nsites
    @assert to != from

    R = zeros(length(basis), length(basis))
    for (state_vec, state_idx) in basis.basis
        if state_vec[from] == 1 && state_vec[to] == 0
            new_state_vec = copy(state_vec)
            new_state_vec[from] = 0
            new_state_vec[to]   = 1

            new_state_idx = get(basis.basis, new_state_vec, nothing)

            parity_sites = min(from,to):max(from,to)
            parity = -(-1)^sum(state_vec[parity_sites])

            if new_state_idx !== nothing
                R[new_state_idx, state_idx] = parity
            end
        end
    end

    return R
end

function fermion_exchange(basis::FermionBasis, pos1::Int, pos2::Int)
    return fermion_hop(basis, pos1, pos2) + fermion_hop(basis, pos2, pos1)
end

function fermion_exchange(basis::FermionBasis, pos1::Int, pos2::Int, J)
    return J*fermion_hop(basis, pos1, pos2) + conj(J)*fermion_hop(basis, pos2, pos1)
end

function fermion_pair_create(basis::FermionBasis, pos1::Int, pos2::Int)
    @assert 1 <= pos1 <= basis.Nsites
    @assert 1 <= pos2 <= basis.Nsites
    @assert pos1 != pos2

    if pos2 < pos1
        return -fermion_pair_create(basis, pos2, pos1)
    end

    R = zeros(length(basis), length(basis))
    for (state_vec, state_idx) in basis.basis
        if state_vec[pos1] == state_vec[pos2] == 0
            new_state_vec = copy(state_vec)
            new_state_vec[pos1] = 1
            new_state_vec[pos2] = 1

            new_state_idx = get(basis.basis, new_state_vec, nothing)

            parity_sites = min(pos1,pos2)+1:max(pos1,pos2)-1
            parity = (-1)^sum(state_vec[parity_sites])

            if new_state_idx !== nothing
                R[new_state_idx, state_idx] = parity
            end
        end
    end

    return R
end

function fermion_pair_annihilate(basis::FermionBasis, pos1::Int, pos2::Int)
    return -fermion_pair_create(basis, pos2, pos1)'
end

function fermion_anomalous_hop(basis::FermionBasis, pos1::Int, pos2::Int)
    return fermion_pair_create(basis, pos1, pos2) + fermion_pair_create(basis, pos1, pos2)'
end

function fermion_anomalous_hop(basis::FermionBasis, pos1::Int, pos2::Int, J)
    return J*fermion_pair_create(basis, pos1, pos2) + conj(J)*fermion_pair_create(basis, pos1, pos2)'
end

# For an explanation of conventions used here, please see test_QubitPlusQDs.md
# This is especially relevant for the signs that go into the Majorana-dot coupling terms

function majorana_hopping(basis::FermionBasis, pos1::Int, pos2::Int)
    if pos1 == pos2
        return im*I(size(basis))
    end

    i = div(pos1+1, 2)
    j = div(pos2+1, 2)

    if i == j
        return (pos1 < pos2 ? +1 : -1) * fermion_site_parity_op(basis, i)
    end

    if pos1 % 2 == 1 && pos2 % 2 == 1
        O = im*(fermion_hop(basis, i, j) + fermion_pair_create(basis, i, j))
    elseif pos1 % 2 == 1 && pos2 % 2 == 0
        O = -fermion_hop(basis, i, j) + fermion_pair_create(basis, i, j)
    elseif pos1 % 2 == 0 && pos2 % 2 == 1
        O = fermion_hop(basis, i, j) + fermion_pair_create(basis, i, j)
    else
        O = im*(fermion_hop(basis, i, j) - fermion_pair_create(basis, i, j))
    end
    return O + O'
end

function majorana_dot_coupling_op(basis::FermionBasis, majorana_index, dot_index, Nmaj)
    @assert 0 < majorana_index <= Nmaj
    @assert Nmaj %2 ==0 # Total number of Majoranas in basis should be even
    @assert dot_index+div(Nmaj,2) <= basis.Nsites

    m_fermion_index = div(majorana_index+1, 2)
    d_fermion_index = div(Nmaj,2)+dot_index
    if majorana_index % 2 == 1
        op = im*(
            fermion_hop(basis, m_fermion_index, d_fermion_index) -
            fermion_pair_annihilate(basis, m_fermion_index, d_fermion_index)
        )
    else
        op = (
            fermion_hop(basis, m_fermion_index, d_fermion_index) +
            fermion_pair_annihilate(basis, m_fermion_index, d_fermion_index)
        )
    end
    return op
end
