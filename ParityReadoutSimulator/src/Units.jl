# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

module Units
    using Unitful
    using PhysicalConstants.CODATA2018

    # Energy conversions
    const meV_to_GHz = uconvert(Unitful.NoUnits, u"meV"*u"ns"/ReducedPlanckConstant)
    const GHz_to_meV = 1.0/meV_to_GHz
    const μeV_to_GHz = uconvert(Unitful.NoUnits, u"μeV"*u"ns"/ReducedPlanckConstant)
    const GHz_to_μeV = 1.0/μeV_to_GHz

    const ms_to_ns = 1e6
    const μs_to_ns = 1e3

    # Temperature conversions
    const mK_to_GHz  = uconvert(Unitful.NoUnits, BoltzmannConstant*u"mK"/ReducedPlanckConstant*u"ns")

    # Capacitance conversions
    const Cunit = uconvert(u"fF", ElementaryCharge^2/(2*ReducedPlanckConstant/u"ns"))
    const Cunit_fF = upreferred(Cunit/u"fF")

    function CΣ_from_Ec(Ec; units=:two_pi_GHz)
        @assert units==:two_pi_GHz
        return (Cunit_fF/Ec)
    end
end # module Units
