# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from argparse import ArgumentError


class CoulombDiamond():

    def __init__(self, *args, Ec=None, alpha=None, dV=None):

        if len(args) > 0:
            raise ArgumentError("CoulombDiamond() accepts no positional arguments, only 2 of: {Ec, alpha, dV}")

        if Ec is not None and alpha is not None:
            self.from_Ec_and_alpha(Ec, alpha)
        elif alpha is not None and dV is not None:
            self.from_alpha_and_dv(alpha, dV)
        elif Ec is not None and dV is not None:
            self.from_Ec_and_dv(Ec, dV)
        else:
            ArgumentError("CoulombDiamond() requires exactly 2 of {Ec, alpha, dV}")

    def __repr__(self):
        return f"CoulombDiamond(\n\tEc={self.Ec},\n\talpha={self.alpha},\n\tdV={self.dV}\n)"

    def from_Ec_and_alpha(self, Ec, alpha):
        self.Ec = Ec
        self.alpha = alpha
        self.dV = self.alpha_from_Ec_and_alpha(Ec, alpha)
        self.definition = {"Ec", "alpha"}

    def from_alpha_and_dv(self, alpha, dV):
        self.alpha = alpha
        self.dV = dV
        self.Ec = self.Ec_from_alpha_and_dV(alpha, dV)
        self.definition = {"alpha", "dV"}

    def from_Ec_and_dv(self, Ec, dV):
        self.Ec = Ec
        self.dV = dV
        self.alpha = self.alpha_from_Ec_and_dV(Ec, dV)
        self.definition = {"Ec", "dV"}

    def Ec_from_alpha_and_dV(self, alpha, dV):
        return 0.5 * alpha * dV

    def alpha_from_Ec_and_dV(self, Ec, dV):
        return 2*Ec/dV

    def alpha_from_Ec_and_alpha(self, Ec, alpha):
        return 2*Ec/alpha

    def relative_alpha_from_dV(self, dV):
        # assuming Ec constant, calculate effective leverarm from coulomb peak spacing
        return self.alpha * self.dV / dV

    def xscale(self, target_Ec, alpha=None, dV=None):
        if alpha is None:
            alpha = self.alpha
        if dV is None:
            dV = self.dV

        return dV * target_Ec / self.Ec

    def yscale(self, target_alpha, alpha=None):
        if alpha is None:
            alpha = self.alpha

        return alpha**2 / target_alpha**2

class CoulombDiamondForSecondaryGate(CoulombDiamond):
    # assuming constant Ec, new measured peak spacing can determine leverarm of other gates

    def __init__(self, *args, dVg, Ec=None, alpha=None, dV=None):
        super().__init__(*args, Ec=Ec, alpha=alpha, dV=dV)

        self.dVg = dVg
        self.alpha_g = self.relative_alpha_from_dV(dVg)

    def xscale(self, target_Ec, alpha=None):
        return super().xscale(target_Ec, alpha=alpha, dV=self.dVg)

    def __repr__(self):
        return f"CoulombDiamondForSecondaryGate(\n\tEc={self.Ec},\n\talpha={self.alpha},\n\tdV={self.dV},\n\tdVg={self.dVg}\n)"
