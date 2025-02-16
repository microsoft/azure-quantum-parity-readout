# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from functools import partial

from thermometry.helpers import _reshape

class ThermometryOptimizationVector:

    def __init__(self,
                 Cqs_v,
                 coulomb_diamond,
                 temp, offset, tm, Γ, shifts,
                 slice_count,
                 bounds,
                 thermometry_model,
                 global_shift=0,
                ):

        self.Cqs_v = Cqs_v
        self.slice_count = slice_count
        self.n_lines = len(slice_count)
        self.model = thermometry_model
        self.coulomb_diamond = coulomb_diamond

        self._param_order = ["temp", "leverarm", "tm", "Γ", "global_shift", "shifts", "re_offset", "im_offset"]
        self._param_shape = {
            "temp": (),
            "leverarm": (),
            "re_offset": (),
            "tm": (self.n_lines, ),
            "Γ": (self.n_lines, ),
            "global_shift": (),
            "shifts": self.slice_count,
            "im_offset": ()
        }

        (self.temp, self.leverarm,
         self.tm, self.Γ, self.global_shift,
         self.shifts, self.re_offset, self.im_offset) = self.fill_params(
            temp=temp, leverarm=self.coulomb_diamond.alpha,
            tm=tm, Γ=Γ, global_shift=global_shift,
            shifts=shifts, offset=offset
        )

        if "offset" in bounds.keys():
            bounds["re_offset"] = np.real(bounds["offset"])
            bounds["im_offset"] = np.imag(bounds["offset"])
            del bounds["offset"]

        if "global_shift" not in bounds.keys():
            bounds["global_shift"] = bounds["shifts"]

        assert all([i in bounds.keys() for i in self._param_order])
        self.bounds = bounds

    def fill_params(self, temp, leverarm, tm, Γ, global_shift, shifts, offset=None, re_offset=None, im_offset=None):
        if re_offset is None:
            if offset is None:
                raise Exception()
            re_offset, im_offset = np.real(offset), np.imag(offset)
        elif (im_offset is None):
                raise Exception()

        temp = self._fill(temp, self._param_shape["temp"])
        leverarm = self._fill(leverarm, self._param_shape["leverarm"])
        re_offset = self._fill(re_offset, self._param_shape["re_offset"])

        tm = self._fill(tm, self._param_shape["tm"])
        Γ = self._fill(Γ, self._param_shape["Γ"])

        global_shift = self._fill(global_shift, self._param_shape["global_shift"])

        shifts = self._fill(shifts, self._param_shape["shifts"])
        im_offset = self._fill(im_offset, self._param_shape["im_offset"])

        return temp, leverarm, tm, Γ, global_shift, shifts, re_offset, im_offset

    def _fill(self, val, shape):

        if isinstance(shape, (tuple, int)):
            return val + np.zeros(shape)

        if isinstance(val, (list, np.ndarray)):
            if all([len(v) == s for v, s in zip(val, shape)]):
                return val

        return [self._fill(val, s) for s in shape]

    def get_vector(self, params=None, params_ignored=set()):
        vec = []
        if params is None:
            param_func = self.__getattribute__
        else:
            def param_func(key):
                return params[key]

        for p in self._param_order:
            if p in params_ignored:
                continue
            v, s = param_func(p), self._param_shape[p]
            if isinstance(s, list):
                for vi in v:
                    vec.append(vi)
            elif np.shape(v) == ():
                vec.append([v])
            elif isinstance(v, np.ndarray):
                vec.append(v)
        return np.concatenate(vec)

    def get_bounds_vector(self, params_ignored=set()):

        temp, leverarm, tm, Γ, global_shift, shifts, re_offset, im_offset = self.fill_params(**{k: v[0] for k, v in self.bounds.items()})
        lower_bounds = self.get_vector(params=dict(
            temp=temp, leverarm=leverarm, tm=tm, Γ=Γ, global_shift=global_shift, shifts=shifts, re_offset=re_offset, im_offset=im_offset
        ), params_ignored=params_ignored)

        temp, leverarm, tm, Γ, global_shift, shifts, re_offset, im_offset = self.fill_params(**{k: v[1] for k, v in self.bounds.items()})
        upper_bounds = self.get_vector(params=dict(
            temp=temp, leverarm=leverarm, tm=tm, Γ=Γ, global_shift=global_shift, shifts=shifts, re_offset=re_offset, im_offset=im_offset
        ), params_ignored=params_ignored)

        return list(zip(lower_bounds, upper_bounds))

    def set_params_from_vector(self, vector, params_ignored=set()):
        i_start = 0
        for p in self._param_order:
            if p in params_ignored:
                continue
            s = self._param_shape[p]
            if isinstance(s, tuple):
                i_end = i_start + np.prod(s, dtype=np.int64)
                v = vector[i_start:i_end]
                v = _reshape(v, s)
                self.__setattr__(p, v)
                i_start = i_end
            else:
                vs = []
                for _s in s:
                    i_end = i_start + _s
                    v = vector[i_start:i_end]
                    v = _reshape(v, _s)
                    vs.append(v)
                    i_start = i_end
                self.__setattr__(p, vs)

        return self.get_params()

    def get_params(self):
        return dict(
            temp= self.get_temp(), # one per optim vector
            leverarm=self.get_leverarm(), # one per optim vector
            offset= self.get_offset(), # one per optim vector
            tm= self.get_tm(), # one per line
            Γ= self.get_Γ(), # one per line
            shifts= self.get_shifts(), # one per curve per line
        )

    def get_temp(self):
        return self.temp

    def get_leverarm(self):
        return self.leverarm

    def get_offset(self):
        return self.re_offset + 1j*self.im_offset

    def get_tm(self):
        return self.tm

    def get_Γ(self):
        return self.Γ

    def get_shifts(self, line_index=None):
        if line_index is None:
            return self.shifts
        assert line_index < self.n_lines
        return self.shifts[line_index] + self.global_shift

    def _minimize(self, costfunc,
                  params_ignored=set(),
                  method="L-BFGS-B", options={"maxiter": 30, "disp": False}):

        minimization = minimize(costfunc,
            self.get_vector(params_ignored=params_ignored),
            bounds=self.get_bounds_vector(params_ignored=params_ignored),
            options=options,
            method=method,
        )

        self.minimization = minimization
        self.success = minimization.success
        self.message = minimization.message

        return self.set_params_from_vector(minimization.x, params_ignored=params_ignored)

    def minimize(self, params_ignored=set(), scale_imag=1, method="L-BFGS-B", options={"maxiter": 30, "disp": False}):
        """
            Method to call to fit electron temperature and other parameters from a DQD model to
            measured DQD.
        """
        cf = partial(self.cost_function, params_ignored=params_ignored, scale_imag=scale_imag)
        return self._minimize(cf, params_ignored=params_ignored, method=method)

    def cost_function(self, params, params_ignored=set(), scale_imag=1):
        """
            The cost function used to fit the model to the measured DQD data.
            `params` is a vector returned by scipy minimize. It's a concatenated representation of all parameters for a dataset.
            `params_ignored` is a collection of parameters we don't want to fit in this round of fitting. Helpful if you want to adjust
            shifts at first and then drop them to make the optimization vector smaller for speed-up/convergence.
        """

        # a vector of the data we want to fit together
        Cqs_v = self.Cqs_v

        cost = 0.

        # update the inner optimization vector based on the optimizer output vector
        # without updating ignored parameters
        self.set_params_from_vector(params, params_ignored)

        # Get the parameters we fit as a vector
        temp = self.get_temp()
        leverarm = self.get_leverarm()
        offset = self.get_offset()
        tm_v = self.get_tm()
        Γ_v = self.get_Γ()
        shifts_v = self.get_shifts()

        leverarm_xscaling = self.coulomb_diamond.xscale(self.model.coulomb_diamond.Ec)

        # Loop over the fit parameters and data
        for tm, Γ, shifts, Cqs in zip(tm_v, Γ_v, shifts_v, Cqs_v):

            # evaluate the fitted model curve for the current data slice
            modelCq = self.model.curve(
                tm,
                temp,
                Γ,
            ) * self.coulomb_diamond.yscale(target_alpha=self.model.coulomb_diamond.alpha, alpha=leverarm)

            # scale the imaginary data if requested
            if scale_imag != 1:
                modelCq = np.real(modelCq) + 1j*scale_imag*np.imag(modelCq)
                offset = np.real(offset) + 1j*scale_imag*np.imag(offset)

            # interpolate the model
            modelCq_itp = interp1d(self.model.V0s, modelCq, fill_value = "extrapolate")

            # loop over each individual curve in a set of curves to be fit together
            # update the cost function based on the norm of the difference after applying
            # the necessary shifts
            for shift, (x, rCq, iCq) in zip(shifts, Cqs):
                Cq_theo = modelCq_itp(x*leverarm_xscaling - shift)
                cost += np.linalg.norm(rCq + 1j*scale_imag*iCq - offset - Cq_theo)

        return cost

    def plot(self):
        Cqs_v = self.Cqs_v

        temp = self.get_temp()
        leverarm = self.get_leverarm()
        offset = self.get_offset()
        tm_v = self.get_tm()
        Γ_v = self.get_Γ()
        shifts_v = self.get_shifts()

        fig, ax = plt.subplots(len(Cqs_v), 2, figsize=(8, 3*len(Cqs_v)))
        if len(Cqs_v)==1:
            ax = np.array([ax])

        leverarm_xscaling = self.coulomb_diamond.xscale(self.model.coulomb_diamond.Ec)

        for iCqs, tm, Γ, shifts, Cqs in zip(range(len(Cqs_v)), tm_v, Γ_v, shifts_v, Cqs_v):

            modelCq = self.model.curve(
                tm,
                temp,
                Γ,
            ) * self.coulomb_diamond.yscale(target_alpha=self.model.coulomb_diamond.alpha, alpha=leverarm)

            for shift, (x, rCq, iCq) in zip(shifts, Cqs):
                ax[iCqs, 0].plot(x*leverarm_xscaling - shift, rCq - np.real(offset))
                ax[iCqs, 1].plot(x*leverarm_xscaling - shift, iCq - np.imag(offset))

            ax[iCqs, 0].plot(self.model.V0s, np.real(modelCq), color="k",
                            label="t0={}μeV\nΓ={}μeV".format(
                             np.round(tm, decimals=2),
                             np.round(Γ, decimals=2)
                         ))
            ax[iCqs, 1].plot(self.model.V0s, np.imag(modelCq), color="k")

            ax[iCqs, 0].set_xlabel("$N_g$")
            ax[iCqs, 1].set_xlabel("$N_g$")

            ax[iCqs, 0].set_ylabel("Re $C_Q$ [fF]")
            ax[iCqs, 1].set_ylabel("Im $C_Q$ [fF]")

            ax[iCqs, 0].legend(fontsize=8)

            ax[iCqs, 0].annotate(f"Line #{iCqs}", xy=(0, 0.5), xytext=(-ax[iCqs, 0].yaxis.labelpad - 5, 0),
                xycoords=ax[iCqs, 0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

        ax[0, 0].set_title("Real")
        ax[0, 1].set_title("Imag")
        fig.suptitle(f"Extracted $T_e={np.round(temp, 0)}$mK")
        fig.tight_layout()
