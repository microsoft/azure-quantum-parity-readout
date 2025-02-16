import numpy as np
from copy import deepcopy
from scipy.optimize import curve_fit

import sys
sys.path.append("../..")

from thermometry import ComplexOSDThermometryModel, ThermometryOptimizationVector, find_houghlines

from thermometry.coulomb_diamond import CoulombDiamondForSecondaryGate

def saturation_temperature_model(p, Ts, T):
    """
        Fit function for saturation temperature with exponent p that dictates primary mechanism for heating.
    """
    return (Ts ** p + T ** p) ** (1 / p)

def fit_tsat_curve_errorbars(data_to_plot, power):
    """
        Helper function to fit Tsat from the model curve above
    """
    X, Y = list(data_to_plot.keys()), list(data_to_plot.values())
    X, Y = tmc_to_tpuck(np.array(X)), np.array(Y)
    X, Y = np.repeat(X, np.shape(Y)[1]), np.reshape(Y, (np.size(Y)))

    def temp_minimizer(X, Ts):
        return saturation_temperature_model(p=power, T=X, Ts=Ts)

    fitted_Te, pcov = curve_fit(temp_minimizer, X, Y, bounds=(10, 300))

    fitted_Te_err_bar = np.sqrt(np.diag(pcov))

    return fitted_Te, fitted_Te_err_bar

def parallelized_find_houghlines(rZ, debug=False):

    lines = find_houghlines(
        rZ,
        resize_vert=100,
        norm_kwargs=dict(debug=debug),
    )

    return lines

def numerically_estimate_err_bar_dist(fits, N=10_000):
    """
        Calculate error bars from the fits across different leverarms
    """

    np.random.seed(42)

    X = np.array([])
    for mean, std in fits:
        X = np.concatenate([X, np.random.normal(mean, std, N)])

    return np.mean(X), np.std(X)


### Calibrating T_Puck to T_MC
def tmc_to_tpuck(Tmeas):
    """
        Fit of T_Puck to T_MC
    """
    return np.array([np.sqrt(0.989 * T ** 2 + 668) for T in Tmeas])

# Measured Data of T_Puck vs. T_MC from sensors in a separate cooldown
Tmc = [10, 52.2, 58.4, 48.8, 64.6, 79.92, 79.69, 99.11, 158.16]
Tpuck = [34.7, 58.1, 62.9, 55.4, 68.9, 83.27, 83.4, 101.79, 159.75]

def parallel_fit_DQD_model(inputs):
    """
        A helper function to define the minimization problem and run it, will be called
        from within ProcessPoolExecutor on multiple threads
    """

    slice_idx = inputs[0] # an index useful for plotting later (ds_path, cutter_value (choice of gate_gate map), leverarm value)
    _Cqs_v = deepcopy(inputs[1]) # the slices to be fit
    temp_init_condition = inputs[2] # an initial condition for temperature, we start somewhere close to T_MC to avoid getting stuck in local minima

    leverarm = slice_idx[2]

    n_lines = len(_Cqs_v)
    if n_lines == 0: # if there are no slices to be fit, return None
        return None

    # Initialize an interpolator over the model data to fit to
    thermometry_model = ComplexOSDThermometryModel(V0s=np.linspace(-0.8, 0.8, 201))

    # Defined the rescaling of the data to match simulations based on experimental data acquired.
    # dV and dVg are peak spacings acquired from sweeping the plunger gate and V_DG2
    coulomb_diamond = CoulombDiamondForSecondaryGate(alpha=leverarm, dV=0.0012, dVg=0.0038)

    # pre-processing to remove large shifts, we will be fitting the small shifts again
    shifts = [[i[0][np.argmax(i[1])] for i in j] for j in _Cqs_v]
    for j, __j in enumerate(_Cqs_v):
        for k, __k in enumerate(__j):
            _Cqs_v[j][k] = (_Cqs_v[j][k][0] - shifts[j][k], _Cqs_v[j][k][1], _Cqs_v[j][k][2])

    optim_vector = ThermometryOptimizationVector(
        _Cqs_v,
        thermometry_model=thermometry_model,
        temp=temp_init_condition,
        coulomb_diamond=coulomb_diamond,
        offset=  -1-1j,
        tm=4, Γ=10,
        shifts=0.01,#0.01,
        slice_count=[len(i) for i in _Cqs_v],
        bounds=dict(
            temp=(25., 300.),
            leverarm=(0.28, 0.5),
            offset=(-100-100j, 100+100j),
            tm=(1, 15),
            Γ=(0.1, 20.),
            shifts=(-0.1, 0.1)
        )
    )

    # First fit the individual shifts
    optim_vector.minimize(params_ignored={"leverarm", "global_shift", "temp", "tm", "Γ"}, scale_imag=1, options={"maxiter": 10_000, "maxfun": 100_000})

    # Now fit all the parameters except for the shifts
    optim_vector.minimize(params_ignored=["shifts", "leverarm"], scale_imag=1, options={"maxiter": 1_000_000, "maxfun": 1_000_000})

    # Note, both minimizations didn't fit leverarm, we instead will fit the temps as a
    # function of the input leverarm since we can extract that value using a separate technique

    return slice_idx, optim_vector

def prepare_data_into_dict(opts_for_plot, leverarm):
    """
        Get all the fitted temperatures across different datasets and gate-gate maps in a dict (key goes with temperature) of lists (index goes with gate-gate map).
    """
    data_to_plot = {}
    for i in opts_for_plot[leverarm]:
        idx, temp, ovec = i

        if temp not in data_to_plot.keys():
            data_to_plot[temp] = []

        data_to_plot[temp].append(ovec.get_temp())
    return data_to_plot
