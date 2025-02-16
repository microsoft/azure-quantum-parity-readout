# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# +
import numpy as np
import xarray as xr
from xarray_einstats.stats import kurtosis
from sklearn.mixture import GaussianMixture
import sympy
import arviz as az
from pymc import Exponential, Model, Uniform, sample
from lmfit import Model as lmfit_model
from tqdm.auto import tqdm
from functools import partial
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def calc_kurtosis(da):
    """
    Calculate the kurtosis of Cq data array along the 'time' dimension.

    Parameters:
    da (xr.DataArray): The input data array for which to compute the kurtosis.

    Returns:
    xr.DataArray: An array containing the kurtosis values along the 'time' dimension,
                  with an updated attribute 'long_name' if one existed in the input data array.
    """
    kurtosis_da = kurtosis(
        da,
        dims="time",
        fisher=True,
        bias=True,
        nan_policy=None,
    )
    if "long_name" in da.attrs.keys():
        kurtosis_da.attrs["long_name"] = "K(" + da.attrs["long_name"] + ")"

    return kurtosis_da


def return_peaks(ds):
    """
    Identifies the peak values in the data series based on prominence.

    Parameters:
    ds (xr.DataArray): The input dataset from which peaks are to be identified.

    Returns:
    list: A list containing the values of 'V_lin_qd' at the identified peak positions.
    """
    ipeaks = find_peaks(ds.data, prominence=np.std(ds.data))[0]
    return [ds["V_lin_qd"].data[peak] for peak in ipeaks]


def determine_ng_range(ds, V_wire, i_peak, verbose=True):
    """
    Determines the range of gate voltages (V_gate) between two peaks for a specified wire voltage (V_wire).

    Parameters:
    ds (xr.DataArray): The input dataset.
    V_wire (float): The specified wire voltage for which the gate voltage range is to be determined.
    i_peak (int): Index of the peak of interest.
    verbose (bool): If True, plots the data and highlights the identified peaks and the determined range. Default is True.

    Returns:
    tuple: A tuple containing the lower and upper bounds of the gate voltage range around the specified peak.
    """
    peaks = return_peaks(
        ds.sel(V_wire=V_wire, method="nearest").Cq.mean(dim="B_perp").squeeze()
    )
    dV = (peaks[i_peak + 1] - peaks[i_peak - 1]) / 2
    if verbose:
        ds.sel(V_wire=V_wire, method="nearest").Cq.mean(dim="B_perp").plot()
        for peak in peaks:
            plt.axvline(x=peak)
        plt.axline(peaks[i_peak], c="r")
    return (peaks[i_peak] - dV / 2, peaks[i_peak] + dV / 2)


def _run_gmm(
    time,
    data,
    number_of_states: int = 2,
):
    """
    Applies Gaussian Mixture Modeling (GMM) to the given data to estimate the states.

    Parameters:
    time (np.ndarray): An array representing the time points.
    data (np.ndarray): The Cq data array on which GMM is to be applied.
    number_of_states (int): The number of states to be estimated by the GMM. Default is 2.

    Returns:
    tuple: A tuple containing the difference between the means of the two states, and the means of the two states.
           If GMM fitting is not feasible, returns a tuple of NaNs.
    """
    with np.errstate(invalid="ignore"):
        if any(np.isnan(data)) or all(np.diff(data) == 0):
            return (np.nan,) * 3
        data_reshaped = data.reshape(-1, 1)
        scores = list()
        models = list()
        for n_components in range(1, number_of_states + 1):
            for idx in range(10):
                gm = GaussianMixture(n_components=n_components, random_state=idx)

                gm.fit(data_reshaped)
                models.append(gm)
                scores.append(gm.aic(data_reshaped))

        model = models[np.argmin(scores)]
        state_predictions = model.predict(data_reshaped).astype(float)
        fitted_gm = model.fit(data_reshaped)

        if model.n_components > 1:
            mean1 = fitted_gm.means_[0][0]
            mean2 = fitted_gm.means_[1][0]
            mean1_, mean2_ = (mean1, mean2) if mean1 < mean2 else (mean2, mean1)

        else:
            mean1_, mean2_ = (fitted_gm.means_[0], fitted_gm.means_[0])

        return (
            mean2_ - mean1_,
            mean1_,
            mean2_,
        )


def run_gmm(da):
    """
    Applies the GMM fitting process to an xarray.DataArray and vectorizes the operation.

    Parameters:
    da (xr.DataArray): The input data array on which GMM is to be applied.

    Returns:
    xr.DataArray: The results of the GMM fitting process
    """
    return xr.apply_ufunc(
        _run_gmm,
        da["time"],
        da,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[
            [],
            [],
            [],
        ],
        vectorize=True,
    )


def _histogram(data, **kwargs):
    """
    Compute the histogram of a dataset.

    Parameters:
    data (np.ndarray): Input data array.
    **kwargs: Additional keyword arguments for numpy.histogram.

    Returns:
    tuple: A tuple containing the counts for each bin and the bin centers.
    """
    counts, edges = np.histogram(data, **kwargs)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return counts, bin_centers


def histogram(
    da: xr.DataArray,
    bins: int = 50,
    bin_range=None,
) -> xr.Dataset:
    """
    Create a histogram from an xarray.DataArray.

    Parameters:
    da (xr.DataArray): The input data array.
    bins (int): The number of bins for the histogram. Default is 50.
    # range (tuple): The range of values for the histogram. Not used currently.

    Returns:
    xr.Dataset: A dataset containing the histogram counts and bin centers.
    """

    counts_da, bins_da = xr.apply_ufunc(
        _histogram,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[["bins"], ["bins"]],
        vectorize=True,
        kwargs={
            "bins": bins,
            "range": (
                bin_range
                if bin_range is not None
                else (da.min().values, da.max().values)
            ),
        },
    )

    dim_to_reduce = [dim for dim in bins_da.dims if dim != "bins"]
    counts_da["bins"] = bins_da.reduce(np.mean, dim_to_reduce)
    counts_da["bins"].attrs.update(da.attrs)
    if "long_name" in da.attrs.keys():
        counts_da.attrs["long_name"] = da.attrs["long_name"] + " count"
    return counts_da


def _detect_switches(trace, peak1, peak2, r=0):
    """
    Detect switches between two states in a trace based on thresholds derived from peak values.

    Parameters:
    trace (xr.DataArray): The input data trace where switches are to be detected.
    peak1 (float): The value of the first peak.
    peak2 (float): The value of the second peak.
    r (float): A parameter to adjust the threshold calculation. Default is 0.

    Returns:
    xr.DataArray: An array indicating the states detected with 1 representing above high threshold,
                  0 representing below low threshold, and NaN for in-between values.
    """
    peak_high = np.max([peak1, peak2])
    peak_low = np.min([peak1, peak2])

    t2 = peak_high * (1 - r) + peak_low * r
    t1 = peak_low * (1 - r) + peak_high * r

    out = xr.where(trace > t2, 1, np.nan)
    out = xr.where(np.isnan(out) & (trace < t1), 0, out)
    return out


def digitize_traces(
    da: xr.DataArray, mean1_da: xr.DataArray, mean2_da: xr.DataArray, r=0
):
    """
    Digitize continuous data traces into binary states based on detected switches.

    Parameters:
    da (xr.DataArray): The input data array to be digitized.
    mean1_da (xr.DataArray): An array of the first mean values used for threshold calculations.
    mean2_da (xr.DataArray): An array of the second mean values used for threshold calculations.
    r (float): A parameter to adjust the threshold calculation. Default is 0.

    Returns:
    xr.DataArray: The digitized data array with binary states.
    """
    switches_da = xr.apply_ufunc(
        _detect_switches,
        da,
        mean1_da,
        mean2_da,
        input_core_dims=[["time"], [], []],
        output_core_dims=[["time"]],
        vectorize=True,
        kwargs={"r": r},
    )

    digital_traces_da = switches_da.ffill(dim="time")
    return digital_traces_da


def _dwell_time(time, digital_bool):
    """
    Calculate dwell times in up and down states from a digitized trace.

    Parameters:
    time (np.ndarray): Array representing the time points.
    digital_bool (np.ndarray): Array representing the digitized states (binary).

    Returns:
    tuple: Two arrays containing the dwell times in the up state and down state respectively.
    """
    time = time[:-1]

    dts = np.diff(digital_bool)
    event_times = time[np.abs(dts) > 0]
    dwell_time = np.diff(event_times)
    if digital_bool[0] == 0.0:
        return dwell_time[::2], dwell_time[1::2]  # up times, down times
    else:
        return dwell_time[1::2], dwell_time[0::2]


def extract_dwell_times(da):
    """
    Extract dwell times for up and down states from a digitized DataArray.

    Parameters:
    da (xr.DataArray): The input digitized data array with binary states (0 for down, 1 for up).

    Returns:
    tuple: Two lists containing the dwell times for up states and down states respectively.
    """
    dwells_up = []
    dwells_down = []
    if da.dims == ("time",):
        da = da.expand_dims(dim="dim0")
    stacked = da.stack(dim_0=[dim for dim in da.dims if dim != "time"])
    for d in stacked.dim_0:
        trace = stacked.sel(dim_0=d)
        time = trace.time.data
        digital_bool = trace.data

        dwell_time_up, dwell_time_down = _dwell_time(
            time=time, digital_bool=digital_bool
        )
        dwells_up += [dwell_time_up]
        dwells_down += [dwell_time_down]

    flat_dwells_up = [item for sublist in dwells_up for item in sublist]
    flat_dwells_down = [item for sublist in dwells_down for item in sublist]

    return flat_dwells_up, flat_dwells_down


def _dwells_fit_model(dwells, sample_rate, trace_len, plot_debug=True):
    """
    Fit a model to dwell times using Bayesian Inference.

    Parameters:
    dwells (list or np.ndarray): Array of dwell times to fit the model to.
    sample_rate (float): The sampling rate in seconds per sample. Default is 4.5e-6.
    trace_len (float): The total length of the trace in seconds. Default is 65e-3.
    plot_debug (bool): Whether to display a progress bar during sampling. Default is True.

    Returns:
    pd.DataFrame: A summary dataframe containing the Bayesian fit results.
    """
    # Calculate the longest and shortest feasible dwell time
    shortest_interval = sample_rate
    longest_interval = trace_len

    # Fit the resulting dataset using Bayesian Inference. This allows us to extract a model error
    rate_model = Model()
    with rate_model:
        # Define prior for scale parameter
        scale = Uniform("tau", lower=shortest_interval, upper=longest_interval)

        # Define likelihood
        _ = Exponential("dwells", scale=scale, observed=dwells)

        # Sample the distribution
        idata = sample(10000, progressbar=plot_debug)
    # Calculate stats with a 1 sigma interval, down to microsecond precision
    summary = az.summary(idata, round_to=6, hdi_prob=0.6827)
    return summary


def prepare_plot_dwells_bayesian(dwells, sample_rate, trace_len, model_fit=None):
    """
    Prepare Bayesian analysis results for plotting dwell times.

    Parameters:
    dwells (list or np.ndarray): Array of dwell times.
    model_fit (pd.DataFrame, optional): Precomputed Bayesian fit results. If None, the model
                                         will be fit to the provided dwell times.

    Returns:
    dict: A dictionary containing the mean and highest density interval (HDI) bounds.
    """
    if all(np.isnan(dwells)):
        return dict(
            mean=np.nan,
            hdi_minus=np.nan,
            hdi_plus=np.nan,
        )
    if model_fit is None:
        model_fit = _dwells_fit_model(dwells, sample_rate, trace_len, plot_debug=False)

    model_fit_res = dict(
        mean=model_fit["mean"].values[0],
        hdi_minus=model_fit["hdi_15.865%"].values[0],
        hdi_plus=model_fit["hdi_84.135%"].values[0],
    )
    return model_fit_res


def aggregate_dwells_from_frame(frame_ds):
    """
    Aggregate dwell times from a given data frame.

    Parameters:
    frame_ds (xr.Dataset): The input dataset containing 'Cq' variable.

    Returns:
    tuple: A tuple containing the following:
        - dwells_up (list): Dwell times when the signal is up.
        - dwells_down (list): Dwell times when the signal is down.
        - dwells_up_fit (dict): Bayesian fit results for the ups.
        - dwells_down_fit (dict): Bayesian fit results for the downs.
    """
    da = frame_ds["Cq"]

    _, mean1_da, mean2_da = run_gmm(da)
    digital_traces_da = digitize_traces(
        da,
        mean1_da,
        mean2_da,
    )

    sample_rate = da.time.diff(dim="time")[0].values
    trace_len = da.time.max(dim="time").values

    dwells_up, dwells_down = extract_dwell_times(digital_traces_da)

    dwells_up_fit = prepare_plot_dwells_bayesian(
        dwells=dwells_up, sample_rate=sample_rate, trace_len=trace_len
    )
    dwells_down_fit = prepare_plot_dwells_bayesian(
        dwells=dwells_down, sample_rate=sample_rate, trace_len=trace_len
    )

    return dwells_up, dwells_down, dwells_up_fit, dwells_down_fit


def _calc_chunk_dwell_times(param_value, da, param_name, Kurt_threshold):
    """
    Calculate dwell times for a specific slice of data with Kurtosis < Kurt_threshold based on a parameter value.

    Parameters:
    param_value (float): The value of the parameter to filter the data chunk.
    da (xr.DataArray): The input data array.
    param_name (str): The name of the dimension along which the data will be sliced.
    Kurt_threshold (float): The threshold for kurtosis to filter the traces.

    Returns:
    tuple: A tuple containing the following:
        - mean dwell-up time (float)
        - mean dwell-down time (float)
        - mean splitting value (float)
    """
    da_chunk = da.sel(**{param_name: param_value})

    kurtosis_da = kurtosis(
        da_chunk,
        dims="time",
        fisher=True,
        bias=True,
        nan_policy=None,
    )

    splitting, mean1_da, mean2_da = run_gmm(da_chunk)
    digital_traces_da = digitize_traces(
        da_chunk,
        mean1_da,
        mean2_da,
    )

    dwells_up, dwells_down = extract_dwell_times(
        digital_traces_da.where(kurtosis_da < Kurt_threshold)
    )

    sample_rate = da.time.diff(dim="time")[0].values
    trace_len = da.time.max(dim="time").values

    dwells_up_fit = prepare_plot_dwells_bayesian(
        dwells=dwells_up, sample_rate=sample_rate, trace_len=trace_len
    )
    dwells_down_fit = prepare_plot_dwells_bayesian(
        dwells=dwells_down, sample_rate=sample_rate, trace_len=trace_len
    )

    return dwells_up_fit["mean"], dwells_down_fit["mean"], splitting.mean().values


def calc_dwell_times_along_dim(da, param_name, Kurt_threshold):
    """
    Calculate dwell times along a specified dimension of a data array.

    Parameters:
    da (xr.DataArray): The input data array.
    param_name (str): The name of the dimension along which to calculate dwell times.
    Kurt_threshold (float): The threshold for kurtosis to filter the traces.

    Returns:
    tuple: A tuple containing arrays of the following:
        - mean dwell-up times along the specified dimension.
        - mean dwell-down times along the specified dimension.
        - mean splitting values along the specified dimension.
    """
    f = partial(
        _calc_chunk_dwell_times,
        da=da,
        param_name=param_name,
        Kurt_threshold=Kurt_threshold,
    )
    my_iter = da[param_name].values
    n = len(my_iter)

    results = list(
        tqdm(
            map(f, my_iter),
            total=n,
            leave=False,
            desc=f"dwell times extraction",
        )
    )

    (
        dwells_ups,
        dwells_downs,
        splittings,
    ) = zip(*results)

    return (
        np.array(dwells_ups),
        np.array(dwells_downs),
        np.array(splittings),
    )


def get_detuning(ds, N_qd, data_type):
    """
    Get the detuning voltage for a quantum dot from the dataset.

    Parameters:
    ds (xr.Dataset): The input dataset.
    N_qd (int): The quantum dot identifier.
    data_type (str): The type of data ('Measured' or other).

    Returns:
    np.ndarray: The detuning voltage array for the specified
    """
    Nqd_name = f"V_qd_{N_qd}_plunger_gate" if data_type == "Measured" else f"ng{N_qd}"
    Vqd = ds[Nqd_name].data
    return Vqd


def CPB_eigenvals(Ec=110.0, Ej=23.0):
    """
    Calculate the eigenvalues and Hamiltonians for a CPB.

    This function calculates the eigenvalues of the Hamiltonian for even and odd parities
    of a CPB given the charging energy (Ec) and the Josephson energy (Ej).

    Parameters:
    Ec (float): Charging energy of the CPB. Default is 110.
    Ej (float): Josephson energy of the CPB. Default is 23.

    Returns:
    tuple: A tuple containing the following:
        - ng (sympy.Symbol): Symbol for the offset charge.
        - even_energies (list): Eigenvalues of the even Hamiltonian.
        - even_hamiltonian (sympy.Matrix): Hamiltonian matrix for the even parity states.
        - odd_hamiltonian (sympy.Matrix): Hamiltonian matrix for the odd parity states.
    """
    ng = sympy.Symbol("ng", real=True)

    # Separately calculate the odd/even parities. This makes coloring the graph below much easier.
    even_hamiltonian = sympy.eye(2) * Ec
    odd_hamiltonian = sympy.eye(3) * Ec
    # Construct the even Hamiltonian
    for i in range(2):
        even_hamiltonian[i, i] *= (2 * i - ng) ** 2
    for i in range(1):
        even_hamiltonian[i, i + 1] = -Ej / 2
        even_hamiltonian[i + 1, i] = -Ej / 2
    # Construct the odd Hamiltonian. We need to model the three charge states to get all the excited states.
    for i in range(3):
        odd_hamiltonian[i, i] *= ((2 * i - 1) - ng) ** 2
    for i in range(2):
        odd_hamiltonian[i, i + 1] = -Ej / 2
        odd_hamiltonian[i + 1, i] = -Ej / 2

    even_energies = list(even_hamiltonian.eigenvals())

    return ng, even_energies, even_hamiltonian, odd_hamiltonian
