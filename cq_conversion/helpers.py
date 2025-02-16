# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from numpy.typing import NDArray
from enum import Enum, unique
from typing import NamedTuple
import numpy as np

def get_findex(fs, freq):
    return np.argmin(np.abs(fs - freq))

def lin_to_dB(x):
    return 20 * np.log10(
        abs(x)
    )

def create_frequency_mask(
    frequency: NDArray, f_low: float, f_high: float
) -> list[bool]:
    """Returns a boolean mask indicating frequencies between f_low and f_high"""
    f_mask = [f_low < freq < f_high for freq in frequency]
    if not np.any(f_mask):
        raise ValueError(
            f"No data frequencies found between {f_low*1e-6:.1f}MHz and {f_high*1e-6:.1f}MHz"
        )
    return f_mask
