import os
import pickle

import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset

from .config import DATA_DIR, SAVE_DIR

FILL_VALUE = 99999
MASK_THRESHOLD = 9999
ULIM, LLIM = 30, -30  # latitude rows kept, from 60S to 60N on the 1x1 grid
UPSCALE_RATE = 5  # from 1x1 to 5x5

# unit conversion applied to the raw values
CONVERT = {
    "pr": lambda x: x * 87400,  # from kg m-2 s-1 to mm/day
    "tsl": lambda x: x - 273,  # from K to degC
}


def input_file(model, variable, begin, end):
    return f"{DATA_DIR}/{model}/{variable}/{variable}_{begin}-{end}.nc"


def output_file(model, variable):
    return f"{SAVE_DIR}/{model}/{variable}_{model}.pickle"


def _conc(x):
    """Swap the eastern and western halves of the longitude."""
    return np.concatenate([x[:, :, -180:], x[:, :, :180]], 2)


def load_file(path, variable, n_months, depth=0, masked_depth_mean=False, conc=False):
    """Read the first ``n_months`` of one file, flip the latitude and cut 60S-60N."""
    with Dataset(path, "r") as ds:
        var = ds.variables[variable]
        if variable == "tsl":
            data = var[:] if masked_depth_mean else np.array(var)
            data = np.mean(data[:n_months, :(depth + 1), ::-1, :], axis=1)
        else:
            data = var[:][:n_months, ::-1, :]
    data = ma.filled(data[:, ULIM:LLIM, :], fill_value=FILL_VALUE)
    if conc:
        data = _conc(data)
    return data


def make_val(settings):
    """Join all files of one model and variable into a masked array (time, lat, lon)."""
    model, variable = settings["model"], settings["variable"]
    files = settings["files"]
    total = sum(12 * (end - begin + 1) for begin, end in files)

    val = None
    pos = 0
    for begin, end in files:
        n_months = 12 * (end - begin + 1)
        data = load_file(input_file(model, variable, begin, end), variable, n_months,
                         depth=settings["depth"],
                         masked_depth_mean=settings["masked_depth_mean"],
                         conc=settings["conc"])
        if val is None:
            val = np.empty((total,) + data.shape[1:])
        val[pos:pos + n_months] = data
        pos += n_months
        print(f"{begin}-{end} loaded")

    if variable in CONVERT:
        val = CONVERT[variable](val)
    # mask the filled values
    return ma.masked_where(val > MASK_THRESHOLD, val)


def standardize(x):
    """Monthly climatology, standard deviation, anomaly and standardized anomaly."""
    clim = np.empty((12,) + x.shape[1:])
    variance = np.empty((12,) + x.shape[1:])
    anom = np.empty(x.shape)
    std = np.empty(x.shape)
    for mon in range(12):
        sub = x[mon::12]
        c = sub.mean(axis=0)
        s = sub.std(axis=0)
        clim[mon] = c
        variance[mon] = s
        anom[mon::12] = sub - c
        std[mon::12] = (sub - c) / s
    return clim, variance, anom, std


def coarsen(x):
    """Nearest neighbour regridding from 1x1 to 5x5 degrees."""
    return x[:, ::UPSCALE_RATE, ::UPSCALE_RATE].copy()


def preprocess(settings):
    """Run all the steps and return the arrays saved in the pickle file."""
    variable = settings["variable"]
    val = make_val(settings)
    clim, variance, anom, std = standardize(val)
    val_coarse = coarsen(val)
    coarse_clim, coarse_variance, coarse_anom, coarse_std = standardize(val_coarse)

    result = {
        f"{variable}_raw": val,
        f"{variable}_clim": clim,
        f"{variable}_variance": variance,
        f"{variable}_anom": anom,
        f"{variable}_std": std,
        f"{variable}_coarse": val_coarse,
        f"{variable}_coarse_clim": coarse_clim,
        f"{variable}_coarse_variance": coarse_variance,
        f"{variable}_coarse_anom": coarse_anom,
        f"{variable}_coarse_std": coarse_std,
    }

    drop = settings["drop_last_months"]
    if drop:
        for key in ["raw", "anom", "std", "coarse", "coarse_anom", "coarse_std"]:
            result[f"{variable}_{key}"] = result[f"{variable}_{key}"][:-drop]
    return result


def save(result, model, variable):
    path = output_file(model, variable)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"{path} has been SAVED")
