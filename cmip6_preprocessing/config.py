"""Settings for every model and variable.

Each model lists the netCDF files it has, as (begin_year, end_year) pairs in
chronological order. A file is read from
``{DATA_DIR}/{model}/{variable}/{variable}_{begin}-{end}.nc``.
Use ``variables`` to override the model defaults for a single variable.
"""

DATA_DIR = "/work/kajiyama/cdo/cmip6"
SAVE_DIR = "/work/kajiyama/preprocessed/cmip6"

VARIABLES = ["pr", "tsl", "mrso", "snc", "tos"]


def chunks(begin, end, step):
    """Year ranges of files that each hold ``step`` years, e.g. 1860-1869, 1870-1879, ..."""
    return [(y, y + step - 1) for y in range(begin, end + 1, step)]


FULL = [(1850, 2014)]
SPLIT_1950 = [(1850, 1949), (1950, 2014)]
YEARLY = [(1850, 1850), *chunks(1851, 2013, 1), (2014, 2014)]
DECADAL = [(1850, 1859), *chunks(1860, 2009, 10), (2010, 2014)]
HALF_CENTURY = [(1850, 1899), *chunks(1900, 1999, 50), (2000, 2014)]
TWENTY = [(1850, 1869), *chunks(1870, 2009, 20), (2010, 2014)]
GISS_E2_1 = [(1850, 1900), *chunks(1901, 2000, 50), (2001, 2014)]

# Per model settings
#   files               : list of (begin, end) year ranges of the input files
#   depth               : tsl uses the mean of soil layers 0..depth
#   conc                : roll the longitude by 180 degrees
#   masked_depth_mean   : tsl ignores masked layers when averaging over depth
#   drop_last_months    : drop months at the end of the time series before saving
#   variables           : per variable overrides of the keys above
MODELS = {
    "AWI-ESM-1-1-LR": {
        "files": YEARLY, "depth": 1, "masked_depth_mean": True,
        "variables": {
            "tos": {"files": [(1850, 1850), *chunks(1851, 2010, 10), (2011, 2014)], "conc": True},
        },
    },
    "BCC-CSM2-MR": {
        "files": FULL, "depth": 3, "masked_depth_mean": True,
        "variables": {"tsl": {"files": [(1850, 1929), (1930, 2014)]}},
    },
    "BCC-ESM1": {"files": FULL, "depth": 3, "masked_depth_mean": True},
    "CAS-ESM2-0": {
        "files": FULL, "depth": 3, "masked_depth_mean": True,
        "variables": {"tsl": {"files": [(1850, 1899), (1900, 1949), (1950, 2014)]}},
    },
    "CESM2":{"files": FULL, "depth": 3},
    "CESM2-FV2": {"files": HALF_CENTURY, "depth": 3},
    "CESM2-WACCM": {"files": FULL, "depth": 3},
    "CMCC-CM2-HR4": {
        "files": FULL, "depth": 3,
        "variables": {"tos": {"files": [(1850, 1874), *chunks(1875, 1999, 25), (2000, 2014)]}},
    },
    "CMCC-CM2-SR5": {"files": FULL, "depth": 3},
    "CMCC-ESM2": {"files": FULL, "depth": 3},
    "CNRM-CM6-1": {"files": FULL, "depth": 3},
    "CNRM-CM6-1-HR": {"files": FULL, "depth": 3},
    "CNRM-ESM2-1": {"files": FULL, "depth": 3},
    "CanESM5": {"files": FULL, "depth": 0, "masked_depth_mean": True},
    "CanESM5-CanOE": {"files": FULL, "depth": 0, "masked_depth_mean": True},
    "EC-Earth3": {"files": YEARLY, "depth": 1},
    "EC-Earth3-Veg": {"files": [(1950, 1950), *chunks(1951, 2013, 1), (2014, 2014)], "depth": 1},
    "EC-Earth3-Veg-LR": {"files": YEARLY, "depth": 1},
    "FGOALS-f3-L": {"files": FULL, "depth": 3},
    "FGOALS-g3": {
        # the files run until 2016, the last two years are dropped before saving
        "files": [(1850, 2016)], "depth": 3, "drop_last_months": 24,
        "variables": {
            "pr": {"files": [(1850, 1859), *chunks(1860, 2009, 10), (2010, 2016)]},
            "tos": {"files": FULL, "drop_last_months": 0},
        },
    },
    "GISS-E2-1-G": {"files": GISS_E2_1, "depth": 1},
    "GISS-E2-1-G-CC": {"files": GISS_E2_1, "depth": 1},
    "GISS-E2-1-H": {"files": GISS_E2_1, "depth": 1},
    "GISS-E2-2-G": {"files": [(1850, 1875), *chunks(1876, 1975, 25), (1976, 1990), (1991, 2014)], "depth": 1},
    "GISS-E2-2-H": {"files": [(1850, 1875), *chunks(1876, 2000, 25), (2001, 2014)], "depth": 1},
    "GISS-E3-G": {"files": HALF_CENTURY, "depth": 1},
    "HadGEM3-GC31-LL": {"files": SPLIT_1950, "depth": 1},
    "HadGEM3-GC31-MM": {
        "files": [(1850, 1869), *chunks(1870, 1969, 20), (1970, 1987), (1988, 1989), (1990, 2009), (2010, 2014)],
        "depth": 1,
    },
    "ICON-ESM-LR": {"files": DECADAL, "depth": 1, "conc": True},
    "IPSL-CM5A2-INCA": {"files": FULL, "depth": 2},
    "IPSL-CM6A-LR": {"files": FULL, "depth": 6},
    "IPSL-CM6A-LR-INCA": {"files": FULL, "depth": 6},
    "MIROC-ES2H": {"files": FULL, "depth": 1},
    "MIROC-ES2L": {"files": FULL, "depth": 1},
    "MIROC6": {"files": SPLIT_1950, "depth": 1},
    "MPI-ESM-1-2-HAM": {"files": TWENTY, "depth": 1},
    "MPI-ESM1-2-HR": {"files": [(1850, 1854), *chunks(1855, 2009, 5), (2010, 2014)], "depth": 1},
    "MPI-ESM1-2-LR": {"files": TWENTY, "depth": 1},
    "MRI-ESM2-0": {
        "files": FULL, "depth": 3,
        "variables": {"tsl": {"files": HALF_CENTURY}},
    },
    "NorESM2-MM": {"files": DECADAL, "depth": 3},
    "TaiESM1": {
        "files": FULL, "depth": 3,
        "variables": {"tos": {"files": YEARLY}},
    },
    "UKESM1-0-LL": {"files": SPLIT_1950, "depth": 1},
    "UKESM1-1-LL": {"files": SPLIT_1950, "depth": 1},
}

DEFAULTS = {"conc": False, "depth": 0, "masked_depth_mean": False, "drop_last_months": 0}


def get_settings(model, variable):
    """Return the settings of one model and variable with the overrides applied."""
    if model not in MODELS:
        raise KeyError(f"unknown model: {model}")
    if variable not in VARIABLES:
        raise KeyError(f"unknown variable: {variable}")
    base = {k: v for k, v in MODELS[model].items() if k != "variables"}
    settings = {**DEFAULTS, **base, **MODELS[model].get("variables", {}).get(variable, {})}
    settings["model"] = model
    settings["variable"] = variable
    return settings
