import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import colorcet as cc
import cmaps


def _resampled(cmap, n):
    cmap = plt.get_cmap(cmap)
    if hasattr(cmap, "resampled"):
        return cmap.resampled(n)
    return cmap._resample(n)  # matplotlib < 3.6


def _snc_cmap():
    cmap_data = _resampled(cc.cm.blues_r, 256)(np.arange(256))
    cmap_data[0, 3] = 0  # make the alpha of the zero value transparent
    return colors.ListedColormap(cmap_data)


# colormap of the raw values and of the anomalies for each variable
RAW_CMAP = {
    "pr": lambda: _resampled(cmaps.WhiteBlueGreenYellowRed, 10),
    "tsl": lambda: _resampled(cc.cm.rainbow, 10),
    "tos": lambda: _resampled(cc.cm.rainbow, 10),
    "mrso": lambda: _resampled(cmaps.GMT_ocean_r, 10),
    "snc": _snc_cmap,
}
ANOM_CMAP = {
    "pr": lambda: _resampled("BrBG", 10),
    "tsl": lambda: _resampled("RdBu_r", 10),
    "tos": lambda: _resampled("RdBu_r", 10),
    "mrso": lambda: _resampled("BrBG", 10),
    "snc": lambda: _resampled(cc.cm.bwy_r, 256),
}

# time index shown in the check plots
PLOT_INDEX = {"snc": 0}
DEFAULT_PLOT_INDEX = 8


def plot_map(image, variable, kind="raw", reverse=False, title=None):
    """Draw one map. ``kind`` is 'raw', 'anom' or 'std'."""
    projection = ccrs.PlateCarree(central_longitude=180)
    fig = plt.figure()
    ax = plt.subplot(projection=projection)

    colorbar_kw = {}
    if kind == "raw":
        cmap, norm = RAW_CMAP[variable](), None
        if variable == "snc":
            ax.add_feature(cfeature.LAND, edgecolor="face", facecolor="dimgrey")
            ax.add_feature(cfeature.OCEAN, edgecolor="face", facecolor="darkgrey")
            colorbar_kw["extend"] = "both"
        else:
            ax.coastlines()
    else:
        cmap = ANOM_CMAP[variable]()
        norm = colors.CenteredNorm() if kind == "anom" else colors.Normalize(vmin=-3, vmax=3)
        ax.coastlines()

    mat = ax.matshow(image,
                     origin="lower" if reverse else "upper",
                     extent=(-180, 180, -60, 60),
                     transform=projection,
                     norm=norm,
                     cmap=cmap)
    fig.colorbar(mat, ax=ax, orientation="horizontal", **colorbar_kw)
    if title:
        ax.set_title(title)
    plt.show(block=False)


def plot_check(result, model, variable):
    """Plot the raw value and the standardized anomaly like the old scripts did."""
    i = PLOT_INDEX.get(variable, DEFAULT_PLOT_INDEX)
    plot_map(result[f"{variable}_raw"][i], variable, "raw", title=f"{model} {variable} raw [{i}]")
    plot_map(result[f"{variable}_std"][i], variable, "std", title=f"{model} {variable} std [{i}]")
