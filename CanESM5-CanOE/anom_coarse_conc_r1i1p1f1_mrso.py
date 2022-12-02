import pickle
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
from mpl_toolkits import basemap
import colorcet as cc
import cmaps

from anom_coarse_conc_r1i1p1f1_tos import Preprocess

def main():
    save_flag = True
    conc_flag = False
    first_flag = True
    loop_flag = False
    last_flag = False
    reverse_flag = False
    model = 'CanESM5-CanOE'
    variable = 'mrso'
    timestep_loop = 99999
    first_begin, first_end = 1850, 2014
    # if no loop or last is required, edit like start_loop=stop_loop=99999 and timestep_loop > 0
    last_begin, last_end = 1, 0
    start_loop, stop_loop = 99999, 99999

    # load preprocess class
    pre = Preprocess(model, variable, timestep_loop,
                     first_begin, first_end, last_begin, last_end, start_loop, stop_loop)

    # val_raw
    val = pre.make_val(conc_flag=conc_flag, first_flag=first_flag, loop_flag=loop_flag, last_flag=last_flag)
    pre._imshow(val[8,:,:], reverse_flag=reverse_flag)

    # val_anom
    val_anom = pre.anomaly(val)
    #pre._plot_anom(val_anom[8,:,:], reverse_flag=reverse_flag)

    # val_std
    val_clim, val_variance, val_std = pre.standardize(val)
    #pre._imshow(val_clim[8,:,:], reverse_flag=reverse_flag)
    #pre._imshow(val_variance[8,:,:], reverse_flag=reverse_flag)
    pre._plot_std(val_std[8,:,:], reverse_flag=reverse_flag)

    # interpolation
    val_coarse = pre.interpolation(val)
    #pre._imshow(val_coarse[8,:,:], reverse_flag=reverse_flag)

    # val_coarse_anom
    val_coarse_anom = pre.anomaly(val_coarse)
    #pre._plot_anom(val_coarse_anom[8,:,:], reverse_flag=reverse_flag)

    # val_coarse_std
    val_coarse_clim, val_coarse_variance, val_coarse_std = pre.standardize(val_coarse)
    #pre._imshow(val_coarse_clim[8,:,:], reverse_flag=reverse_flag)
    #pre._imshow(val_coarse_variance[8,:,:], reverse_flag=reverse_flag)
    #pre._plot_std(val_coarse_std[8,:,:], reverse_flag=reverse_flag)

    # save as pickle file
    pre.save(val, val_clim, val_variance, val_anom, val_std,
             val_coarse, val_coarse_clim, val_coarse_variance, val_coarse_anom, val_coarse_std, save_flag=save_flag)
    plt.show()


if __name__ == '__main__':
    main()

