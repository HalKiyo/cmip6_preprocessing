import pickle
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits import basemap
import colorcet as cc
import cmaps

def main():
    save_flag = True
    conc_flag = False
    first_flag = True
    loop_flag = False
    last_flag = False
    reverse_flag = False
    model = 'BCC-ESM1'
    variable = 'snc'
    timestep_loop = 99999
    first_begin, first_end = 1850, 2014
    # if last_flag is False, edit (last_begin - last_end = 1)
    last_begin, last_end = 1, 0
    # if loop_flag is False, edit like start_loop=stop_loop=99999 and timestep_loop > 0
    timestep_loop = 99999
    start_loop, stop_loop = 99999, 99999

    # load preprocess class
    pre = Preprocess(model, variable, timestep_loop,
                     first_begin, first_end, last_begin, last_end, start_loop, stop_loop)

    # val_raw
    val = pre.make_val(conc_flag=conc_flag, first_flag=first_flag, loop_flag=loop_flag, last_flag=last_flag)
    pre._imshow(val[0,:,:], reverse_flag=reverse_flag)

    # val_anom
    val_anom = pre.anomaly(val)
    #pre._plot_anom(val_anom[0,:,:], reverse_flag=reverse_flag)

    # val_std
    val_clim, val_variance, val_std = pre.standardize(val)
    #pre._imshow(val_clim[0,:,:], reverse_flag=reverse_flag)
    #pre._imshow(val_variance[0,:,:], reverse_flag=reverse_flag)
    pre._plot_std(val_std[0,:,:], reverse_flag=reverse_flag)

    # interpolation
    val_coarse = pre.interpolation(val)
    #pre._imshow(val_coarse[0,:,:], reverse_flag=reverse_flag)

    # val_coarse_anom
    val_coarse_anom = pre.anomaly(val_coarse)
    #pre._plot_anom(val_coarse_anom[0,:,:], reverse_flag=reverse_flag)

    # val_coarse_std
    val_coarse_clim, val_coarse_variance, val_coarse_std = pre.standardize(val_coarse)
    #pre._imshow(val_coarse_clim[0,:,:], reverse_flag=reverse_flag)
    #pre._imshow(val_coarse_variance[0,:,:], reverse_flag=reverse_flag)
    #pre._plot_std(val_coarse_std[0,:,:], reverse_flag=reverse_flag)

    # save as pickle file
    pre._imshow(val[0,:,:], reverse_flag=reverse_flag)
    pre.save(val, val_clim, val_variance, val_anom, val_std,
             val_coarse, val_coarse_clim, val_coarse_variance, val_coarse_anom, val_coarse_std, save_flag=save_flag)
    plt.show()

class Preprocess():
    def __init__(self, model, variable, timestep_loop, 
                 first_begin, first_end, last_begin, last_end, start_loop, stop_loop):
        self.model = model
        self.variable = variable
        self.timestep_loop = timestep_loop
        self.first_begin, self.first_end = first_begin, first_end
        self.last_begin, self.last_end = last_begin, last_end
        self.start_loop, self.stop_loop = start_loop, stop_loop

        self.ulim, self.llim = 30, -30 # upper and lower limits of latitude
        self.lt, self.ln = 120, 360 # grid number of latitude and longitude 
        self.upscale_rate = 5 # from 1x1 to 5x5
        self.data_num= 12*self.timestep_loop # number of data in one file
        self.first_data_num = 12*(self.first_end - self.first_begin + 1)
        self.last_data_num = 12*(self.last_end - self.last_begin + 1)
        self.loopyr = (self.stop_loop - self.start_loop)/self.timestep_loop # become 0 when loop_flag is False
        self.tm = int(self.data_num*self.loopyr + self.first_data_num + self.last_data_num) # number of all data_num

        self.datadir = f"/work/kajiyama/cdo/cmip6/{self.model}/{self.variable}"
        self.first_file = f"{self.datadir}/{self.variable}_{self.first_begin}-{self.first_end}.nc"
        self.last_file = f"{self.datadir}/{self.variable}_{self.last_begin}-{self.last_end}.nc"
        self.save_file = f"/work/kajiyama/preprocessed/cmip6/{self.model}" \
                         f"/{self.variable}_{self.model}.pickle"

        with Dataset(self.first_file, 'r') as nc:
            self.lons = nc.variables['lon'][:]
            self.lats = nc.variables['lat'][self.ulim:self.llim]
        self.lons_sub, self.lats_sub = np.meshgrid(self.lons[::self.upscale_rate], self.lats[::self.upscale_rate])

    def _fill(self, x):
        f = ma.filled(x,fill_value=99999)
        return f

    def _mask(self, x):
        m = ma.masked_where(x>9999, x)
        return m

    def _conc(self, x):
        c = x.copy()
        c = self._fill(c)
        x1, x2 = c[:,:,-180:], c[:,:,:180]
        c = np.concatenate([x1,x2],2)
        c = self._mask(c)

        return c

    def _load_val(self, file, data_num, conc_flag=True):
        ds = Dataset(file, 'r')
        val = ds.variables[self.variable][:]
        val = val[:data_num, ::-1, :]
        if conc_flag is True:
            val = self._conc(val[:,self.ulim:self.llim,:])
        else:
            val = val[:, self.ulim:self.llim, :]
        return val

    def _imshow(self, image, reverse_flag=False):
        if reverse_flag is True:
            origin = 'lower'
        else:
            origin = 'upper'

        if self.variable == 'snc':
            plt.register_cmap('cc_blues_r', cc.cm.blues_r)
            cmap = plt.get_cmap('cc_blues_r', 256)
            cmap_data = cmap(np.arange(cmap.N))
            cmap_data[0, 3] = 0 #make 0 value's alph zero.
            my_cmap = colors.ListedColormap(cmap_data)

        projection = ccrs.PlateCarree(central_longitude=180)
        img_extent = (-180, 180, -60, 60)
        fig = plt.figure()
        ax = plt.subplot(projection=projection)
        ax.add_feature(cfeature.LAND,
                       edgecolor='face',
                       facecolor='dimgrey'
                       )
        ax.add_feature(cfeature.OCEAN,
                       edgecolor='face',
                       facecolor='darkgrey'
                       )
        mat = ax.matshow(image,
                         origin = origin,
                         extent=img_extent,
                         transform=projection,
                         cmap= my_cmap
                         )
        cbar = fig.colorbar(mat,
                            ax=ax,
                            orientation='horizontal',
                            extend='both'
                            )
        plt.show(block=False)

    def _plot_anom(self, image, reverse_flag=False):
        if reverse_flag is True:
            origin = 'lower'
        else:
            origin = 'upper'

        if self.variable == 'snc':
            plt.register_cmap('cc_bwy_r', cc.cm.bwy_r)
            cmap = plt.cm.get_cmap('cc_bwy_r', 256)

        projection = ccrs.PlateCarree(central_longitude=180)
        img_extent = (-180, 180, -60, 60)
        fig = plt.figure()
        ax = plt.subplot(projection=projection)
        ax.coastlines()
        mat = ax.matshow(image,
                         origin = origin,
                         extent=img_extent,
                         transform=projection,
                         norm=colors.CenteredNorm(),
                         cmap= cmap
                         )
        cbar = fig.colorbar(mat,
                            ax=ax,
                            orientation='horizontal'
                            )
        plt.show(block=False)

    def _plot_std(self, image, reverse_flag=False):
        if reverse_flag is True:
            origin = 'lower'
        else:
            origin = 'upper'

        if self.variable == 'snc':
            plt.register_cmap('cc_bwy_r', cc.cm.bwy_r)
            cmap = plt.cm.get_cmap('cc_bwy_r', 256)

        projection = ccrs.PlateCarree(central_longitude=180)
        img_extent = (-180, 180, -60, 60)
        fig = plt.figure()
        ax = plt.subplot(projection=projection)
        ax.coastlines()
        mat = ax.matshow(image,
                         origin = origin,
                         extent=img_extent,
                         transform=projection,
                         norm=colors.Normalize(vmin=-3, vmax=3),
                         cmap= cmap
                         )
        cbar = fig.colorbar(mat,
                            ax=ax,
                            orientation='horizontal'
                            )
        plt.show(block=False)

    def make_val(self, conc_flag, first_flag=True, loop_flag=True, last_flag=True):
        # making empty box for save file
        val = np.empty((self.tm, self.lt, self.ln))

        # first netCDF4 files
        if first_flag is True:
            val[:self.first_data_num, :, :] = self._fill(
                    self._load_val(self.first_file, self.first_data_num, conc_flag))
            print(f"{self.first_begin}-{self.first_end} loaded")

        # middle netcdf4 files
        if loop_flag is True:
            # time augument count
            ind = self.data_num 
            for i in range(self.start_loop, self.stop_loop+1, self.timestep_loop):
                file = f"{self.datadir}/{self.variable}_{i}-{i+self.timestep_loop-1}.nc"
                val[(self.first_data_num + ind - self.data_num):(self.first_data_num + ind),:,:] = self._fill(
                        self._load_val(file, self.data_num, conc_flag))
                ind += self.data_num
                print(f"{i}-{i+self.timestep_loop-1} loaded")

        # last netcdf4 files
        if last_flag is True:
            val[-self.last_data_num:, :, :] = self._fill(
                    self._load_val(self.last_file, self.last_data_num, conc_flag))
            val = self._mask(val)
            print(f"{self.last_begin}-{self.last_end} loaded")

        #mask filled 99999 value before saving
        val = self._mask(val)
        print("maked out")

        return val

    def anomaly(self, x):
        dup = x.copy()
        val_anom = np.empty(dup.shape)
        for mon in range(12):
            val_clim = dup[mon::12, :, :].mean(axis=0)
            val_anom[mon::12, :, :] = dup[mon::12, :, :] - val_clim
            print(mon)
        return val_anom

    def standardize(self, x):
        dup = x.copy()
        val_std = np.empty(dup.shape)
        val_clim = np.empty((12, dup.shape[1], dup.shape[2]))
        val_variance = np.empty((12, dup.shape[1], dup.shape[2]))

        for mon in range(12):
            clim = dup[mon::12, :, :].mean(axis=0)
            variance = dup[mon::12, :, :].std(axis=0)
            val_clim[mon, :, :] = clim
            val_variance[mon, :, :] = variance
            val_std[mon::12, :, :] = (dup[mon::12, :, :] - clim) / variance
            print(mon)
        return val_clim, val_variance, val_std

    def interpolation(self, x):
        dup = x.copy()
        val_coarse = dup[:, :int(self.lt/self.upscale_rate), :int(self.ln/self.upscale_rate)]
        for time in range(len(val_coarse)):
            val_coarse[time, :, :] = basemap.interp(dup[time, :, :],
                                                    self.lons, self.lats, 
                                                    self.lons_sub, self.lats_sub,
                                                    order=0)
            print(time)
        return val_coarse

    def save(self, val_raw, val_clim, val_variance, val_anom, val_std, 
             val_coarse, val_coarse_clim, val_coarse_variance, val_coarse_anom, val_coarse_std, save_flag=False):
        save_dict = {f"{self.variable}_raw": val_raw,
                     f"{self.variable}_clim": val_clim,
                     f"{self.variable}_variance": val_variance,
                     f"{self.variable}_anom": val_anom,
                     f"{self.variable}_std": val_std,
                     f"{self.variable}_coarse": val_coarse,
                     f"{self.variable}_coarse_clim": val_coarse_clim,
                     f"{self.variable}_coarse_variance": val_coarse_variance,
                     f"{self.variable}_coarse_anom": val_coarse_anom,
                     f"{self.variable}_coarse_std": val_coarse_std
                     }

        print(f"val_raw: {val_raw.shape}",
              f"val_clim: {val_clim.shape}",
              f"val_variance: {val_variance.shape}",
              f"val_anom: {val_anom.shape}",
              f"val_std: {val_std.shape}",
              f"val_coarse: {val_coarse.shape}",
              f"val_coarse_clim: {val_coarse_clim.shape}",
              f"val_coarse_variance: {val_coarse_variance.shape}",
              f"val_coarse_anom: {val_coarse_anom.shape}",
              f"val_coarse_std: {val_coarse_std.shape}"
              )

        if save_flag is True:
            with open(self.save_file, 'wb') as f:
                pickle.dump(save_dict, f)
            print(f'{self.model} pickle file has been SAVED')
        else:
            print(f'{self.model} pickle file has ***NOT*** been saved yet')


if __name__ == '__main__':
    main()

