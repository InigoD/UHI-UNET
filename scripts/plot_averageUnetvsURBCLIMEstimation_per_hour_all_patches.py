import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
import matplotlib.pyplot as plt
import utm
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pickle
from scipy.interpolate import griddata # type: ignore

IMG_SIZE_X = 32
IMG_SIZE_Y = 32
FOLDER_RESULTS = '/home/jdelser/Work/UNET-NEW/MODELS/LWT_PER_HOUR/'
FOLDER_DATOS = '/home/jdelser/Work/UNET-NEW/DATA/'
FOLDER_PLOTS = '/home/jdelser/Work/UNET-NEW/PLOTS/'
LOOKBACK=2

with open(FOLDER_DATOS+'dictTest_LWT.pkl','rb') as fid:
    testData = pickle.load(fid)

coords = np.load(FOLDER_DATOS+'coords_joined.npy')

estaciones = {'arboleda':{'coords':[43.2967, -3.06747],'history':None}, 
              'arrigorriaga':{'coords':[43.2165, -2.89976],'history':None}, 
              'derio':{'coords':[43.2911, -2.87342],'history':None}, 
              'deusto':{'coords':[43.2834, -2.96791],'history':None}, 
              'galindo':{'coords':[43.3062, -2.99878],'history':None}, 
              'puntagalea':{'coords':[43.3752, -3.03608],'history':None}, 
              'zorroza':{'coords':[43.284980, -2.968458],'history':None}}

VENTANASVAL = [4,25,38,46,61]
VENTANASTEST = [27,12,32,34,57,29]
VENTANASTRAIN = list(set(range(64))-set(VENTANASTEST)-set(VENTANASVAL))

coords_train = coords[VENTANASTRAIN]
coords_val = coords[VENTANASVAL]
coords_test = coords[VENTANASTEST]

osm_tiles = OSM()

from cartopy.io.img_tiles import QuadtreeTiles
osm_tiles = QuadtreeTiles()

for SELECTEDHOUR in range(0,24):

    fig, axs = plt.subplots(6,3, subplot_kw={'projection':osm_tiles.crs})

    #SELECTEDHOUR = 5

    VMIN = 10 # Valor minimo para la escala de colores
    VMAX = 35

    irow = 0
    icol = 0
    ngridx = 100
    ngridy = 100

    labels = ['a','b','c','d','e','f']

    for i in range(len(VENTANASTEST)):
        p = coords_test[i]
        pmin_0 = np.min(p[:,1])
        pmax_0 = np.max(p[:,1])
        pmin_1 = np.min(p[:,0])
        pmax_1 = np.max(p[:,0])

        axs[irow,icol].set_extent([pmin_0-0.001, pmax_0+0.001, pmin_1-0.001, pmax_1+0.001], ccrs.PlateCarree())
        axs[irow+1,icol].set_extent([pmin_0-0.001, pmax_0+0.001, pmin_1-0.001, pmax_1+0.001], ccrs.PlateCarree())
        axs[irow+2,icol].set_extent([pmin_0-0.001, pmax_0+0.001, pmin_1-0.001, pmax_1+0.001], ccrs.PlateCarree())

        axs[irow,icol].add_image(osm_tiles, 14, cmap='gray')
        axs[irow+1,icol].add_image(osm_tiles, 14, cmap='gray')
        axs[irow+2,icol].add_image(osm_tiles, 14, cmap='gray')

        axs[irow,icol].plot([p[0][1],p[1][1],p[3][1],p[2][1],p[0][1]],[p[0][0],p[1][0],p[3][0],p[2][0],p[0][0]],'m',lw=2,transform=ccrs.PlateCarree())

        patch = VENTANASTEST[i]
        dataToPlot_UNET = np.mean(testData[patch]['UNET'][SELECTEDHOUR],0).ravel()
        dataToPlot_UCLIM = np.mean(testData[patch]['URBCLIM'][SELECTEDHOUR],0).ravel()

        coordsPoints_X = testData[patch]['Xs'] # Coordenadas X de los puntos del patch
        coordsPoints_Y = testData[patch]['Ys'] # Coordenadas Y de los puntos del patch
        coordsBoundingBox = testData[patch]['boundingBox'] # Bounding box del patch

        min_X = min(coordsPoints_X)
        max_X = max(coordsPoints_X)
        min_Y = min(coordsPoints_Y)
        max_Y = max(coordsPoints_Y)

        xi = np.linspace(min_X, max_X, ngridx)
        yi = np.linspace(min_Y, max_Y, ngridy)

        zi_unet = griddata((coordsPoints_X,coordsPoints_Y), dataToPlot_UNET,(xi[None, :], yi[:, None]), method='linear')
        zi_uclim = griddata((coordsPoints_X,coordsPoints_Y), dataToPlot_UCLIM,(xi[None, :], yi[:, None]), method='linear')

        im = axs[irow+1,icol].contourf(xi,yi,zi_uclim,levels=20, vmin=VMIN,vmax=VMAX, alpha=0.6, cmap="RdBu_r",transform=ccrs.PlateCarree())
        im = axs[irow+2,icol].contourf(xi,yi,zi_unet,levels=20, vmin=VMIN,vmax=VMAX, alpha=0.6, cmap="RdBu_r",transform=ccrs.PlateCarree())

        for estacion in estaciones:

            axs[irow,icol].scatter(estaciones[estacion]['coords'][1],estaciones[estacion]['coords'][0],s=100,c='indianred',edgecolors='k',transform=ccrs.PlateCarree())

        axs[irow+2,icol].get_xaxis().set_visible(True)
        axs[irow+2,icol].xaxis.set_ticklabels([])
        axs[irow+2,icol].xaxis.set_ticks([])
        axs[irow+2,icol].set_xlabel('('+labels[i]+')',fontsize=16)


        if icol==2:
            icol = 0
            irow = irow + 3
        else:
            icol = icol + 1

    fig.set_size_inches(5.62, 12.84)

    fig.subplots_adjust(top=0.988,
    bottom=0.034,
    left=0.027,
    right=0.973,
    hspace=0.259,
    wspace=0.0)

    plt.savefig(FOLDER_PLOTS+'HOUR_'+str(SELECTEDHOUR)+'_UNETvsURBCLIM_aggregated_LWT_all_patches.pdf')
    plt.close()