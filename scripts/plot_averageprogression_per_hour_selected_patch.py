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

# 34 GALINDO
# 12 ARRIGO

SELECTEDWINDOW = 34 # 27,12,32,34,57,29
ACRONYM = 'GALINDO'
patch = SELECTEDWINDOW

fig, axs = plt.subplots(5,5, subplot_kw={'projection':osm_tiles.crs})

patch = SELECTEDWINDOW
p = coords_test[VENTANASTEST.index(patch)]
pmin_0 = np.min(p[:,1])
pmax_0 = np.max(p[:,1])
pmin_1 = np.min(p[:,0])
pmax_1 = np.max(p[:,0])

VMIN = 10 # Valor minimo para la escala de colores
VMAX = 35

irow = 0
icol = 0

axs[irow,icol].set_extent([pmin_0-0.001, pmax_0+0.001, pmin_1-0.001, pmax_1+0.001], ccrs.PlateCarree())
axs[irow,icol].add_image(osm_tiles, 14, cmap='gray')
axs[irow,icol].plot([p[0][1],p[1][1],p[3][1],p[2][1],p[0][1]],[p[0][0],p[1][0],p[3][0],p[2][0],p[0][0]],'m',lw=2,transform=ccrs.PlateCarree())

for estacion in estaciones:
    axs[irow,icol].scatter(estaciones[estacion]['coords'][1],estaciones[estacion]['coords'][0],s=100,c='indianred',edgecolors='k',transform=ccrs.PlateCarree())

icol = icol+1

ngridx = 100
ngridy = 100

for SELECTEDHOUR in range(24):

    axs[irow,icol].set_extent([pmin_0-0.001, pmax_0+0.001, pmin_1-0.001, pmax_1+0.001], ccrs.PlateCarree())
    axs[irow,icol].add_image(osm_tiles, 14, cmap='gray')
    
    dataToPlot_UNET = np.mean(testData[patch]['UNET'][SELECTEDHOUR],0).ravel()
    
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

    im = axs[irow,icol].contourf(xi,yi,zi_unet,levels=20, vmin=VMIN,vmax=VMAX, alpha=0.6, cmap="RdBu_r",transform=ccrs.PlateCarree())
    
    axs[irow,icol].get_xaxis().set_visible(True)
    axs[irow,icol].xaxis.set_ticklabels([])
    axs[irow,icol].xaxis.set_ticks([])

    axs[irow,icol].set_xlabel(str(SELECTEDHOUR)+':00',fontsize=16)

    if icol==4:
        icol = 0
        irow = irow + 1
    else:
        icol = icol + 1


fig.set_size_inches(11.71, 12.06)

fig.subplots_adjust(top=0.988,
bottom=0.036,
left=0.012,
right=0.988,
hspace=0.224,
wspace=0.0)

plt.savefig(FOLDER_PLOTS+str(patch)+'_patch_'+ACRONYM+'_day_progression_mean_across_LWT_days.pdf')
plt.close()