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

ngridx = 100
ngridy = 100



fig = plt.figure(figsize=(12, 12))

# Use the tile's projection for the underlying map.
ax = plt.axes(projection=osm_tiles.crs)

####################################

p = coords_test[5]
pmin_0 = np.min(p[:,1])
pmax_0 = np.max(p[:,1])
pmin_1 = np.min(p[:,0])
pmax_1 = np.max(p[:,0])

# Specify a region of interest
ax.set_extent([pmin_0-0.001, pmax_0+0.001, pmin_1-0.001, pmax_1+0.001], ccrs.PlateCarree())

# Add the tiles at zoom level 12.
ax.add_image(osm_tiles, 15, cmap='gray')

ax.plot([p[0][1],p[1][1],p[3][1],p[2][1],p[0][1]],[p[0][0],p[1][0],p[3][0],p[2][0],p[0][0]],'m',lw=4,transform=ccrs.PlateCarree())

###################################

# Specify a region of interest
ax.set_extent([-3.1115718-0.01, -2.7624278+0.01, 43.160213-0.01, 43.41766+0.01], ccrs.PlateCarree())

# Add the tiles at zoom level 12.
ax.add_image(osm_tiles, 14, cmap='gray')



patches_train = []

for p in coords_train:
        
    ax.plot([p[0][1],p[1][1],p[3][1],p[2][1],p[0][1]],[p[0][0],p[1][0],p[3][0],p[2][0],p[0][0]],'k',lw=3,transform=ccrs.PlateCarree())
    patches_train.append(Polygon(np.fliplr([p[0,:],p[1,:],p[3,:],p[2,:],p[0,:]]), closed=True))

patches_val = []

for p in coords_val:
        
    ax.plot([p[0][1],p[1][1],p[3][1],p[2][1],p[0][1]],[p[0][0],p[1][0],p[3][0],p[2][0],p[0][0]],'y',lw=3,transform=ccrs.PlateCarree())
    patches_val.append(Polygon(np.fliplr([p[0,:],p[1,:],p[3,:],p[2,:],p[0,:]]), closed=True))

patches_test = []

for p in coords_test:
        
    ax.plot([p[0][1],p[1][1],p[3][1],p[2][1],p[0][1]],[p[0][0],p[1][0],p[3][0],p[2][0],p[0][0]],'m',lw=3,transform=ccrs.PlateCarree())
    patches_test.append(Polygon(np.fliplr([p[0,:],p[1,:],p[3,:],p[2,:],p[0,:]]), closed=True))

allPatches_train = PatchCollection(patches_train, alpha=0.4, facecolor='k', transform=ccrs.PlateCarree())
ax.add_collection(allPatches_train)
allPatches_val = PatchCollection(patches_val, alpha=0.4, facecolor='y', transform=ccrs.PlateCarree())
ax.add_collection(allPatches_val)
allPatches_test = PatchCollection(patches_test, alpha=0.4, facecolor='m', transform=ccrs.PlateCarree())
ax.add_collection(allPatches_test)

SELECTEDHOUR = 14

VMIN = 10 # Valor minimo para la escala de colores
VMAX = 35 # Valor maximo para la escala de colores

patchesInTest = list(testData.keys()) 

for patch in patchesInTest:
    
    coordsPoints_X = testData[patch]['Xs'] # Coordenadas X de los puntos del patch
    coordsPoints_Y = testData[patch]['Ys'] # Coordenadas Y de los puntos del patch
    coordsBoundingBox = testData[patch]['boundingBox'] # Bounding box del patch

    dataToPlot_UNET = np.mean(testData[patch]['UNET'][SELECTEDHOUR],0).ravel()

    min_X = min(coordsPoints_X)
    max_X = max(coordsPoints_X)
    min_Y = min(coordsPoints_Y)
    max_Y = max(coordsPoints_Y)

    xi = np.linspace(min_X, max_X, ngridx)
    yi = np.linspace(min_Y, max_Y, ngridy)

    zi = griddata((coordsPoints_X,coordsPoints_Y), dataToPlot_UNET,(xi[None, :], yi[:, None]), method='linear')

    im = ax.contourf(xi,yi,zi,levels=14, vmin=VMIN,vmax=VMAX, alpha=0.5, cmap="RdBu_r",transform=ccrs.PlateCarree())

with open(FOLDER_DATOS+'dictVal_LWT.pkl','rb') as fid:
    valData = pickle.load(fid)

patchesInVal = list(valData.keys()) 

for patch in patchesInVal:
    
    coordsPoints_X = valData[patch]['Xs'] # Coordenadas X de los puntos del patch
    coordsPoints_Y = valData[patch]['Ys'] # Coordenadas Y de los puntos del patch
    coordsBoundingBox = valData[patch]['boundingBox'] # Bounding box del patch

    dataToPlot_UNET = np.mean(valData[patch]['UNET'][SELECTEDHOUR],0).ravel()

    min_X = min(coordsPoints_X)
    max_X = max(coordsPoints_X)
    min_Y = min(coordsPoints_Y)
    max_Y = max(coordsPoints_Y)

    xi = np.linspace(min_X, max_X, ngridx)
    yi = np.linspace(min_Y, max_Y, ngridy)

    zi = griddata((coordsPoints_X,coordsPoints_Y), dataToPlot_UNET,(xi[None, :], yi[:, None]), method='linear')

    im = ax.contourf(xi,yi,zi,levels=14, vmin=VMIN,vmax=VMAX, alpha=0.5, cmap="RdBu_r",transform=ccrs.PlateCarree())

with open(FOLDER_DATOS+'dictTrain_LWT.pkl','rb') as fid:
    trainData = pickle.load(fid)

patchesInTrain = list(trainData.keys()) 

for patch in patchesInTrain:
    
    coordsPoints_X = trainData[patch]['Xs'] # Coordenadas X de los puntos del patch
    coordsPoints_Y = trainData[patch]['Ys'] # Coordenadas Y de los puntos del patch
    coordsBoundingBox = trainData[patch]['boundingBox'] # Bounding box del patch

    dataToPlot_UNET = np.mean(trainData[patch]['UNET'][SELECTEDHOUR],0).ravel()

    min_X = min(coordsPoints_X)
    max_X = max(coordsPoints_X)
    min_Y = min(coordsPoints_Y)
    max_Y = max(coordsPoints_Y)

    xi = np.linspace(min_X, max_X, ngridx)
    yi = np.linspace(min_Y, max_Y, ngridy)

    zi = griddata((coordsPoints_X,coordsPoints_Y), dataToPlot_UNET,(xi[None, :], yi[:, None]), method='linear')

    im = ax.contourf(xi,yi,zi,levels=14, vmin=VMIN,vmax=VMAX, alpha=0.5, cmap="RdBu_r",transform=ccrs.PlateCarree())

for estacion in estaciones:
    ax.scatter(estaciones[estacion]['coords'][1],estaciones[estacion]['coords'][0],s=100,c='indianred',edgecolors='k',transform=ccrs.PlateCarree())

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0, color='gray', alpha=0.5, linestyle='-')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False
gl.ylines = False

fig.set_size_inches(12.43, 12.84)

fig.subplots_adjust(
top=0.988,
bottom=0.012,
left=0.064,
right=0.936,
hspace=0.2,
wspace=0.2)

plt.tight_layout()

plt.savefig(FOLDER_PLOTS+'HOUR_'+str(SELECTEDHOUR)+'_UNET_spatial_distribution_all_patches.pdf')
plt.close()