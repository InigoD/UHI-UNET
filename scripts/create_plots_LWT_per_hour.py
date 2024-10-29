import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms
import pickle
from tqdm import tqdm
import osmnx as ox
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd

# =============================================================================
# DATOS
# =============================================================================

IMG_SIZE_X = 32
IMG_SIZE_Y = 32
FOLDER_PLOTS = '/home/jdelser/Work/UNET-NEW/PLOTS/'
FOLDER_DATOS = '/home/jdelser/Work/UNET-NEW/DATA/'
FOLDER_RESULTS = '/home/jdelser/Work/UNET-NEW/MODELS/LWT_PER_HOUR/'
LOOKBACK=2

COMPUTEGRIDS = False
PLOT_TABLE = True
PLOT_TIME_SERIES = True

def read_csv(filename,indexColData=0,indexColFecha=1,delimiter=','):
    indexRow = 0
    fid = open(filename,'r')
    data = []
    for line in fid:
        if indexRow>=1:
            dato = line.strip().split(delimiter)[indexColData]
            fecha = line.strip().split(delimiter)[indexColFecha]
            if dato=='':
                data.append([-1,fecha])
            else:
                data.append([float(dato),fecha])
        indexRow = indexRow + 1
    
    return data

# =============================================================================
# LEEMOS ESTACIONES
# =============================================================================

PATCHES_VAL = [4,25,38,46,61]
PATCHES_TEST = [27,12,32,34,57,29]
PATCHES_TRAIN = list(set(range(64))-set(PATCHES_TEST)-set(PATCHES_VAL))

estaciones = {'arboleda':{'coords':[43.2967, -3.06747],'history_LWT_allyears':None}, 
              'arrigorriaga':{'coords':[43.2165, -2.89976],'history_LWT_allyears':None}, 
              'derio':{'coords':[43.2911, -2.87342],'history_LWT_allyears':None}, 
              'deusto':{'coords':[43.2834, -2.96791],'history_LWT_allyears':None}, 
              'galindo':{'coords':[43.3062, -2.99878],'history_LWT_allyears':None}, 
              'puntagalea':{'coords':[43.3752, -3.03608],'history_LWT_allyears':None}, 
              'zorroza':{'coords':[43.284980, -2.968458],'history_LWT_allyears':None}}

coords = np.load(FOLDER_DATOS+'all_coords_1.npy')

for estacion in estaciones.keys():
    
    fileName = FOLDER_DATOS + estacion+'_filtered.csv'
    estaciones[estacion]['history_LWT_allyears'] = read_csv(fileName,2,1,',')

    fileName = FOLDER_DATOS + estacion+'_filtered_2017.csv'
    estaciones[estacion]['history_allDays_2017'] = read_csv(fileName,1,0,',')

# =============================================================================
# LEEMOS LWT DAYS Y GENERAMOS DAYS 2017
# =============================================================================

fileLWT = FOLDER_DATOS + 'LWT_days.csv'
from datetime import datetime
dateparse = lambda x: datetime.strptime(x, '%d/%m/%Y')

dfLWT = pd.read_csv(fileLWT, parse_dates=['date'], date_parser=dateparse)
listLWT = list(dfLWT.date.dt.strftime("%Y-%m-%d"))

dfDays2017 = pd.date_range(start="2017-06-01",end="2017-09-30")
listDays2017 = list(dfDays2017.strftime("%Y-%m-%d"))

# =============================================================================
# LEEMOS Y NORMALIZACION
# =============================================================================

minY,maxY = np.load(FOLDER_DATOS+'min_max_Ta_data.npy')
pendiente = maxY-minY
offset = (maxY+minY)/2.

dictResults = {}

for h in tqdm(range(24)):

    dictResults[h] = {'history':None,'scoresTrain':None,'scoresTest':None}

    with open(FOLDER_RESULTS+'results_unet_trained_over_LWT_of_range(2008, 2018)_hour_'+str(h)+'.pkl','rb') as fid:
        dictRead = pickle.load(fid)

    dictResults[h]['historyloss'] = dictRead['history']
    dictResults[h]['scoresTrain'] = dictRead['scoresTrain_LWT']
    dictResults[h]['scoresTest'] = dictRead['scoresTest_LWT']
    
    dictResults[h]['predsTrain'] = dictRead['predsTrain_LWT']
    dictResults[h]['predsVal'] = dictRead['predsVal_LWT']
    dictResults[h]['predsTest'] = dictRead['predsTest_LWT']
    dictResults[h]['trueTrain'] = dictRead['trueTrain_LWT']
    dictResults[h]['trueVal'] = dictRead['trueVal_LWT']
    dictResults[h]['trueTest'] = dictRead['trueTest_LWT']
    
    dictResults[h]['predsTrain'] = pendiente*dictResults[h]['predsTrain'] + offset-273.15
    dictResults[h]['predsVal'] = pendiente*dictResults[h]['predsVal'] + offset-273.15
    dictResults[h]['predsTest'] = pendiente*dictResults[h]['predsTest'] + offset-273.15
    dictResults[h]['trueTrain'] = pendiente*dictResults[h]['trueTrain'] + offset-273.15
    dictResults[h]['trueVal'] = pendiente*dictResults[h]['trueVal'] + offset-273.15
    dictResults[h]['trueTest'] = pendiente*dictResults[h]['trueTest'] + offset-273.15

# =============================================================================
# BUSCAMOS PREDICCION PARA CADA ESTACION Y HORA
# =============================================================================

if COMPUTEGRIDS == True:

    import itertools

    DIST = 16000
        
    place = ["Bilbao, Spain"]
    G = ox.graph_from_place(place, retain_all=True, simplify = True, network_type='drive', buffer_dist=DIST)
    buildings = ox.geometries_from_address(place, dist=DIST, tags={'building':True})
    fig, ax = ox.plot_figure_ground(G, buildings, network_type='all', dist=DIST, edge_color="gray", bgcolor = 'w', dpi=300)

    indexCoord = 0
    grids = {}

    row,col = np.unravel_index(range(31*31), (31,31))
    rowcol = np.vstack((row,col)).T

    for p in coords:
        
        ax.plot([p[0][1],p[1][1],p[3][1],p[2][1],p[0][1]],[p[0][0],p[1][0],p[3][0],p[2][0],p[0][0]],'b',lw=2)

        imToPlot = np.reshape(np.linspace(0,1,31*31),(31,31))#*np.nan
        angle = np.arctan((p[1][0]-p[0][0])/(p[1][1]-p[0][1]))*180./np.pi
        scale_x = np.sqrt((p[1][0]-p[0][0])**2+(p[1][1]-p[0][1])**2)
        scale_y = np.sqrt((p[2][0]-p[0][0])**2+(p[2][1]-p[0][1])**2)
        
        im = ax.matshow(imToPlot, origin='lower',extent=[0, scale_x, 0, scale_y])
        limits = im.get_extent()
        x = np.linspace(limits[0],limits[1],(2*31)+1)
        y = np.linspace(limits[2],limits[3],(2*31)+1)
        x = x[1::2]
        y = y[1::2]
        xx=np.squeeze(list(itertools.product(x,y)))

        transform = mtransforms.Affine2D().skew_deg(-5.2, 0).rotate_deg(angle).translate(p[0][1],p[0][0])
        
        trans_data = transform + ax.transData
        im.set_transform(trans_data)
        xxt = transform.transform(xx)
        grids[indexCoord] = xxt
        
        if indexCoord in PATCHES_TEST:
            indexPoint = 0
            for xy in xxt:
                ax.plot(xy[0],xy[1],'ko')
                ax.annotate(str(rowcol[indexPoint]),xy=xy,fontsize=8,ha='center',va='top')
                indexPoint = indexPoint + 1

        indexCoord = indexCoord + 1

    for estacion in estaciones.keys():
        ax.plot(estaciones[estacion]['coords'][1],estaciones[estacion]['coords'][0],'ro',80)

    plt.close()

    with open(FOLDER_DATOS + 'gridsPatches.pkl', 'wb') as f:
        pickle.dump(grids, f)

else:

    with open(FOLDER_DATOS + 'gridsPatches.pkl', 'rb') as f:
        grids = pickle.load(f)

# We create the scores dict

scores = {}
row,col = np.unravel_index(range(31*31), (31,31))
rowcol = np.vstack((row,col)).T

FOLDER_DATOS = '/home/jdelser/Work/UNET-NEW/DATA/'
LOOKBACK=2

with open(FOLDER_DATOS+'dictTest_LWT.pkl','rb') as fid:
    testData = pickle.load(fid)

for estacion in tqdm(estaciones.keys()):
    
    scores[estacion] = {'ventana':[],'pointInVentana':[],
                      'vsURBCLIM':{'R2':None,'MSE':None,'MAE':None,'pearson':None,'MAPE':None,'RMSE':None},
                      'vsUNET':{'R2':None,'MSE':None,'MAE':None,'pearson':None,'MAPE':None,'RMSE':None},
                      'UNETvsURBCLIM':{'R2':None,'MSE':None,'MAE':None,'pearson':None,'MAPE':None,'RMSE':None}}

    serieEstacion = estaciones[estacion]['history_LWT_allyears']

    coordsEstacion = estaciones[estacion]['coords']

    minDist = np.inf
    selectedVentanaTest = None
    selectedPointInVentanaTest = None
    selectedPointInVentanaTest_rowcol = None

    for itest in PATCHES_TEST:
            
            grid = grids[itest]
            distances = [np.linalg.norm(coordsEstacion-g) for g in np.fliplr(grid)]
            minDistanceInGrid = min(distances)
            pointMinDistanceInGrid = np.argmin(distances)

            if minDistanceInGrid<minDist:
                minDist = minDistanceInGrid
                selectedVentanaTest = itest
                selectedVentanaTestIndiceLineal = PATCHES_TEST.index(itest)
                selectedPointInVentanaTest = pointMinDistanceInGrid
                selectedPointInVentanaTest_rowcol = rowcol[pointMinDistanceInGrid]
    
    #print(estacion,coordsEstacion,selectedVentanaTest)

    predsUNET = []
    predsUNET_nans = []
    predsURBCLIM = []
    predsURBCLIM_nans = []
    serieEstacionCom = []
    serieEstacionCom_nans = []
    labelsEstacionCom = []
    labelsEstacionCom_nans = []

    for day in listLWT:
        
        indicesLWTDayInserieEstacion = [i for i in range(len(serieEstacion)) if serieEstacion[i][1]==day]
        indiceLWTDayinListLWT = listLWT.index(day)

        if len(indicesLWTDayInserieEstacion)==24:                

            datosLWTDayReal = [serieEstacion[index][0] for index in indicesLWTDayInserieEstacion]
            serieEstacionCom.extend(datosLWTDayReal)
            serieEstacionCom_nans.extend(datosLWTDayReal)
            
            for HOUR in range(24):
                labelsEstacionCom.append(str(HOUR)+':00,'+day)
                labelsEstacionCom_nans.append(str(HOUR)+':00,'+day)
                
                if HOUR in [0,1] and indiceLWTDayinListLWT==66:
                    predsUNET.append(-1)
                    predsUNET_nans.append(np.nan)
                    predsURBCLIM.append(-1)
                    predsURBCLIM_nans.append(np.nan)
                else:
                    if HOUR in [0,1] and indiceLWTDayinListLWT>66:
                        indexDayEffective = indiceLWTDayinListLWT -1
                    else:
                        indexDayEffective = indiceLWTDayinListLWT
                    
                    predsUNET.append(testData[selectedVentanaTest]['UNET'][HOUR][indexDayEffective,selectedPointInVentanaTest_rowcol[1],selectedPointInVentanaTest_rowcol[0]])
                    predsUNET_nans.append(testData[selectedVentanaTest]['UNET'][HOUR][indexDayEffective,selectedPointInVentanaTest_rowcol[1],selectedPointInVentanaTest_rowcol[0]])
                    predsURBCLIM.append(testData[selectedVentanaTest]['URBCLIM'][HOUR][indexDayEffective,selectedPointInVentanaTest_rowcol[1],selectedPointInVentanaTest_rowcol[0]])
                    predsURBCLIM_nans.append(testData[selectedVentanaTest]['URBCLIM'][HOUR][indexDayEffective,selectedPointInVentanaTest_rowcol[1],selectedPointInVentanaTest_rowcol[0]])
        
        else:
            
            predsUNET_nans.extend([np.nan]*24)
            predsURBCLIM_nans.extend([np.nan]*24)
            serieEstacionCom_nans.extend([np.nan]*24)
            for HOUR in range(24):
                labelsEstacionCom.append(str(HOUR)+':00,'+day)
                labelsEstacionCom_nans.append(str(HOUR)+':00,'+day)


    minusOnes_predsUNET =  [i for i in range(len(predsUNET)) if predsUNET[i]<0]
    minusOnes_predsURBCLIM =  [i for i in range(len(predsURBCLIM)) if predsURBCLIM[i]<0]
    minusOnes_serieEstacionCom = [i for i in range(len(serieEstacionCom)) if serieEstacionCom[i]<0]

    allMinuses = list(set(minusOnes_predsUNET + minusOnes_predsURBCLIM + minusOnes_serieEstacionCom))

    predsUNET = [predsUNET[i] for i in range(len(predsUNET)) if i not in allMinuses]
    predsURBCLIM = [predsURBCLIM[i] for i in range(len(predsURBCLIM)) if i not in allMinuses]
    labelsEstacionCom = [labelsEstacionCom[i] for i in range(len(labelsEstacionCom)) if i not in allMinuses]
    serieEstacionCom = [serieEstacionCom[i] for i in range(len(serieEstacionCom)) if i not in allMinuses]

    pearsonUNET = np.corrcoef(predsUNET,serieEstacionCom)
    pearsonURBCLIM = np.corrcoef(predsURBCLIM,serieEstacionCom)
    pearsonURBCLIMUNET = np.corrcoef(predsURBCLIM,predsUNET)

    scores[estacion]['predsURBCLIM'] = predsURBCLIM
    scores[estacion]['predsUNET'] = predsUNET
    scores[estacion]['serieEstacion'] = serieEstacionCom
    scores[estacion]['labelsPlot'] = labelsEstacionCom

    minusOnes_predsUNET_nans =  [i for i in range(len(predsUNET_nans)) if predsUNET_nans[i]<0]
    minusOnes_predsURBCLIM_nans =  [i for i in range(len(predsURBCLIM_nans)) if predsURBCLIM_nans[i]<0]
    minusOnes_serieEstacionCom_nans = [i for i in range(len(serieEstacionCom_nans)) if serieEstacionCom_nans[i]<0]

    allMinuses_nans = list(set(minusOnes_predsUNET_nans + minusOnes_predsURBCLIM_nans+ minusOnes_serieEstacionCom_nans))

    for am in allMinuses_nans:
        predsUNET_nans[am] = np.nan
        predsURBCLIM_nans[am] = np.nan
        serieEstacionCom_nans[am] = np.nan

    scores[estacion]['predsURBCLIM_nans'] = predsURBCLIM_nans
    scores[estacion]['predsUNET_nans'] = predsUNET_nans
    scores[estacion]['serieEstacion_nans'] = serieEstacionCom_nans
    scores[estacion]['labelsPlot_nans'] = labelsEstacionCom_nans

    print('unet',pearsonUNET[0,1])
    print('urbclim',pearsonURBCLIM[0,1])

    scores[estacion]['ventana'] = selectedVentanaTest
    scores[estacion]['pointInVentana'] = selectedPointInVentanaTest_rowcol

    scores[estacion]['vsURBCLIM']['R2'] = r2_score(serieEstacionCom,predsURBCLIM)
    scores[estacion]['vsUNET']['R2'] = r2_score(serieEstacionCom,predsUNET)
    scores[estacion]['UNETvsURBCLIM']['R2'] = r2_score(predsURBCLIM,predsUNET)

    scores[estacion]['vsURBCLIM']['MSE'] = mean_squared_error(serieEstacionCom,predsURBCLIM)
    scores[estacion]['vsUNET']['MSE'] = mean_squared_error(serieEstacionCom,predsUNET)
    scores[estacion]['UNETvsURBCLIM']['MSE'] = mean_squared_error(predsURBCLIM,predsUNET)

    scores[estacion]['vsURBCLIM']['RMSE'] = root_mean_squared_error(serieEstacionCom,predsURBCLIM)
    scores[estacion]['vsUNET']['RMSE'] = root_mean_squared_error(serieEstacionCom,predsUNET)
    scores[estacion]['UNETvsURBCLIM']['RMSE'] = root_mean_squared_error(predsURBCLIM,predsUNET)

    scores[estacion]['vsURBCLIM']['MAE'] = mean_absolute_error(serieEstacionCom,predsURBCLIM)
    scores[estacion]['vsUNET']['MAE'] = mean_absolute_error(serieEstacionCom,predsUNET)
    scores[estacion]['UNETvsURBCLIM']['MAE'] = mean_absolute_error(predsURBCLIM,predsUNET)

    scores[estacion]['vsURBCLIM']['MAPE'] = 100.*mean_absolute_percentage_error(serieEstacionCom,predsURBCLIM)
    scores[estacion]['vsUNET']['MAPE'] = 100.*mean_absolute_percentage_error(serieEstacionCom,predsUNET)
    scores[estacion]['UNETvsURBCLIM']['MAPE'] = 100.*mean_absolute_percentage_error(predsURBCLIM,predsUNET)

    scores[estacion]['vsURBCLIM']['pearson'] = pearsonURBCLIM[0,1]
    scores[estacion]['vsUNET']['pearson'] = pearsonUNET[0,1]
    scores[estacion]['UNETvsURBCLIM']['pearson'] = pearsonURBCLIMUNET[0,1]

if PLOT_TABLE == True:

    estacionesNombres = list(estaciones.keys())

    for estacion in estacionesNombres:
        print('\\texttt{',estacion,'} &',sep='',end='')
        for technique in ['vsURBCLIM','vsUNET']:#,UNET'','vsUNET']:
            for score in ['pearson','RMSE','MAE']:
                print(round(scores[estacion][technique][score],2),'& ',end='')
            print(round(scores[estacion][technique]['MAPE'],2),'\\% \\\\')


if PLOT_TIME_SERIES == True:

    fig,axs = plt.subplots(7,1,sharex=True)

    for estacion in estacionesNombres:
        for axis in ['top','bottom','left','right']:
            axs[estacionesNombres.index(estacion)].spines[axis].set_linewidth(2)
        
        if estacion == estacionesNombres[0]:
            axs[estacionesNombres.index(estacion)].plot(scores[estacion]['serieEstacion_nans'],'k',lw=2,label='Real')
            axs[estacionesNombres.index(estacion)].plot(scores[estacion]['predsURBCLIM_nans'],'r',label='URBCLIM')
            axs[estacionesNombres.index(estacion)].plot(scores[estacion]['predsUNET_nans'],'g',label='UNET')
            axs[estacionesNombres.index(estacion)].legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=3, fancybox=True, shadow=True,fontsize=14)
        else:
            axs[estacionesNombres.index(estacion)].plot(scores[estacion]['serieEstacion_nans'],'k',lw=2)
            axs[estacionesNombres.index(estacion)].plot(scores[estacion]['predsURBCLIM_nans'],'r')
            axs[estacionesNombres.index(estacion)].plot(scores[estacion]['predsUNET_nans'],'g')

        ticksWhere = []
        for x in range(len(scores[estacion]['labelsPlot_nans'])):
            if scores[estacion]['labelsPlot_nans'][x][0:4]=='0:00':
                axs[estacionesNombres.index(estacion)].axvline(x,0,1,c='k',ls='--')
                ticksWhere.append(x+12)
        
        axs[estacionesNombres.index(estacion)].set_ylabel(estacion,fontsize=14)
        axs[estacionesNombres.index(estacion)].set_xticks(ticksWhere)
        axs[estacionesNombres.index(estacion)].set_xticklabels([scores[estacion]['labelsPlot_nans'][t].split(',')[1][0:4]+'\n'+scores[estacion]['labelsPlot_nans'][t].split(',')[1][5:].replace('-','/') for t in ticksWhere],fontsize=10)
        axs[estacionesNombres.index(estacion)].set_xlim(3000,3930)
        axs[estacionesNombres.index(estacion)].set_ylim(0,50)

    fig.set_size_inches([14.46,  9.29])

    fig.subplots_adjust(top=0.944,
    bottom=0.032,
    left=0.054,
    right=0.988,
    hspace=0.172,
    wspace=0.2)