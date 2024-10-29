import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm

IMG_SIZE_X = 32
IMG_SIZE_Y = 32

FOLDER_DATOS = '/home/jdelser/Work/UNET-NEW/DATA/'
FOLDER_RESULTS = '/home/jdelser/Work/UNET-NEW/MODELS/LWT_PER_HOUR/'
LOOKBACK=2
PATCHES_VAL = [4,25,38,46,61]
PATCHES_TEST = [27,12,32,34,57,29]
PATCHES_TRAIN = list(set(range(64))-set(PATCHES_TEST)-set(PATCHES_VAL))

patches = np.load(FOLDER_DATOS+'coords_joined.npy')

RADIUS = 2
COMPUTE_TRAIN = True
COMPUTE_VAL = True
COMPUTE_TEST = True

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def computePoints(patchCoords):
    
    ticks = []
    A = tuple(patchCoords[0,:])
    B = tuple(patchCoords[1,:])
    C = tuple(patchCoords[3,:])
    D = tuple(patchCoords[2,:])

    polygon = [A, B, C, D]
    #n = 30  # number of parts on each side of the grid
    n=30

    # we first find ticks on each side of the polygon
    for j in range(4):  # because we are talking about 4-gons
        temp_ticks = []
        for i in range(n-1):
            t = (i+1)*(1/n)
            Ex = polygon[j][0] * (1-t) + polygon[(j+1) % 4][0] * t
            Ey = polygon[j][1] * (1-t) + polygon[(j+1) % 4][1] * t
            temp_ticks.append((Ex, Ey))
        if j < 2:
            ticks.append(temp_ticks)
        else: # because you are moving backward in this part
            temp_ticks.reverse()
            ticks.append(temp_ticks)

    # then we find lines of the grid
    h_lines = []
    v_lines = []

    h_lines.append(((A[0],A[1]),(D[0],D[1])))
    v_lines.append(((B[0],B[1]),(A[0],A[1])))

    for i in range(n-1):
        h_lines.append((ticks[0][i], ticks[2][i]))
        v_lines.append((ticks[1][i], ticks[3][i]))

    h_lines.append(((B[0],B[1]),(C[0],C[1])))
    
    v_lines.append(((D[0],D[1]),(C[0],C[1])))
    # then we find the intersection of grid lines

    Xs = []
    Ys = []

    for j in range(len(v_lines)):
        for i in range(len(h_lines)):
            line1 = h_lines[i]
            line2 = v_lines[j]
            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
            div = det(xdiff, ydiff)
            if div == 0:
                raise Exception('lines do not intersect')

            d = (det(*line1), det(*line2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            # print(x,y) # this is an intersection point that you want
            Xs.append(x)
            Ys.append(y)

    return Xs, Ys, polygon, h_lines, v_lines

def neighbors(a, radius, row_number, column_number):
    neighbors = -666*np.ones_like(a)
    for i in range(row_number-radius,row_number+radius+1):
        for j in range(column_number-radius,column_number+radius+1):
            if i >= 0 and i < np.shape(a)[1] and j >= 0 and j < np.shape(a)[2]:
                if i!=row_number or j!=column_number:
                    neighbors[:,i,j] = a[:,i,j]
    return neighbors

# =============================================================================
# LEEMOS Y NORMALIZACION
# =============================================================================

minY,maxY = np.load(FOLDER_DATOS+'min_max_Ta_data.npy')
pendiente = maxY-minY
offset = (maxY+minY)/2.

dictResults = {}
dictResults[LOOKBACK] = {}

for h in tqdm(range(24)):

    dictResults[LOOKBACK][h] = {'history':None,'scoresTrain':None,'scoresTest':None}

    with open(FOLDER_RESULTS+'results_unet_trained_over_LWT_of_range(2008, 2018)_hour_'+str(h)+'.pkl','rb') as fid:
        dictRead = pickle.load(fid)

    dictResults[LOOKBACK][h]['historyloss'] = dictRead['history']
    dictResults[LOOKBACK][h]['scoresTrain'] = dictRead['scoresTrain_LWT']
    dictResults[LOOKBACK][h]['scoresTest'] = dictRead['scoresTest_LWT']
    
    dictResults[LOOKBACK][h]['predsTrain'] = dictRead['predsTrain_LWT'][:,0:31,0:31,:]
    dictResults[LOOKBACK][h]['predsVal'] = dictRead['predsVal_LWT'][:,0:31,0:31,:]
    dictResults[LOOKBACK][h]['predsTest'] = dictRead['predsTest_LWT'][:,0:31,0:31,:]
    dictResults[LOOKBACK][h]['trueTrain'] = np.expand_dims(dictRead['trueTrain_LWT'],3)[:,0:31,0:31,:]
    dictResults[LOOKBACK][h]['trueVal'] = np.expand_dims(dictRead['trueVal_LWT'],3)[:,0:31,0:31,:]
    dictResults[LOOKBACK][h]['trueTest'] = np.expand_dims(dictRead['trueTest_LWT'],3)[:,0:31,0:31,:]
    
    dictResults[LOOKBACK][h]['predsTrain'] = pendiente*dictResults[LOOKBACK][h]['predsTrain'] + offset-273.15
    dictResults[LOOKBACK][h]['predsVal'] = pendiente*dictResults[LOOKBACK][h]['predsVal'] + offset-273.15
    dictResults[LOOKBACK][h]['predsTest'] = pendiente*dictResults[LOOKBACK][h]['predsTest'] + offset-273.15
    dictResults[LOOKBACK][h]['trueTrain'] = pendiente*dictResults[LOOKBACK][h]['trueTrain'] + offset-273.15
    dictResults[LOOKBACK][h]['trueVal'] = pendiente*dictResults[LOOKBACK][h]['trueVal'] + offset-273.15
    dictResults[LOOKBACK][h]['trueTest'] = pendiente*dictResults[LOOKBACK][h]['trueTest'] + offset-273.15

########################################################################################
# SAVE
########################################################################################

dictResultsPlotTrain = {}
dictResultsPlotVal = {}
dictResultsPlotTest = {}

if COMPUTE_TRAIN:

    for patch in tqdm(PATCHES_TRAIN):

        indexPatchLineal = PATCHES_TRAIN.index(patch)
        dictResultsPlotTrain[patch] = {'boundingBox':None,'Xs':None,'Ys':None,'UNET':{},'URBCLIM':{},  
                                    'UHI_INDEX_PER_HOUR_P_01':{},
                                    'UHI_INDEX_PER_HOUR_P_03':{},
                                    'UHI_INDEX_PER_HOUR_P_05':{},
                                    'UHI_INDEX_PER_HOUR_P_07':{},
                                    'UHI_INDEX_PER_HOUR_P_09':{},
                                    'UHI_INDEX_PER_HOUR_P_095':{}}
        
        patchCorners = np.fliplr(patches[patch])
        Xs, Ys, polig, h_lines, v_lines = computePoints(patchCorners.astype('float64'))
        dictResultsPlotTrain[patch]['boundingBox']=np.array(polig)
        dictResultsPlotTrain[patch]['Xs']=Xs
        dictResultsPlotTrain[patch]['Ys']=Ys

        for HOUR in range(24):

            if HOUR<LOOKBACK:
                offset = 1
            else:
                offset = 0
            
            dictResultsPlotTrain[patch]['UNET'][HOUR] = dictResults[LOOKBACK][HOUR]['predsTrain'][(164-offset)*indexPatchLineal:(164-offset)*(indexPatchLineal+1)].squeeze()
            dictResultsPlotTrain[patch]['URBCLIM'][HOUR] = dictResults[LOOKBACK][HOUR]['trueTrain'][(164-offset)*indexPatchLineal:(164-offset)*(indexPatchLineal+1)].squeeze()

            tensorResults = dictResults[LOOKBACK][HOUR]['predsTrain'][(164-offset)*indexPatchLineal:(164-offset)*(indexPatchLineal+1)]
            indiceIslaCalor_H = np.zeros_like(tensorResults)

            for i in range(np.shape(tensorResults)[1]):
                for j in range(np.shape(tensorResults)[2]):
                    neighbors_over_time_ij = neighbors(tensorResults,RADIUS,i,j)
                    values_over_time = tensorResults[:,i,j]
                    
                    for t in range(len(tensorResults)):
                        indiceIslaCalor_H[t,i,j] = 100.*(values_over_time[t]-np.mean(neighbors_over_time_ij[t,neighbors_over_time_ij[t]!=-666]))/(values_over_time[t])

            dictResultsPlotTrain[patch]['UHI_INDEX_PER_HOUR_P_01'][HOUR] = np.percentile(indiceIslaCalor_H,10,axis=0)
            dictResultsPlotTrain[patch]['UHI_INDEX_PER_HOUR_P_03'][HOUR] = np.percentile(indiceIslaCalor_H,30,axis=0)
            dictResultsPlotTrain[patch]['UHI_INDEX_PER_HOUR_P_05'][HOUR] = np.median(indiceIslaCalor_H,0)
            dictResultsPlotTrain[patch]['UHI_INDEX_PER_HOUR_P_07'][HOUR] = np.percentile(indiceIslaCalor_H,70,axis=0)
            dictResultsPlotTrain[patch]['UHI_INDEX_PER_HOUR_P_09'][HOUR] = np.percentile(indiceIslaCalor_H,90,axis=0)
            dictResultsPlotTrain[patch]['UHI_INDEX_PER_HOUR_P_095'][HOUR] = np.percentile(indiceIslaCalor_H,95,axis=0)

    with open(FOLDER_DATOS + 'dictTrain_LWT.pkl','wb') as fid:
        pickle.dump(dictResultsPlotTrain,fid)

if COMPUTE_VAL:

    for patch in tqdm(PATCHES_VAL):

        indexPatchLineal = PATCHES_VAL.index(patch)
        dictResultsPlotVal[patch] = {'boundingBox':None,'Xs':None,'Ys':None,'UNET':{},'URBCLIM':{},  
                                    'UHI_INDEX_PER_HOUR_P_01':{},
                                    'UHI_INDEX_PER_HOUR_P_03':{},
                                    'UHI_INDEX_PER_HOUR_P_05':{},
                                    'UHI_INDEX_PER_HOUR_P_07':{},
                                    'UHI_INDEX_PER_HOUR_P_09':{},
                                    'UHI_INDEX_PER_HOUR_P_095':{}}
        
        patchCorners = np.fliplr(patches[patch])
        Xs, Ys, polig, h_lines, v_lines = computePoints(patchCorners.astype('float64'))
        dictResultsPlotVal[patch]['boundingBox']=np.array(polig)
        dictResultsPlotVal[patch]['Xs']=Xs
        dictResultsPlotVal[patch]['Ys']=Ys

        for HOUR in range(24):

            if HOUR<LOOKBACK:
                offset = 1
            else:
                offset = 0

            dictResultsPlotVal[patch]['UNET'][HOUR] = dictResults[LOOKBACK][HOUR]['predsVal'][(164-offset)*indexPatchLineal:(164-offset)*(indexPatchLineal+1)].squeeze()
            dictResultsPlotVal[patch]['URBCLIM'][HOUR] = dictResults[LOOKBACK][HOUR]['trueVal'][(164-offset)*indexPatchLineal:(164-offset)*(indexPatchLineal+1)].squeeze()

            tensorResults = dictResults[LOOKBACK][HOUR]['predsVal'][(164-offset)*indexPatchLineal:(164-offset)*(indexPatchLineal+1)]
            indiceIslaCalor_H = np.zeros_like(tensorResults)

            for i in range(np.shape(tensorResults)[1]):
                for j in range(np.shape(tensorResults)[2]):
                    neighbors_over_time_ij = neighbors(tensorResults,RADIUS,i,j)
                    values_over_time = tensorResults[:,i,j]
                    
                    for t in range(len(tensorResults)):
                        indiceIslaCalor_H[t,i,j] = 100.*(values_over_time[t]-np.mean(neighbors_over_time_ij[t,neighbors_over_time_ij[t]!=-666]))/(values_over_time[t])

            dictResultsPlotVal[patch]['UHI_INDEX_PER_HOUR_P_01'][HOUR] = np.percentile(indiceIslaCalor_H,10,axis=0)
            dictResultsPlotVal[patch]['UHI_INDEX_PER_HOUR_P_03'][HOUR] = np.percentile(indiceIslaCalor_H,30,axis=0)
            dictResultsPlotVal[patch]['UHI_INDEX_PER_HOUR_P_05'][HOUR] = np.median(indiceIslaCalor_H,0)
            dictResultsPlotVal[patch]['UHI_INDEX_PER_HOUR_P_07'][HOUR] = np.percentile(indiceIslaCalor_H,70,axis=0)
            dictResultsPlotVal[patch]['UHI_INDEX_PER_HOUR_P_09'][HOUR] = np.percentile(indiceIslaCalor_H,90,axis=0)
            dictResultsPlotVal[patch]['UHI_INDEX_PER_HOUR_P_095'][HOUR] = np.percentile(indiceIslaCalor_H,95,axis=0)

    with open(FOLDER_DATOS + 'dictVal_LWT.pkl','wb') as fid:
        pickle.dump(dictResultsPlotVal,fid)

if COMPUTE_TEST:

    for patch in tqdm(PATCHES_TEST):

        indexPatchLineal = PATCHES_TEST.index(patch)
        dictResultsPlotTest[patch] = {'boundingBox':None,'Xs':None,'Ys':None,'UNET':{},'URBCLIM':{}, 
                                    'UHI_INDEX_PER_HOUR_P_01':{},
                                    'UHI_INDEX_PER_HOUR_P_03':{},
                                    'UHI_INDEX_PER_HOUR_P_05':{},
                                    'UHI_INDEX_PER_HOUR_P_07':{},
                                    'UHI_INDEX_PER_HOUR_P_09':{},
                                    'UHI_INDEX_PER_HOUR_P_095':{}}
        
        patchCorners = np.fliplr(patches[patch])
        Xs, Ys, polig, h_lines, v_lines = computePoints(patchCorners.astype('float64'))
        dictResultsPlotTest[patch]['boundingBox']=np.array(polig)
        dictResultsPlotTest[patch]['Xs']=Xs
        dictResultsPlotTest[patch]['Ys']=Ys

        for HOUR in range(24):

            if HOUR<LOOKBACK:
                offset = 1
            else:
                offset = 0
            
            dictResultsPlotTest[patch]['UNET'][HOUR] = dictResults[LOOKBACK][HOUR]['predsTest'][(164-offset)*indexPatchLineal:(164-offset)*(indexPatchLineal+1)].squeeze()
            dictResultsPlotTest[patch]['URBCLIM'][HOUR] = dictResults[LOOKBACK][HOUR]['trueTest'][(164-offset)*indexPatchLineal:(164-offset)*(indexPatchLineal+1)].squeeze()
            
            tensorResults = dictResults[LOOKBACK][HOUR]['predsTest'][(164-offset)*indexPatchLineal:(164-offset)*(indexPatchLineal+1)]
            indiceIslaCalor_H = np.zeros_like(tensorResults)

            for i in range(np.shape(tensorResults)[1]):
                for j in range(np.shape(tensorResults)[2]):
                    neighbors_over_time_ij = neighbors(tensorResults,RADIUS,i,j)
                    values_over_time = tensorResults[:,i,j]
                    
                    for t in range(len(tensorResults)):
                        indiceIslaCalor_H[t,i,j] = 100.*(values_over_time[t]-np.mean(neighbors_over_time_ij[t,neighbors_over_time_ij[t]!=-666]))/(values_over_time[t])

            dictResultsPlotTest[patch]['UHI_INDEX_PER_HOUR_P_01'][HOUR] = np.percentile(indiceIslaCalor_H,10,axis=0)
            dictResultsPlotTest[patch]['UHI_INDEX_PER_HOUR_P_03'][HOUR] = np.percentile(indiceIslaCalor_H,30,axis=0)
            dictResultsPlotTest[patch]['UHI_INDEX_PER_HOUR_P_05'][HOUR] = np.median(indiceIslaCalor_H,0)
            dictResultsPlotTest[patch]['UHI_INDEX_PER_HOUR_P_07'][HOUR] = np.percentile(indiceIslaCalor_H,70,axis=0)
            dictResultsPlotTest[patch]['UHI_INDEX_PER_HOUR_P_09'][HOUR] = np.percentile(indiceIslaCalor_H,90,axis=0)
            dictResultsPlotTest[patch]['UHI_INDEX_PER_HOUR_P_095'][HOUR] = np.percentile(indiceIslaCalor_H,95,axis=0)
            
    with open(FOLDER_DATOS + 'dictTest_LWT.pkl','wb') as fid:
        pickle.dump(dictResultsPlotTest,fid)

