import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pickle

###################################################################################################
# CONST
###################################################################################################

IMG_SIZE_X = 32
IMG_SIZE_Y = 32
MASK = np.ones((IMG_SIZE_X,IMG_SIZE_Y), order='C').astype(float)
MASK[:,-1]=0
MASK[-1,:]=0
MASK = K.constant(MASK)

###################################################################################################
# PARAMS
###################################################################################################

folderData = '/home/jdelser/Work/UNET-NEW/DATA/'
folderModels = '/home/jdelser/Work/UNET-NEW/MODELS/PER_YEAR_ALL_HOURS/'
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNINGRATE = 1e-4
YEARS_SIM = [2017] # 2008-2017 (inc.)
LOOKBACK = 3+1 # 3 past steps + current one
PATCHES_VAL = [4,25,38,46,61]
PATCHES_TEST = [27,12,32,34,57,29]
PATCHES_TRAIN = list(set(range(64))-set(PATCHES_TEST)-set(PATCHES_VAL))

###################################################################################################
# FUNCS
###################################################################################################

def create_dataset(dataset, lookback):
    if(len(np.shape(dataset))>2):
        X = [np.transpose(dataset[i-lookback:i],[1,2,0,3]).reshape(IMG_SIZE_X,IMG_SIZE_Y,-1) for i in range(lookback, len(dataset))]
    else:
        X = [dataset[i-lookback:i] for i in range(lookback, len(dataset))]
    return X

def get_masked_loss():
    
    def masked_loss(yTrue,yPred):
        
        yTrue = yTrue * MASK   
        yPred = yPred * MASK
        
        mean_loss = K.sum(K.square(yPred - yTrue))/K.sum(MASK)
        return mean_loss
        
    return masked_loss

def double_conv_block(x, n_filters):

   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

   return x


def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPooling2D(2)(f)
   p = layers.Dropout(0.2)(p)

   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)

   return x

def build_unet_with_meteo_window():
    
    inputs_spatial = layers.Input(shape=(IMG_SIZE_X,IMG_SIZE_Y,3*LOOKBACK))
    inputs_meteo = layers.Input(shape=(5*(LOOKBACK),1))
    
    ### [First half of the network: downsampling inputs] ###

    # encoder: contracting path - downsample
    # downsample
    f1, p1 = downsample_block(inputs_spatial, 64)
    # downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    # f4, p4 = downsample_block(p3, 512)

    # 5 - bottleneck

    x = layers.Flatten()(p3) 
    x_meteo = layers.Flatten()(inputs_meteo) 
    x = layers.Concatenate(axis=1)([x, x_meteo])
    
    u6 = layers.Dense(4*4*256)(x)

    u6 = layers.Reshape((4,4,256))(u6)

    # decoder: expanding path - upsample
    # 6 - upsample
    # u6 = upsample_block(y3, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # upsample
    u9 = upsample_block(u8, f1, 64)

    # outputs
    outputs = layers.Conv2D(1, 1, padding="same", activation = "linear")(u9)

    # unet model with Keras Functional API
    unet_model = Model(inputs= [inputs_spatial, inputs_meteo], outputs=outputs, name="U-Net")

    return unet_model

###################################################################################################
# MAIN
###################################################################################################

# READ DATA

fileData = 'Spatial_data_all_hours.npy'

dataSpatial = np.load(folderData+fileData)
dataSpatial = np.pad(dataSpatial,((0, 0), (0, 1), (0, 1),(0,0)),constant_values=(0,0))

# Save dataSpatial in patch-wise dict

dataSpatialDict = {}

for ipatch in range(64):
    dataSpatialDict[ipatch]={}
    for year in YEARS_SIM:
        indexIni = ipatch*int(1873920/64) + (year-2008)*24*(30+31+31+30)
        indexFin = ipatch*int(1873920/64) + (year-2008+1)*24*(30+31+31+30)
        dataSpatialDict[ipatch][year] = dataSpatial[indexIni:indexFin]

# Now we read climate variables for the selected range

filesClimate = ['ERA5_bilbao_200801010000-201712312300_temperature.csv',
                'ERA5-Mod_bilbao_L137-137_200801010000-201712312300_specific-humidity.csv',
                'ERA5_bilbao_200801010000-201712312300_precipitation.csv',
                'ERA5_bilbao_200801010000-201712312300_wind-v.csv',
                'ERA5_bilbao_200801010000-201712312300_wind-u.csv']

namesClimate = ['temp','humid','prec','wind-v','wind-u']

dataClimateDict = {}

for iClimate in range(len(filesClimate)):
    dataTemp = pd.read_csv(folderData+filesClimate[iClimate],sep=';')
    dataTemp['date'] = pd.to_datetime(dataTemp.date, format="%d/%m/%Y %H:%M")

    # We normalize the climate variable (the remaining ones are already normalized)

    featureName = dataTemp.columns[1]

    minTemp = np.min(dataTemp.values[:,1])
    maxTemp = np.max(dataTemp.values[:,1])

    dataTemp[featureName] = 2*(dataTemp[featureName]-minTemp)/(maxTemp-minTemp)-1

    dataClimateDict[namesClimate[iClimate]]={}

    for year in YEARS_SIM:
        start_date = str(year) + "-06-01 00:00:00"
        end_date = str(year) + "-09-30 23:00:00"
        mask = (dataTemp['date'] >= start_date) & (dataTemp['date'] <= end_date)
        dataClimateDict[namesClimate[iClimate]][year] = dataTemp.loc[mask].values[:,1].astype(float)

# And now we read the variable to be estimated (Ta)

fileTa = 'Ta_windows_allhours.npy'
dataTa = np.load(folderData + fileTa)
dataTa = dataTa.reshape(len(dataTa), dataTa.shape[1], dataTa.shape[2], 1)
dataTa = np.pad(dataTa,((0, 0), (0, 1), (0, 1),(0,0)),constant_values=(0,0))

dataTaDict = {}

for ipatch in range(64):
    dataTaDict[ipatch]={}
    for year in YEARS_SIM:
        indexIni = ipatch*int(1873920/64) + (year-2008)*24*(30+31+31+30)
        indexFin = ipatch*int(1873920/64) + (year-2008+1)*24*(30+31+31+30)
        dataTaDict[ipatch][year] = dataTa[indexIni:indexFin]

# We construct the training/val/test sets: in this case we consider two cases:
# 1) unseen patches; 
# 2) one year for val and another one for test
# We will start with model 1

# YEARS_VAL = [2016]
# YEARS_TEST = [2017]
# YEARS_TRAIN = list(set(range(2008,2018))-set(YEARS_TEST)-set(YEARS_VAL))

Xspatial_train = []
Xmeteo_train = []
Y_train = []

for ipatch in PATCHES_TRAIN:
    for year in YEARS_SIM:
        Xspatial_train.extend(create_dataset(dataSpatialDict[ipatch][year],LOOKBACK))
        Y_train.extend(dataTaDict[ipatch][year][LOOKBACK:])
        Xmeteo_tmp = []
        for iClimate in range(len(namesClimate)):
            Xmeteo_tmp.append(create_dataset(dataClimateDict[namesClimate[iClimate]][year],LOOKBACK))
        Xmeteo_train.extend(np.concatenate(Xmeteo_tmp,-1))

# Xspatial_val = []
# Xmeteo_val = []
# Y_val = []

# for ipatch in PATCHES_VAL:
#     for year in YEARS_SIM:
#         Xspatial_val.extend(create_dataset(dataSpatialDict[ipatch][year],LOOKBACK))
#         Y_val.extend(dataTaDict[ipatch][year][LOOKBACK:])
#         Xmeteo_tmp = []
#         for iClimate in range(len(namesClimate)):
#             Xmeteo_tmp.append(create_dataset(dataClimateDict[namesClimate[iClimate]][year],LOOKBACK))
#         Xmeteo_val.extend(np.concatenate(Xmeteo_tmp,-1))

# Xspatial_test = []
# Xmeteo_test = []
# Y_test = []

# for ipatch in PATCHES_TEST:
#     for year in YEARS_SIM:
#         Xspatial_test.extend(create_dataset(dataSpatialDict[ipatch][year],LOOKBACK))
#         Y_test.extend(dataTaDict[ipatch][year][LOOKBACK:])
#         Xmeteo_tmp = []
#         for iClimate in range(len(namesClimate)):
#             Xmeteo_tmp.append(create_dataset(dataClimateDict[namesClimate[iClimate]][year],LOOKBACK))
#         Xmeteo_test.extend(np.concatenate(Xmeteo_tmp,-1))

del dataTaDict
del dataClimateDict
del dataSpatialDict

Xspatial_train = np.squeeze(Xspatial_train)
Xmeteo_train = np.squeeze(Xmeteo_train)
Y_train = np.squeeze(Y_train)

# Xspatial_val = np.squeeze(Xspatial_val)
# Xmeteo_val = np.squeeze(Xmeteo_val)
# Y_val = np.squeeze(Y_val)

# Model history and train

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

unet_model = build_unet_with_meteo_window()

unet_model.compile(optimizer=Adam(learning_rate=LEARNINGRATE),loss = get_masked_loss(), metrics="mae")

model_history = unet_model.fit([Xspatial_train,Xmeteo_train], Y_train, batch_size=BATCH_SIZE,epochs=NUM_EPOCHS)#, validation_data=([Xspatial_val,Xmeteo_val], Y_val), callbacks=[es])

unet_model.save(folderModels + 'unet_trained_over_' + str(YEARS_SIM) + '.keras')

with open(folderModels + 'unet_trained_over_' + str(YEARS_SIM) + '_history.pkl', 'wb') as file_history:
     pickle.dump(model_history.history, file_history)