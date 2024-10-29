import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import pickle
import tensorflow as tf

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
folderModels = '/home/jdelser/Work/UNET-NEW/MODELS/LWT_PER_HOUR/'
LOOKBACK = 2 # 2 past steps + current one
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNINGRATE = 1e-4
# 2008-2018 (inclusive)
YEARS_SIM = range(2008,2019) 
YEARS_TRAIN = range(2008,2018) 
# YEARS_VAL = range(2017,2018) 
# YEARS_TEST = range(2018,2019)
PATCHES_VAL = [4,25,38,46,61]
PATCHES_TEST = [27,12,32,34,57,29]
PATCHES_TRAIN = list(set(range(64))-set(PATCHES_TEST)-set(PATCHES_VAL))

###################################################################################################
# FUNCS
###################################################################################################

def create_dataset_for_some_dates_and_given_hour(dataset, lookback, selectedDates, selectedHour):
    if(len(np.shape(dataset['data']))>2):
        X = [np.transpose(dataset['data'][i-lookback:i+1],[1,2,0,3]).reshape(IMG_SIZE_X,IMG_SIZE_Y,-1) for i in range(lookback, len(dataset['data'])) 
            if dataset['datetime'][i].strftime("%Y-%m-%d") in selectedDates and dataset['datetime'][i].hour==selectedHour]
    else:
        X = [dataset['data'][i-lookback:i+1] for i in range(lookback, len(dataset['data'])) 
            if dataset['datetime'][i].strftime("%Y-%m-%d") in selectedDates and dataset['datetime'][i].hour==selectedHour]
    return X

def create_target_variable_for_some_dates_and_given_hour(dataset, lookback, selectedDates,selectedHour):
    Y = [dataset['data'][i] for i in range(lookback, len(dataset['data'])) 
         if dataset['datetime'][i].strftime("%Y-%m-%d") in selectedDates and dataset['datetime'][i].hour==selectedHour]
    return Y

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
    
    inputs_spatial = layers.Input(shape=(IMG_SIZE_X,IMG_SIZE_Y,3*(LOOKBACK+1)))
    inputs_meteo = layers.Input(shape=(5*(LOOKBACK+1),1))
    
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

####################################################
# We read LWT days
####################################################

fileLWT = folderData + 'LWT_days.csv'
from datetime import datetime
dateparse = lambda x: datetime.strptime(x, '%d/%m/%Y')

dfLWT = pd.read_csv(fileLWT, parse_dates=['date'], date_parser=dateparse)
listLWT = list(dfLWT.date.dt.strftime("%Y-%m-%d"))

# READ DATA

fileData = 'Spatial_data_all_hours.npy'

dataSpatial = np.load(folderData+fileData)
dataSpatial = np.pad(dataSpatial,((0, 0), (0, 1), (0, 1),(0,0)),constant_values=(0,0))

# Save dataSpatial in patch-wise dict

dataSpatialDict = {}

for ipatch in range(64):
    dataSpatialDict[ipatch]={}
    for year in YEARS_SIM:
        dataSpatialDict[ipatch][year] = {}
        indexIni = ipatch*int(1873920/64) + (year-2008)*24*(30+31+31+30)
        indexFin = ipatch*int(1873920/64) + (year-2008+1)*24*(30+31+31+30)
        d = pd.date_range(str(year)+'-06-01 00:00',str(year)+'-09-30 23:00',freq='h')
        dataSpatialDict[ipatch][year]['data'] = dataSpatial[indexIni:indexFin]
        dataSpatialDict[ipatch][year]['datetime'] = d

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
        dataClimateDict[namesClimate[iClimate]][year]={}
        start_date = str(year) + "-06-01 00:00:00"
        end_date = str(year) + "-09-30 23:00:00"
        mask = (dataTemp['date'] >= start_date) & (dataTemp['date'] <= end_date)
        d = pd.date_range(str(year)+'-06-01 00:00',str(year)+'-09-30 23:00',freq='h')
        dataClimateDict[namesClimate[iClimate]][year]['data'] = dataTemp.loc[mask].values[:,1].astype(float)
        dataClimateDict[namesClimate[iClimate]][year]['datetime'] = d

# And now we read the variable to be estimated (Ta)

fileTa = 'Ta_windows_allhours.npy'
dataTa = np.load(folderData + fileTa)
dataTa = dataTa.reshape(len(dataTa), dataTa.shape[1], dataTa.shape[2], 1)
dataTa = np.pad(dataTa,((0, 0), (0, 1), (0, 1),(0,0)),constant_values=(0,0))

dataTaDict = {}

for ipatch in range(64):
    dataTaDict[ipatch]={}
    for year in YEARS_SIM:
        dataTaDict[ipatch][year] = {}
        indexIni = ipatch*int(1873920/64) + (year-2008)*24*(30+31+31+30)
        indexFin = ipatch*int(1873920/64) + (year-2008+1)*24*(30+31+31+30)
        d = pd.date_range(str(year)+'-06-01 00:00',str(year)+'-09-30 23:00',freq='h')
        dataTaDict[ipatch][year]['data'] = dataTa[indexIni:indexFin]
        dataTaDict[ipatch][year]['datetime'] = d

# We construct the training/val/test sets: in this case we consider two cases:
# 1) unseen patches; 
# 2) one year for val and another one for test

for hour in range(0,24):
    
    print("HOUR: "+str(hour))
    
    Xspatial_train = []
    Xmeteo_train = []
    Y_train = []

    for ipatch in tqdm(PATCHES_TRAIN):
        for year in YEARS_TRAIN:
            Xspatial_train.extend(create_dataset_for_some_dates_and_given_hour(dataSpatialDict[ipatch][year],LOOKBACK,listLWT,hour))
            Y_train.extend(create_target_variable_for_some_dates_and_given_hour(dataTaDict[ipatch][year],LOOKBACK,listLWT,hour))
            Xmeteo_tmp = []
            for iClimate in range(len(namesClimate)):
                Xmeteo_tmp.append(create_dataset_for_some_dates_and_given_hour(dataClimateDict[namesClimate[iClimate]][year],LOOKBACK,listLWT,hour))
            Xmeteo_train.extend(np.concatenate(Xmeteo_tmp,-1))

    Xspatial_train = np.squeeze(Xspatial_train)
    Xmeteo_train = np.squeeze(Xmeteo_train)
    Y_train = np.squeeze(Y_train)

    tf.keras.backend.clear_session()

    unet_model = build_unet_with_meteo_window()

    unet_model.compile(optimizer=Adam(learning_rate=LEARNINGRATE),loss = get_masked_loss(), metrics="mae")

    model_history = unet_model.fit([Xspatial_train,Xmeteo_train], Y_train, batch_size=BATCH_SIZE,epochs=NUM_EPOCHS)#, validation_data=([Xspatial_val,Xmeteo_val], Y_val), callbacks=[es])

    unet_model.save(folderModels + 'unet_trained_over_LWT_of_' + str(YEARS_TRAIN) + '_hour_'+ str(hour) + '.keras')

    with open(folderModels + 'unet_trained_over_LWT_of_' + str(YEARS_TRAIN) + '_hour_'+ str(hour)+'_history.pkl', 'wb') as file_history:
         pickle.dump(model_history.history, file_history)

