import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import *

import sys
import numpy as np
from sklearn.preprocessing import StandardScaler

import util.SCA_dataset as datasets
import util.Attack as Attack

def creat_multi_binary_model(input_length, num_task, desync_level=5, classes=256):
    # Define the shared 1D CNN base model
    inputs = Input(shape=(input_length, 1))
    x = RandomTranslation(width_factor=desync_level/input_length, height_factor=0, fill_mode='wrap')(inputs)

    x = Conv1D(kernel_size=40, strides=20, filters=16, activation="selu", padding="same")(x)
    x = AveragePooling1D(pool_size=2, strides=2, padding="same")(x)
    x = BatchNormalization()(x)  

    x = Flatten()(x)
        
    # Define separate MLPs for each task
    def create_mlp(input_layer):
        mlp = Dense(200, activation="selu", kernel_initializer="random_uniform")(input_layer)
        mlp = Dense(200, activation="selu", kernel_initializer="random_uniform")(mlp)
        return mlp

    # Define separate output layers for each task
    prediction_outputs = [Dense(classes, activation='softmax', name=f'plt_byte{i}')(create_mlp(x)) for i in range(num_task)]

    # Combine the shared base model and output layers into a single Model
    model = Model(inputs=inputs, outputs=prediction_outputs)

    # Compile the model with separate losses for each task
    model.compile(optimizer=Adam(), loss=['categorical_crossentropy']*num_task, metrics=['accuracy'])
    model.summary()
    return model

def get_label(plaintext):
    return {f'plt_byte{i}': to_categorical(plaintext[:, i]) for i in range(plaintext.shape[1])}
 
if __name__ == "__main__":
    data_root = ''
    model_root = ''
    result_root = ''

    datasetss = sys.argv[1].split(',') #ASCAD
    aug_level = float(sys.argv[2]) #5
    epochs = int(sys.argv[3]) #100
    batch_size = int(sys.argv[4]) #768
    train_model = bool(int(sys.argv[5])) #1
    index = sys.argv[6] #1
        
    for dataset in datasetss:
        if dataset == 'ASCAD':
            all_key = [77, 251, 224, 242, 114, 33, 254, 16, 167, 141, 74, 220, 142, 73, 4, 105]
            X_profiling, plt_profiling = datasets.load_ascad(data_root+dataset+'/')
            all_key = all_key[2:]
            print(all_key)
            plt_profiling = plt_profiling[:, 2:]
        if dataset == 'ASCAD_rand':
            all_key = [0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255]
            X_profiling, plt_profiling = datasets.load_ascad_rand(data_root+dataset+'/')
            all_key = all_key[2:]
            plt_profiling = plt_profiling[:, 2:]
        elif dataset == 'CHES_CTF': #45000, 5000
            all_key = [77, 251, 224, 242, 114, 33, 254, 16, 167, 141, 74, 220, 142, 73, 4, 105]
            X_profiling, plt_profiling = datasets.load_chesctf(data_root+dataset+'/')
            all_key = all_key
            plt_profiling = plt_profiling

        # Normalize the data
        scaler = StandardScaler()
        X_profiling = scaler.fit_transform(X_profiling)

        # if model is CNN, we have to make the data dim equals to 3 
        X_profiling = np.expand_dims(X_profiling, axis=-1)

        test_info = '{}_{}_epoch{}_bs{}_{}'.format(dataset, aug_level, epochs, batch_size, index)
        print('====={}====='.format(test_info))
        # load and train model if needed
        if train_model:
            model = creat_multi_binary_model(X_profiling.shape[1], plt_profiling.shape[1], desync_level=aug_level, classes=256)
            model.fit(
                x=X_profiling, 
                y=get_label(plt_profiling), 
                batch_size=batch_size, 
                verbose=2, 
                epochs=epochs)      
            
            model.save(model_root+"model_{}.h5".format(test_info))
        else:
            model = load_model(model_root+"model_{}.h5".format(test_info))

        model_test = creat_multi_binary_model(X_profiling.shape[1], plt_profiling.shape[1], desync_level=0, classes=256)
        model_test.set_weights(model.get_weights()) 

        pred = np.array(model_test.predict(X_profiling))

        rank_evol = Attack.perform_attacks(pred, plt_profiling, all_key, step=100, nb_attacks=1, shuffle=False)

        print("===Error correction===")
        prediction_table, rank_container = Attack.get_prediction_table(pred, plt_profiling, all_key)
        prediction_table = Attack.error_correction(prediction_table, all_key)
