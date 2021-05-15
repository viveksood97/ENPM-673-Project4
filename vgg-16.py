from pathlib import Path
import os
import pandas as pd
import time
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

def prep_fn(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2
    return img

start_lr = 0.00001
min_lr = 0.00001
max_lr = 0.00005




rampup_epochs = 5
sustain_epochs = 0
exp_decay = .8

def lrfn(epoch):
    if epoch < rampup_epochs:
        return (max_lr - start_lr)/rampup_epochs * epoch + start_lr
    elif epoch < rampup_epochs + sustain_epochs:
        return max_lr
    else:
        return (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr

def main(path):
    
    File_Path = list(path.glob(r"**/*.png"))
    df = pd.DataFrame(columns=('path', 'fish_class'))
    data = []

    for path in File_Path:
        fish_class = str(path).split("/")[-2]
        if fish_class.split(" ")[-1] != 'GT':
            data.append([str(path), fish_class])
            
    
    df = pd.DataFrame(data, columns = ['path', 'fish_class'])

    train_set=df.sample(frac=0.8,random_state=200)
    
    test_set=df.drop(train_set.index)
    

    

    img_gen = ImageDataGenerator(preprocessing_function=prep_fn)

# img_gen cannot take in an array, so ensure the data that is been passed is a dataframe
    train = img_gen.flow_from_dataframe(dataframe = train_set,
        x_col = 'path', #name of the column containing the image in the train set
        y_col ='fish_class', #name of column containing the target in the train set
        target_size = (224, 224),
        color_mode = 'rgb',
        class_mode = 'categorical',#the class mode here and that for the model_loss(when using sequential model)
                                        #should be the same
        batch_size = 10,
        shuffle = False #not to shuffle the given data
    )

    test = img_gen.flow_from_dataframe(dataframe = test_set,
        x_col = 'path', #name of the column containing the image in the test set
        y_col ='fish_class', #name of column containing the target in the test set
        target_size =(224, 224),
        color_mode ='rgb',
        class_mode ='categorical',
        batch_size = 10,
        shuffle = False # not to shuffle the given data
    )

    input_shape = (224, 224, 3)

    model = Sequential(
        [
            Conv2D(input_shape = (224, 224, 3), filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"),
            BatchNormalization(),
            Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"),
            BatchNormalization(),
            MaxPool2D(pool_size = (2,2)),

            Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"),
            BatchNormalization(),
            Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"),
            BatchNormalization(),
            MaxPool2D(pool_size = (2,2)),

            Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"),
            BatchNormalization(),
            Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"),
            BatchNormalization(),
            
            Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"),
            BatchNormalization(),
            MaxPool2D(pool_size = (2,2)),

            Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
            BatchNormalization(),
            Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
            BatchNormalization(),
            Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
            BatchNormalization(),
            MaxPool2D(pool_size = (2,2)),

            Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
            BatchNormalization(),
            Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
            BatchNormalization(),
            Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
            BatchNormalization(),
            MaxPool2D(pool_size = (2,2)),

            Flatten(),
            Dense(units = 4096, activation = "relu"),
            BatchNormalization(),
            Dropout(0.5, seed=73),
            Dense(units = 4096, activation = "relu"),
            BatchNormalization(),
            Dropout(0.5, seed=73),
            Dense(units = 9, activation = "softmax"),
        ])



    
    model.summary()
    lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=False)
    model.compile(optimizer='adam', # optimize the model with adam optimizer
                loss="categorical_crossentropy", 
                metrics=['accuracy']) #to get accuracy of the model in each run

    history = model.fit(train, #fit the model on the training set
                        validation_data = test, #add the validation set to evaluate the performance in each run
                        epochs = 15, #train in 10 epochs
                        verbose = 1,
                        callbacks=lr_callback)



if __name__ == "__main__":
    
    path = Path(r"./Fish_Dataset/Fish_Dataset")
    start = time.time()
    main(path)
    end = time.time()
    print(end-start)
