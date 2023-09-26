# ====================================================================================
#  Author: Kunal SK Sukhija
# ====================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import missingno as msno
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#%%
train_datagen=ImageDataGenerator(zoom_range=0.2,shear_range=0.2,horizontal_flip=True,rotation_range=0.2,rescale=1/255)

test_datagen = ImageDataGenerator(rescale=1/255)
#%%
train_dataset=train_datagen.flow_from_directory(r"E:\Coding\Machine Learning\Face Mask Detection\train",class_mode="binary",target_size=(150,150),batch_size=16)

test_dataset = test_datagen.flow_from_directory(r"E:\Coding\Machine Learning\Face Mask Detection\test",class_mode="binary",target_size=(150,150),batch_size=16)
#%%
cnn=tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,input_shape=(150,150,3),activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

cnn.add(tf.keras.layers.Flatten())
#%%
cnn.add(tf.keras.layers.Dense(units=120,activation="relu"))

cnn.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
#%%
cnn.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
#%%
cnn.fit(train_dataset,validation_data=test_dataset,epochs=20)
#%%
cnn.save('Face_mask_model.h5')