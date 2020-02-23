#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')


# In[6]:


import tensorflow
from tensorflow.python.client import device_lib
from  keras.models import Sequential, Model
from  keras.layers import Input, Dense, LeakyReLU, Activation, Concatenate, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, InputLayer, Flatten, BatchNormalization, Reshape, Lambda
from  keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau 
from  keras.optimizers import RMSprop, Adam
from  keras.preprocessing.image import load_img, ImageDataGenerator
from  keras.utils import multi_gpu_model
from IPython.display import Image 

import keras_metrics
import pandas
import ast
import numpy as np
import matplotlib.patches as patches 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw,ImageFont
import os
import sys
import subprocess
#device_lib.list_local_devices()


# In[7]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras.backend as K
import numpy as np
import random


# In[4]:


def yolf_loss(y_true, y_pred):
    #true e pred sono [32][245]
    
    #tensorflow.print("tensors:", y_pred)
    #tensorflow.print("batch_size:", tensorflow.shape(y_pred)[0])
    
    #y_pred = tensorflow.reshape(y_pred, [tensorflow.shape(y_pred)[0], 7, 7, 5])
    #y_true = tensorflow.reshape(y_true, [tensorflow.shape(y_pred)[0], 7, 7, 5])


    b_p_pred = y_pred[0]
    b_x_pred = y_pred[1]
    b_y_pred = y_pred[2]
    b_w_pred = y_pred[3]
    b_h_pred = y_pred[4]


    b_p = y_true[0]
    b_x = y_true[1]
    b_y = y_true[2]
    b_w = y_true[3]
    b_h = y_true[4]

    #print(b_xy_pred.get_shape(),b_xy.get_shape())
    #print(b_wh_pred.get_shape(),b_wh.get_shape())
    #indicator_coord = K.expand_dims(y_true[ 3], axis=-1) * 1.0
    loss_p =K.sum(K.square(b_p - b_p_pred), axis=-1)
    loss_xy = K.sum(b_p * (K.square(b_x - b_x_pred) + K.square(b_y - b_y_pred)), axis=-1)# * indicator_coord)#, axis=[1,2,3,4])
    
    b_w = K.pow(b_w, 0.5)
    b_h = K.pow(b_h, 0.5)
    b_w_pred = K.pow(b_w_pred, 0.5)
    b_h_pred = K.pow(b_h_pred, 0.5)
    
    loss_wh = K.sum(
        b_p * 
        (
            (K.square(b_w - b_w_pred)) + 
            (K.square(b_h - b_h_pred))
        ), axis=-1)# * indicator_coord)#, axis=[1,2,3,4])

    #tensorflow.print("loss_p:", loss_p)
    #tensorflow.print("loss_xy:", loss_xy)
    #tensorflow.print("loss_wh:", loss_wh)

    #print(K.cast(loss_p, dtype="float32"))
    #print(K.cast(loss_xy, dtype="float32"))
    #print(loss_wh)
    #tensorflow.print("Loss:", ( K.cast(loss_p, dtype="float32") + loss_wh +  K.cast(loss_xy, dtype="float32") )/3)
    return (K.cast(loss_p, dtype="float32") + loss_wh +  K.cast(loss_xy, dtype="float32"))/3


# In[5]:


BATCH_SIZE = 128
IMG_SIZE = 224


# In[6]:


train_df = pandas.read_csv("/data01/ML/dataset/FACE_CLASSIFIER/train2.csv")
valid_df = pandas.read_csv("/data01/ML/dataset/FACE_CLASSIFIER/val2.csv")
test_df = pandas.read_csv("/data01/ML/dataset/FACE_CLASSIFIER/test2.csv")


# In[7]:


def convert_paths(path_string):
    temp = path_string.replace("\\", "/")  
    return "/data01/ML/" + temp.split("/",1)[1]


# In[8]:


train_df["image_path"] = train_df["image_path"].apply(convert_paths)
valid_df["image_path"] = valid_df["image_path"].apply(convert_paths)
test_df["image_path"] = test_df["image_path"].apply(convert_paths)


# In[9]:


print(train_df.iloc[0]["image_path"])


# In[10]:


train_df.head(3)


# In[11]:


train_datagen = ImageDataGenerator(
    rescale=1./255)
    #preprocessing_function = custom_preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=None,
        x_col="image_path",
        y_col=["p","x","y","w","h"],
        class_mode="raw",
        shuffle=True,
        color_mode = 'grayscale',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE)

valid_generator = train_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=None,
        x_col="image_path",
        y_col=["p","x","y","w","h"],
        class_mode="raw",
        color_mode = 'grayscale',
        shuffle=True,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE)


# In[12]:


step_size_train = train_generator.samples/train_generator.batch_size
step_size_valid = valid_generator.samples/valid_generator.batch_size


# In[13]:


inp = Input(shape=(IMG_SIZE,IMG_SIZE,1))
darknetv1 = (Conv2D(64,3, strides=(1,1), padding = "same"))(inp)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (MaxPooling2D(pool_size=(2, 2)))(darknetv1)
darknetv1 = (Conv2D(192,3, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (MaxPooling2D(pool_size=(2, 2)))(darknetv1)

darknetv1 = (Conv2D(128,1, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (Conv2D(256,3, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (Conv2D(256,1, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (Conv2D(512,3, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (MaxPooling2D(pool_size=(2, 2)))(darknetv1)

darknetv1 = (Conv2D(256,1, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (Conv2D(512,3, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (Conv2D(256,1, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (Conv2D(512,3, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (Conv2D(256,1, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (Conv2D(512,3, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (Conv2D(256,1, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (Conv2D(512,3, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (Conv2D(512,1, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)

darknetv1 = (Conv2D(1024,3, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (MaxPooling2D(pool_size=(2, 2)))(darknetv1)

darknetv1 = (Conv2D(512,1, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (Conv2D(1024,3, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (Conv2D(512,1, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
darknetv1 = (LeakyReLU(alpha=0.1))(darknetv1)
darknetv1 = (Conv2D(1024,3, strides=(1,1), padding = "same"))(darknetv1)
darknetv1 = (BatchNormalization())(darknetv1)
mid_model = (LeakyReLU(alpha=0.1))(darknetv1)

classifier = (GlobalAveragePooling2D())(mid_model)
classifier = (Dense(512, activation = "relu"))(classifier)
classifier = (Dense(512, activation = "relu"))(classifier)
classifier = (Dense(1, activation = "sigmoid"))(classifier)


# In[14]:


model_temp = Model(inputs=[inp], outputs=classifier)


# In[15]:


model_temp.load_weights("face_classifier_BN_GRAYSCALE.h5") 


# In[16]:


for layer in model_temp.layers:
    layer.trainable=False


# # Side tuning time

# In[17]:


bounding_boxer = (Conv2D(1024,3, strides=(1,1), padding = "same"))(mid_model)
bounding_boxer = (BatchNormalization())(bounding_boxer)
bounding_boxer = (LeakyReLU(alpha=0.1))(bounding_boxer)
bounding_boxer = (Conv2D(1024,3, strides=(1,1), padding = "same"))(bounding_boxer)
bounding_boxer = (BatchNormalization())(bounding_boxer)
bounding_boxer = (LeakyReLU(alpha=0.1))(bounding_boxer)
bounding_boxer = (Conv2D(1024,3, strides=(1,1), padding = "same"))(bounding_boxer)
bounding_boxer = (BatchNormalization())(bounding_boxer)
bounding_boxer = (LeakyReLU(alpha=0.1))(bounding_boxer)

bounding_boxer = (Flatten())(bounding_boxer)

bounding_boxer = (Dense(4096))(bounding_boxer)
bounding_boxer = (LeakyReLU(alpha=0.1))(bounding_boxer)
bounding_boxer = (Dense(2048))(bounding_boxer)
bounding_boxer = (LeakyReLU(alpha=0.1))(bounding_boxer)
bounding_boxer = (Dense(4, activation = "relu"))(bounding_boxer)


# In[18]:


model = Concatenate()([classifier, bounding_boxer])


# In[19]:


model = Model(inputs=[inp], outputs=model)


# In[20]:


model.summary()


# In[21]:


model = multi_gpu_model(model,gpus=2)


# In[22]:


model.compile(optimizer=Adam(lr = 1e-4), loss=yolf_loss, metrics=[yolf_loss])


# darknetv1.evaluate_generator(valid_generator, steps=step_size_valid, verbose = 1)

# In[23]:


#earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
mcp_save = ModelCheckpoint('/data01/ML/standford_darknet.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min')

history = model.fit_generator(generator=train_generator, epochs=20, steps_per_epoch=step_size_train, validation_data=valid_generator, validation_steps=step_size_valid, verbose=1, callbacks=[mcp_save, reduce_lr_loss])


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training accuracy')
plt.plot(val_acc, label='Validation accuracy')
plt.legend(loc='lower right')
plt.ylabel('accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# # Loading model

# In[ ]:


darknetv2 = Sequential()
darknetv2.add(InputLayer(input_shape=(IMG_SIZE,IMG_SIZE,3)))
darknetv2.add(Conv2D(64,3, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(MaxPooling2D(pool_size=(2, 2)))
darknetv2.add(Conv2D(192,3, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(MaxPooling2D(pool_size=(2, 2)))

darknetv2.add(Conv2D(128,1, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(256,3, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(256,1, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(512,3, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(MaxPooling2D(pool_size=(2, 2)))

darknetv2.add(Conv2D(256,1, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(512,3, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(256,1, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(512,3, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(256,1, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(512,3, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(256,1, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(512,3, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(512,1, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(1024,3, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(MaxPooling2D(pool_size=(2, 2)))

darknetv2.add(Conv2D(512,1, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(1024,3, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(512,1, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(1024,3, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(1024,3, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(1024,3, strides=(2,2), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(1024,3, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(1024,3, strides=(1,1), padding = "same"))
darknetv2.add(BatchNormalization())
darknetv2.add(LeakyReLU(alpha=0.1))
darknetv2.add(Conv2D(5,1, strides=(1,1), padding = "same", activation="relu"))
darknetv2.add(Flatten())


# In[ ]:


darknetv2.load_weights("darknet_ev.h5") 
darknetv2.compile(optimizer=Adam(lr = 1e-4), loss="binary_crossentropy", metrics=["accuracy"])


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=validation_set,
        x_col="image_id",
        y_col=train_df.columns[1:],
        class_mode="raw",
        shuffle=True,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE)
STEP_SIZE_TEST = test_generator.n / test_generator.batch_size 


#CHANGE PARALLEL MODEL
pred=darknetv2.predict_generator(test_generator,  steps=STEP_SIZE_TEST,  verbose=1)


# In[ ]:


print(pred[0].reshape(49,5))


# In[ ]:


print(test_generator.labels[0].reshape(49,5))


# In[ ]:


count = 0
max_row = -1
list_max = []
filename= ""
for el in zip(pred,test_generator.labels,test_generator.filenames):
    count = count +1
    for row in el[1].reshape(49,5):
        if row[1] > max_row:
            max_row =  row[1]
            list_max = []
            list_max.append(row)
            filename = el[2]
    try:
        list_max = [item for sublist in list_max for item in sublist]
        img = Image.open(validation_set+"\\"+filename)
        img1 = ImageDraw.Draw(img)
        x1,y1 = (list_max[1]-(list_max[3]/2)),(list_max[2]-(list_max[4]/2))
        x4,y4= (list_max[1]+(list_max[3]/2)),(list_max[2]+(list_max[4]/2))
        img1.rectangle([(x1,y1),(x4,y4)], outline ="red") 
        img.save("./dataset/NUOVO/results/output"+ str(count) + ".jpg")
        #img.show() 
    except:
        pass
    if count == 1000:
        pass


# In[ ]:





# In[ ]:




