#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
from keras.layers import Dense,Conv2D, GlobalAveragePooling2D, Reshape
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model,model_from_json
import os

def gen_model():
    if os.path.exists("model.json"):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        parallel_model = model_from_json(loaded_model_json)
        # load weights into new model
        parallel_model.load_weights("weights/single_face_total.h5")
        return parallel_model
    else:
        model2 = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), alpha=1.4)
        boxer_branch = model2.layers[-1].output
        boxer_branch = Conv2D(4, 7, strides=(1, 1), name="detector_conv_2", activation="relu")(boxer_branch)
        boxer_branch = Reshape((4,), name="coords")(boxer_branch)
        model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), alpha=1.4)
        model.trainable = True
        w = 0
        for layer in model.layers[:len(model.layers) - w]:
            layer.trainable = True
            layer.name = layer.name + "_class"
        last = model.layers[-1].output
        classifier_branch = GlobalAveragePooling2D(name="glob_average_mine")(last)
        classifier_branch = Dense(1, activation="sigmoid", name="classes")(classifier_branch)
        parallel_model = Model([model.input, model2.input], outputs=[classifier_branch, boxer_branch])
        parallel_model.load_weights("weights/single_face_total.h5")
        model_json = parallel_model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        return parallel_model

