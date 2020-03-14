#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
from PIL import Image, ImageDraw
from keras.applications.mobilenet_v2 import preprocess_input
import sys
import model
import os

confidence = 0.5 #global
WIDTH_model,HEIGTH_model = 224,224

def evaluate_face(model, image_path, output_dir):
        image = Image.open(image_path)
        origin_width, origin_heigth = image.size
        rescale_factor_width =origin_width/WIDTH_model
        rescale_factor_heigth = origin_heigth/ HEIGTH_model
        #Resizing into 128x128 because we trained the model with this image size.
        im = image.resize((WIDTH_model,HEIGTH_model))
        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        #Calling the predict method on model to predict 'me' on the image
        prediction = model.predict([img_array,img_array])
        #print(prediction)
        face, box = round(prediction[0][0][0],2), prediction[1][0]
        print("There is a face? ", "Yes" if face>=confidence else "No")
        print("Box guess: " + str(box))
        #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
        if face >= confidence:
                #rescale box to original size
                x_1, x_2 = int(box[0]*rescale_factor_width),int(box[1]*rescale_factor_heigth)
                x_3, x_4 = int(box[0]*rescale_factor_width+box[2]*rescale_factor_width), int(box[1]*rescale_factor_heigth+box[3]*rescale_factor_heigth)
                img1 = ImageDraw.Draw(image)
                img1.rectangle([(x_1,x_2),(x_3,x_4)], outline="red")
                filename = os.path.basename(image_path).split(".")[0]
                image.save(output_dir + filename + "_evalued.jpg")

if __name__ == '__main__':
        image_path = sys.argv[1]
        output_dir = sys.argv[2]
        # Load the saved model
        model = model.gen_model()
        print("-----Model loaded-----")
        print("-----Start Evaluation-----")
        evaluate_face(model, image_path, output_dir)
        print("-----End evaluation-----")