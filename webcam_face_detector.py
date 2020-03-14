#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
from PIL import Image
from keras.applications.mobilenet_v2 import preprocess_input
import model

confidence = 0.5
WIDTH,HEIGTH = 224,224

if __name__ == '__main__':
        #Load the saved model
        model = model.gen_model()
        video = cv2.VideoCapture(0)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)

        while True:
                _, frame = video.read()
                #Convert the captured frame into RGB
                im = Image.fromarray(frame, 'RGB')
                origin_width,origin_heigth = im.size
                rescale_factor_width =origin_width/WIDTH
                rescale_factor_heigth = origin_heigth/ HEIGTH
                #Resizing into 128x128 because we trained the model with this image size.
                im = im.resize((WIDTH,HEIGTH))
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
                        color = (255, 0, 0)
                        thickness = 2
                        frame = cv2.rectangle(frame, (x_1,x_2),(x_3,x_4), color, thickness)
                cv2.imshow("Capturing", frame)
                key=cv2.waitKey(1)
                if key == ord('q'):
                        break
        video.release()
        cv2.destroyAllWindows()