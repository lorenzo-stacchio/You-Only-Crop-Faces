import cv2
import numpy as np
from PIL import Image
from keras.layers import Dense,Conv2D, GlobalAveragePooling2D, Reshape
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.models import Model


def gen_model():
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
        return parallel_model

#Load the saved model
model = gen_model()
model.load_weights('weights/single_face_total.h5')
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
WIDTH,HEIGTH = 224,224

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
        face, box = prediction[0][0], prediction[1][0]
        print(face)
        print(box)

        #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
        if face >= 0.5:
                #rescale box to original size
                x_1, x_2 = int(box[0]*rescale_factor_width),int(box[1]*rescale_factor_heigth)
                x_3, x_4 = int(box[0]*rescale_factor_width+box[2]*rescale_factor_width), int(box[1]*rescale_factor_heigth+box[3]*rescale_factor_heigth)
                color = (255, 0, 0)
                thickness = 2
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.rectangle(frame, (x_1,x_2),(x_3,x_4), color, thickness)

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()