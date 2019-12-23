from PIL import Image, ImageDraw
import pandas as pd
import ast
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
from os.path import isfile, join
import scipy.io
import os
import csv

if __name__ == '__main__':

    image_folder_after_step2 = "//media//Disco_Secondario//datasets//face_segmentation//test_set//resized//0_Parade_marchingband_1_356.jpg"
    image_csv_step2 = pd.read_csv("//media//Disco_Secondario//datasets//face_segmentation//test_set//output.csv")
    # Setting params
    im = Image.open(image_folder_after_step2, 'r')
    width, height = im.size
    print(width)
    print(height)
    new_width, new_height = 300, 300
    scaling_factor_width, scaling_factor_height = width/new_width, height/new_height
    print(scaling_factor_width)
    print(scaling_factor_height)
    # Get labels info about the image
    label = image_csv_step2.loc[image_csv_step2["filename"] == "0_Parade_marchingband_1_356.jpg"]["labels"]
    label_fixed = ast.literal_eval(label.iloc[0]) # to threat the string as tuple
    label_x, label_y, label_width, label_height = label_fixed[0], label_fixed[1], label_fixed[2], label_fixed[3]
    print(label_x)
    print(label_y)
    print(label_width)
    print(label_height)
    # try to resize
    new_label_x, new_label_y, new_label_width, new_label_height = round(label_x/scaling_factor_width), round(label_y/scaling_factor_height), round(label_width/scaling_factor_width), round(label_height/scaling_factor_height)
    im = im.resize((new_width, new_height))
    draw = ImageDraw.Draw(im)
    draw.rectangle(((new_label_x, new_label_y), (new_label_x+new_label_width, new_label_y+new_label_height)), outline="Red")
    im.save("test.jpg")
    img = mpimg.imread('test.jpg')
    imgplot = plt.imshow(img)
    plt.show()