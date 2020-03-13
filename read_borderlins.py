#!/usr/bin/python
# -*- coding: utf-8 -*-
from os import listdir,path,curdir
from os.path import isfile, join
import re
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import random

root = "/data01/ML/dataset/FACE_CLASSIFIER"
#face_image_path = "/data01/ML/dataset/FACE_CLASSIFIER/CelebA2/CelebA/Img/img_celeba/img_celeba/img_celeba"
objects_image_path = "/data01/ML/dataset/FACE_CLASSIFIER/256_ObjectCategories_modified"
face_image_path2 = "/data01/ML/dataset/FACE_CLASSIFIER/Borderline"



def get_faces(face_image_path):
    filenames = []
    classification = []
    files = [f for f in listdir(face_image_path) if
             isfile(join(face_image_path, f))]
    for f in files:
        filenames.append(face_image_path + "/" + f)
        classification.append(1)
    return filenames, classification


def write_csv(category,faces):
    with open("./dataset/" + category+".csv" , mode='w', newline='') as file:
        file = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file.writerow(['image_path', 'face'])
        for el in zip(faces[0],faces[1]):
            file.writerow([el[0], el[1]])


if __name__ == '__main__':
    faces = get_faces(face_image_path2)
    write_csv("test_borderline", faces)