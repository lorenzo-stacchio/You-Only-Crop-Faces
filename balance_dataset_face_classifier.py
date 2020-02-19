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

root = ".\\dataset\\FACE_CLASSIFIER\\"
face_image_path = ".\\dataset\\FACE_CLASSIFIER\\CelebA2\\CelebA\\Img\\img_celeba\\img_celeba\\img_celeba"
objects_image_path = ".\\dataset\\FACE_CLASSIFIER\\256_ObjectCategories"



def get_faces(balanced, face_image_path):
    rows = []
    files = [f for f in listdir(face_image_path) if
             isfile(join(face_image_path, f))]
    files = random.sample(files, balanced)
    for f in files:
        rows.append((face_image_path+"\\"+f, 1))
    return rows


def get_balanced_data_from_classes(min,objects_main_folder):
    rows = []
    dirs = [f for f in listdir(objects_main_folder) if not isfile(join(objects_main_folder, f))]
    for dir_ in dirs:
        files = [f for f in listdir(objects_main_folder + "\\" + dir_) if
                 isfile(join(objects_main_folder + "\\" + dir_, f))]
        files = random.sample(files, min)
        for f in files:
            rows.append((objects_main_folder+"\\"+dir_+"\\"+f, 0))
    return rows


def count_min_in_objects_types(objects_main_folder):
    dirs = [f for f in listdir(objects_main_folder) if not isfile(join(objects_main_folder, f))]
    min_f_numbers = 1000000
    max_f_numbers = 0
    for dir_ in dirs:
        files = [f for f in listdir(objects_main_folder+"\\"+dir_) if isfile(join(objects_main_folder+"\\"+dir_, f))]
        if len(files)< min_f_numbers:
            min_f_numbers = len(files)
        if len(files) > max_f_numbers:
            max_f_numbers = len(files)
    #print(min_f_numbers)
    #print(max_f_numbers)
    #print(dirs)
    return min_f_numbers


def write_csv(category, non_faces, faces):
    with open(root + "\\"+ category+".csv" , mode='w', newline='') as file:
        file = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file.writerow(['image_path', 'face'])
        for el in non_faces:
            file.writerow([el[0], el[1]])
        for el in faces:
            file.writerow([el[0], el[1]])


if __name__ == '__main__':
    min = count_min_in_objects_types(objects_image_path)
    non_faces = get_balanced_data_from_classes(min,objects_image_path)
    #TODO: creare un insieme a partire dalle robe random al 90% e poi fare cose vere
    non_faces_train_val = non_faces[:int(len(non_faces) * 0.90)]
    non_faces_train, non_faces_val = non_faces_train_val[:int(len(non_faces_train_val) * 0.90)], non_faces_train_val[int(len(non_faces_train_val) * 0.90): len(non_faces_train_val)]
    non_faces_test = non_faces[int(len(non_faces) * 0.90):len(non_faces)]
    print(len(non_faces), len(non_faces_test), len(non_faces_train), len(non_faces_val))

    faces = get_faces(len(non_faces), face_image_path)
    faces_train_val = faces[:int(len(faces) * 0.90)]
    faces_train, faces_val = faces_train_val[:int(len(faces_train_val) * 0.90)], faces_train_val[int(len(faces_train_val) * 0.90): len(faces_train_val)]
    faces_test = faces[int(len(faces) * 0.90):len(faces)]
    print(len(faces),len(faces_test), len(faces_train), len(faces_val))


    for tuple in [("train", non_faces_train,faces_train),("val",non_faces_val,faces_val),("test", non_faces_test,faces_test)]:
        print("Writing:", tuple[0])
        write_csv(tuple[0],tuple[1],tuple[2])
    # with open('C:\\Users\\Lorenzo\\Desktop\\Università\\Machine_learning_project\\dataset\\FACE_CLASSIFIER\\dataset.csv' , mode='w', newline='') as file:
    #     file = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     file.writerow(['image_path', 'face'])
    #     for el in non_faces:
    #         file.writerow([el[0], el[1]])
    #     for el in faces:
    #         file.writerow([el[0], el[1]])
