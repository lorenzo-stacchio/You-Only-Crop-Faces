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
face_image_path = "/data01/ML/dataset/FACE_CLASSIFIER/CelebA2/CelebA/Img/img_celeba/img_celeba/img_celeba"
objects_image_path = "/data01/ML/dataset/FACE_CLASSIFIER/256_ObjectCategories_modified"
face_image_path2 = "/data01/ML/dataset/FACE_CLASSIFIER/Borderline"



def get_faces(balanced, face_image_path, face_image_path2):
    filenames = []
    classification = []
    files = [f for f in listdir(face_image_path) if
             isfile(join(face_image_path, f))]
    files = random.sample(files, balanced-600)
    for f in files:
        filenames.append(face_image_path + "/" + f)
        classification.append(1)
    files2 = [f for f in listdir(face_image_path2) if
             isfile(join(face_image_path2, f))]
    files2 = random.sample(files2, 600)
    for f in files2:
        filenames.append(face_image_path2 + "/" + f)
        classification.append(1)
    return filenames, classification


def get_balanced_data_from_classes(min,objects_main_folder):
    filenames = []
    classification = []
    dirs = [f for f in listdir(objects_main_folder) if not isfile(join(objects_main_folder, f))]
    for dir_ in dirs:
        files = [f for f in listdir(objects_main_folder + "/" + dir_) if
                 isfile(join(objects_main_folder + "/" + dir_, f))]
        files = random.sample(files, min)
        for f in files:
            filenames.append(objects_main_folder+"/"+dir_+"/"+f)
            classification.append(0)
    return filenames, classification

def get_not_balanced_data_from_classes(objects_main_folder):
    filenames = []
    classification = []
    dirs = [f for f in listdir(objects_main_folder) if not isfile(join(objects_main_folder, f))]
    for dir_ in dirs:
        files = [f for f in listdir(objects_main_folder + "/" + dir_) if
                 isfile(join(objects_main_folder + "/" + dir_, f))]
        for f in files:
            filenames.append(objects_main_folder + "/" + dir_ + "/" + f)
            classification.append(0)
    return filenames, classification

def count_min_in_objects_types(objects_main_folder):
    dirs = [f for f in listdir(objects_main_folder) if not isfile(join(objects_main_folder, f))]
    min_f_numbers = 1000000
    max_f_numbers = 0
    for dir_ in dirs:
        files = [f for f in listdir(objects_main_folder+"/"+dir_) if isfile(join(objects_main_folder+"/"+dir_, f))]
        if len(files)< min_f_numbers:
            min_f_numbers = len(files)
        if len(files) > max_f_numbers:
            max_f_numbers = len(files)
    return min_f_numbers

def get_min_between_celeba_and_objects(objects_main_folder, celeba_images_folder):
    dirs = [f for f in listdir(objects_main_folder) if not isfile(join(objects_main_folder, f))]
    count_objects = 0
    count_faces = 0
    for dir_ in dirs:
        files = [f for f in listdir(objects_main_folder + "/" + dir_) if
                 isfile(join(objects_main_folder + "/" + dir_, f))]
        count_objects = count_objects + len(files)
    files_Faces = [f for f in listdir(celeba_images_folder) if
                   isfile(join(celeba_images_folder, f))]
    count_faces = count_faces + len(files_Faces)
    print(count_faces, count_objects)
    return count_faces if count_faces < count_objects else count_objects

def write_csv(category, non_faces, faces):
    with open("./dataset/" + category+".csv" , mode='w', newline='') as file:
        file = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file.writerow(['image_path', 'face'])
        for el in non_faces:
            file.writerow([el[0], el[1]])
        for el in faces:
            file.writerow([el[0], el[1]])


if __name__ == '__main__':
    #min = count_min_in_objects_types(objects_image_path)
    #non_faces = get_balanced_data_from_classes(min,objects_image_path)
    min = get_min_between_celeba_and_objects(objects_image_path, face_image_path)
    non_faces = get_not_balanced_data_from_classes(objects_image_path)
    print(len(non_faces[0]))
    #print(non_faces[1])
    non_faces_x_train, non_faces_x_test, non_faces_y_train, non_faces_y_test = train_test_split(non_faces[0], non_faces[1], test_size = 0.15,shuffle=True)
    non_faces_x_train, non_faces_x_val, non_faces_y_train, non_faces_y_val = train_test_split(non_faces_x_train, non_faces_y_train, test_size= 0.15,shuffle=True)
    print(len(non_faces_x_train),len(non_faces_x_val), len(non_faces_x_test))
    #for el in zip(non_faces_x_train, non_faces_y_train):
    #    print(el)

    faces = get_faces(len(non_faces[0]), face_image_path, face_image_path2)
    print(len(faces[0]))
    #print(faces[1])
    faces_x_train, faces_x_test, faces_y_train, faces_y_test = train_test_split(faces[0],faces[1],test_size=0.15,shuffle=True)
    faces_x_train, faces_x_val, faces_y_train, faces_y_val = train_test_split(faces_x_train,faces_y_train,
                                                                                              test_size=0.15,shuffle=True)
    print(len(non_faces_x_train), len(non_faces_x_val), len(non_faces_x_test))
    #Define true dataset
    non_faces_train, faces_train = zip(non_faces_x_train,non_faces_y_train),zip(faces_x_train,faces_y_train)
    non_faces_val, faces_val = zip(non_faces_x_val,non_faces_y_val),zip(faces_x_val,faces_y_val)
    non_faces_test, faces_test = zip(non_faces_x_test,non_faces_y_test),zip(faces_x_test,faces_y_test)

    for tuple in [("train3", non_faces_train,faces_train),("val3",non_faces_val,faces_val),("test3", non_faces_test,faces_test)]:
         print("Writing:", tuple[0])
         write_csv(tuple[0],tuple[1],tuple[2])
    # with open('C:/Users/Lorenzo/Desktop/Università/Machine_learning_project/dataset/FACE_CLASSIFIER/dataset.csv' , mode='w', newline='') as file:
    #     file = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     file.writerow(['image_path', 'face'])
    #     for el in non_faces:
    #         file.writerow([el[0], el[1]])
    #     for el in faces:
    #         file.writerow([el[0], el[1]])
