#!/usr/bin/python
# -*- coding: utf-8 -*-
from os import listdir, path, curdir
from os.path import isfile, join
import re
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import random
import numpy as np

root = ".\\dataset\\FACE_CLASSIFIER\\"
root_dat = ".\\dataset"
bbox_path = "bbox_original.csv"
bb_new_file_name = "parsed_list_bbox_celeba.csv"

face_image_path = ".\\dataset\\FACE_CLASSIFIER\\CelebA2\\CelebA\\Img\\img_celeba\\img_celeba\\img_celeba"
objects_image_path = ".\\dataset\\FACE_CLASSIFIER\\256_ObjectCategories"

output_final_scaled_list = [".\\dataset\\NUOVO\\train\\output_final.csv",".\\dataset\\NUOVO\\val\\output_final.csv",".\\dataset\\NUOVO\\test\\output_final.csv"]

def prepare_bb_txt_file():
    for el in output_final_scaled_list:
        with open(el, "r") as fr:
            with open(root_dat + "\\" + bb_new_file_name, "a") as fw:
                for line in fr:
                    tmp = line.split(",",1)[1]
                    fw.writelines(tmp)


def get_faces(balanced, face_image_path, bbox_csv_path):
    df = pd.read_csv(bbox_csv_path)
    df_temp = df.sample(n=balanced)
    count = 0
    rows = []
    for temp in df_temp.iterrows():
        count = count + 1
        if count % 100 == 0:
            print(count)
        x_center, y_center = temp[1]["x_1"] + int(temp[1]["width"]/2),temp[1]["y_1"] + int(temp[1]["height"]/2)
        #(temp[1]["image_id"])
        rows.append((face_image_path + "\\" + str(temp[1]["image_id"]), np.array([1, x_center, y_center, temp[1]["width"], temp[1]["height"]])))
    return rows


def get_balanced_data_from_classes(objects_main_folder):
    rows = []
    dirs = [f for f in listdir(objects_main_folder) if not isfile(join(objects_main_folder, f))]
    for dir_ in dirs:
        files = [f for f in listdir(objects_main_folder + "\\" + dir_) if
                 isfile(join(objects_main_folder + "\\" + dir_, f))]
        for f in files:
            rows.append((objects_main_folder + "\\" + dir_ + "\\" + f, [0, 0, 0, 0, 0]))
    return rows


def get_min_between_celeba_and_objects(objects_main_folder, celeba_images_folder):
    dirs = [f for f in listdir(objects_main_folder) if not isfile(join(objects_main_folder, f))]
    count_objects = 0
    count_faces = 0
    for dir_ in dirs:
        files = [f for f in listdir(objects_main_folder + "\\" + dir_) if
                 isfile(join(objects_main_folder + "\\" + dir_, f))]
        count_objects = count_objects + len(files)
    files_Faces = [f for f in listdir(celeba_images_folder) if
                   isfile(join(celeba_images_folder, f))]
    count_faces = count_faces + len(files_Faces)
    print(count_faces, count_objects)
    return count_faces if count_faces < count_objects else count_objects


def write_csv(category, non_faces, faces):
    with open(root + "\\" + category + ".csv", mode='w', newline='') as file:
        file = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file.writerow(['image_path','p','x','y','w','h' ])
        for el in non_faces:
            file.writerow([el[0], el[1][0],el[1][1],el[1][2],el[1][3],el[1][4]])
        for el in faces:
            file.writerow([el[0], el[1][0],el[1][1],el[1][2],el[1][3],el[1][4]])



if __name__ == '__main__':
    #prepare_bb_txt_file()

    min = get_min_between_celeba_and_objects(objects_image_path, face_image_path)
    non_faces = get_balanced_data_from_classes(objects_image_path)
    # TODO: creare un insieme a partire dalle robe random al 90% e poi fare cose vere
    non_faces_train_val = non_faces[:int(len(non_faces) * 0.90)]
    non_faces_train, non_faces_val = non_faces_train_val[:int(len(non_faces_train_val) * 0.90)], non_faces_train_val[
                                                                                                 int(len(
                                                                                                     non_faces_train_val) * 0.90): len(
                                                                                                     non_faces_train_val)]
    non_faces_test = non_faces[int(len(non_faces) * 0.90):len(non_faces)]
    print(len(non_faces), len(non_faces_test), len(non_faces_train), len(non_faces_val))

    faces = get_faces(len(non_faces), face_image_path, root_dat + "\\" + bb_new_file_name)
    faces_train_val = faces[:int(len(faces) * 0.90)]
    faces_train, faces_val = faces_train_val[:int(len(faces_train_val) * 0.90)], faces_train_val[
                                                                                 int(len(faces_train_val) * 0.90): len(
                                                                                     faces_train_val)]
    faces_test = faces[int(len(faces) * 0.90):len(faces)]
    print(len(faces), len(faces_test), len(faces_train), len(faces_val))

    for tuple in [("train2", non_faces_train, faces_train), ("val2", non_faces_val, faces_val),
                  ("test2", non_faces_test, faces_test)]:
        print("Writing:", tuple[0])
        write_csv(tuple[0], tuple[1], tuple[2])
    # with open('C:\\Users\\Lorenzo\\Desktop\\Università\\Machine_learning_project\\dataset\\FACE_CLASSIFIER\\dataset.csv' , mode='w', newline='') as file:
    # file = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # file.writerow(['image_path', 'face'])
    # for el in non_faces:
    #     file.writerow([el[0], el[1]])
    # for el in faces:
    #     file.writerow([el[0], el[1]])
