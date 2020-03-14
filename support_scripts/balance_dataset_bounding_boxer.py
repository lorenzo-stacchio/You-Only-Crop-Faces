#!/usr/bin/python
# -*- coding: utf-8 -*-
from os import listdir, path, curdir
from os.path import isfile, join
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import numpy as np
from PIL import Image

root = "./dataset/"
root_dat = "./dataset"
bbox_path = "/data01/ML/dataset/FACE_CLASSIFIER/CelebA2/CelebA/Anno/list_bbox_celeba.csv"
bb_new_file_name = "./dataset/parsed_list_bbox_celeba.csv"

face_image_path = "/data01/ML/dataset/FACE_CLASSIFIER/CelebA2/CelebA/Img/img_celeba/img_celeba/img_celeba"
objects_image_path = "/data01/ML/dataset/FACE_CLASSIFIER/256_ObjectCategories"

output_final_scaled_list = ["./dataset/NUOVO/train/output_final.csv","./dataset/NUOVO/val/output_final.csv","./dataset/NUOVO/test/output_final.csv"]

IMG_SIZE = 224

def prepare_bb_txt_file():
    with open(bbox_path, "r") as fr:
        with open(root_dat + "/" + bb_new_file_name, "w") as fw:
            for line in fr:
                tmp = re.sub("\s+", ",", line)
                tmp = tmp[:len(tmp)-1] + "\n"
                #print(tmp)
                fw.writelines(tmp)


def get_faces(balanced, face_image_path, bbox_csv_path):
    df = pd.read_csv(bbox_csv_path)
    df_temp = df.sample(n=balanced)
    count = 0
    filenames = []
    classification = []
    for temp in df_temp.iterrows():
        count = count + 1
        if count % 1000 == 0:
            print(count)
        image = Image.open(face_image_path+"/"+temp[1]["image_id"])
        w,h = image.size
        image.close()
        ratio_w = 0
        ratio_h = 0
        ratio_w, ratio_h = round(IMG_SIZE/w,2),round(IMG_SIZE/h,2)
        x_1, = int(temp[1]["x_1"]*ratio_w),
        y_1 = int(temp[1]["y_1"]*ratio_h)
        w_bb = int(temp[1]["width"]*ratio_w)
        h_bb = int(temp[1]["height"]*ratio_h)
        x_2, y_2 = x_1+w_bb, y_1+h_bb
        #x_center, y_center = temp[1]["x_1"] + int(temp[1]["width"]/2),temp[1]["y_1"] + int(temp[1]["height"]/2)
        filenames.append(face_image_path + "/" + str(temp[1]["image_id"]))
        classification.append(np.array([x_1,y_1, w_bb, h_bb]))
    return filenames, classification


def get_balanced_data_from_classes(objects_main_folder):
    filenames = []
    classification = []
    dirs = [f for f in listdir(objects_main_folder) if not isfile(join(objects_main_folder, f))]
    for dir_ in dirs:
        files = [f for f in listdir(objects_main_folder + "/" + dir_) if
                 isfile(join(objects_main_folder + "/" + dir_, f))]
        for f in files:
            filenames.append(objects_main_folder + "/" + dir_ + "/" + f)
            classification.append([0, 0, 0, 0])
    return filenames, classification


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



def write_csv(category, faces):
    with open(root + "/" + category + ".csv", mode='w', newline='') as file:
        file = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file.writerow(['image_path','x','y','x_2','y_2'])
        #for el in non_faces:
        #    file.writerow([el[0], el[1][0], el[1][1], el[1][2], el[1][3]])
            #file.writerow([el[0], el[1][0],el[1][1],el[1][2],el[1][3],el[1][4]])
        for el in faces:
            #file.writerow([el[0], el[1][0],el[1][1],el[1][2],el[1][3],el[1][4]])
            file.writerow([el[0], el[1][0], el[1][1], el[1][2], el[1][3]])



if __name__ == '__main__':
    #prepare_bb_txt_file()
    min = get_min_between_celeba_and_objects(objects_image_path, face_image_path)
    # non_faces = get_balanced_data_from_classes(objects_image_path)
    #
    # print(len(non_faces[0]))
    # # print(non_faces[1])
    # non_faces_x_train, non_faces_x_test, non_faces_y_train, non_faces_y_test = train_test_split(non_faces[0],
    #                                                                                             non_faces[1],
    #                                                                                             test_size=0.15,
    #                                                                                             shuffle=True)
    # non_faces_x_train, non_faces_x_val, non_faces_y_train, non_faces_y_val = train_test_split(non_faces_x_train,
    #                                                                                           non_faces_y_train,
    #                                                                                           test_size=0.15,
    #                                                                                           shuffle=True)
    # print(len(non_faces_x_train), len(non_faces_x_val), len(non_faces_x_test))
    # # for el in zip(non_faces_x_train, non_faces_y_train):
    # #    print(el)

    #faces = get_faces(len(non_faces[0]), face_image_path, bb_new_file_name)
    #202599
    faces = get_faces(202599, face_image_path, bb_new_file_name)
    print(len(faces[0]))
    # print(faces[1])
    faces_x_train, faces_x_test, faces_y_train, faces_y_test = train_test_split(faces[0], faces[1], test_size=0.15,
                                                                                shuffle=True)
    faces_x_train, faces_x_val, faces_y_train, faces_y_val = train_test_split(faces_x_train, faces_y_train,
                                                                              test_size=0.15, shuffle=True)
    print(len(faces_x_train), len(faces_x_val), len(faces_x_test))
    # Define true dataset
    #non_faces_train, faces_train = zip(non_faces_x_train, non_faces_y_train), zip(faces_x_train, faces_y_train)
    #non_faces_val, faces_val = zip(non_faces_x_val, non_faces_y_val), zip(faces_x_val, faces_y_val)
    #non_faces_test, faces_test = zip(non_faces_x_test, non_faces_y_test), zip(faces_x_test, faces_y_test)

    for tuple in [("train2", zip(faces_x_train,faces_y_train)), ("val2", zip(faces_x_val,faces_y_val)),
                  ("test2", zip(faces_x_test,faces_y_test))]:
        print("Writing:", tuple[0])
        write_csv(tuple[0], tuple[1])

    #for tuple in [("train2", non_faces_train, faces_train), ("val2", non_faces_val, faces_val),
        #              ("test2", non_faces_test, faces_test)]:
        #print("Writing:", tuple[0])
        #write_csv(tuple[0], tuple[1], tuple[2])

    # for el in faces:
    #     file.writerow([el[0], el[1]])
