from PIL import Image
from os import listdir
from os.path import isfile, join
import scipy.io
import os
import csv

max_height = 0

image_folder = 'C:\\Users\\Lorenzo Stacchio\\Downloads\\WIDER_val\\images'
resized_image_folder = 'C:\\Users\\Lorenzo Stacchio\\Desktop\\Machine_learning_project\\dataset\\test_set\\resized'


def resize(file_name):
    global max_height
    im = Image.open(join(image_folder, file_name))
    out = Image.new(im.mode, (im.size[0], max_height))
    out.paste(im)
    out.save(join(resized_image_folder, file_name))


def getsize(file_name):
    global max_height
    im = Image.open(join(image_folder, file_name))
    if im.size[1] > max_height:
        max_height = im.size[1]


def getRect(values, paddingW, paddingH):
    return (values[0] - round(paddingW / 2, 0), values[1] - round(paddingH / 2, 0),
            values[0] + values[2] + round(paddingW / 2, 0), values[1] + values[3] + round(paddingH / 2, 0))


def getArea(values):
    return values[2] * values[3]


if __name__ == "__main__":
    image_path = image_folder
    pictures = [f for f in listdir(image_path) if isfile(join(image_path, f))]

    dict_photo = {}
    new_class_photo = {}

    mat2 = scipy.io.loadmat('C:\\Users\\Lorenzo Stacchio\\Desktop\\Machine_learning_project\\dataset\\test_set'
                            '\\wider_face_val.mat')

    image_path = 'C:\\Users\\Lorenzo Stacchio\\Downloads\\WIDER_val\\images'

    max_area_rect = 0
    WH_max = ()

    if True:
        onlyDir = [f for f in listdir(image_path) if not isfile(join(image_path, f))]
        print(onlyDir)
        list_all = zip(mat2["face_bbx_list"], mat2["file_list"], onlyDir)
        for bbx, f_l, dir_name in list_all:  # bbx is nested, f_l is directly connected to filenames of the folder
            pair_photo_filename = zip(bbx[0], f_l[0])
            for bbx_single_photo, single_photo_filename in pair_photo_filename:
                if len(bbx_single_photo[0]) == 1:
                    dict_photo[single_photo_filename[0][0] + ".jpg"] = bbx_single_photo[0][0]
                    new_class_photo[single_photo_filename[0][0] + ".jpg"] = (
                        bbx_single_photo[0][0][2], bbx_single_photo[0][0][3])
                    if max_area_rect < getArea(bbx_single_photo[0][0]):
                        max_area_rect = getArea(bbx_single_photo[0][0])
                        WH_max = (bbx_single_photo[0][0][2], bbx_single_photo[0][0][3])

        print(len(dict_photo))
        print(WH_max)

        for key, value in dict_photo.items():
            new_class_photo[key] = (
                round((WH_max[0] - value[2]) / 2, 0), round((WH_max[1] - value[3]) / 2, 0), new_class_photo[key][0],
                new_class_photo[key][1])
            dict_photo[key] = getRect(value, WH_max[0] - value[2], WH_max[1] - value[3])

        with open('dataset\\test_set\\output.csv', "w", newline='') as f:
            csv_out = csv.writer(f)
            csv_out.writerow(['filename', 'labels'])
            for l in new_class_photo.items():
                csv_out.writerow(l)

        for root, dirs, files in os.walk(image_path, topdown=False):
            for name in files:
                if name in dict_photo.keys():
                    im = Image.open(join(root, name))
                    im = im.crop(dict_photo[name])
                    out = Image.new(im.mode, WH_max)
                    out.paste(im)
                    out.save(join(resized_image_folder, name))
