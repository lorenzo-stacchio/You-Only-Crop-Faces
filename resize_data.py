from PIL import Image
from os import listdir
from os.path import isfile, join

max_height = 0

image_folder = 'C:\\Users\\Lorenzo Stacchio\\Desktop\\Machine_learning_project\\dataset\\test_set'
resized_image_folder = 'C:\\Users\\Lorenzo Stacchio\\Desktop\\Machine_learning_project\\dataset\\test_set' \
                       '\\resized'


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


if __name__ == "__main__":
    image_path = image_folder
    pictures = [f for f in listdir(image_path) if isfile(join(image_path, f))]

    print("Max size calculation")
    for pic in pictures:
        getsize(pic)

    print("Resize started")
    for pic in pictures:
        #print("Resizing " + pic)
        resize(pic)
