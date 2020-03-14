# convert test and train set in pixels
from PIL import Image
import pandas as pd
import csv


def write_csv_pixels(input_csv_path, images_input_path, output_csv_path):
    count = 0
    df_input = pd.read_csv(input_csv_path)
    with open(output_csv_path, "w", newline='') as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['pixels', 'labels'])
        for index, row in df_input.iterrows():  # df_traing has columns: filename, labels
            count = count + 1
            if (count % 100) == 0:
                print("Elaborate le prime " + str(count) + "immagini")
            path_image = images_input_path + "//" + str(row['filename'])
            csv_out.writerow((convert_image_pixel(path_image), row['labels']))


def convert_image_pixel(image_path):
    im = Image.open(image_path, 'r')
    return list(im.getdata())


# tuple of: input_images_path, input_dataframe_describe_data, output_path_dataframe_describe_data
path_description = [["//media//Disco_Secondario//datasets//face_segmentation//test_set//output.csv",
                        "//media//Disco_Secondario//datasets//face_segmentation//test_set//resized",
                     "//media//Disco_Secondario//datasets//face_segmentation//test_set//output_pixels.csv"],
                    ["//media//Disco_Secondario//datasets//face_segmentation//train_set//output.csv",
                     "//media//Disco_Secondario//datasets//face_segmentation//train_set//resized",
                     "//media//Disco_Secondario//datasets//face_segmentation//train_set//output_pixels.csv"]]

if __name__ == '__main__':
    for el in path_description:
        write_csv_pixels(el[0], el[1], el[2])
