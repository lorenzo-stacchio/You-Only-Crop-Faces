from PIL import Image, ImageDraw
import pandas as pd
import ast
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import walk

image_paths_and_description_file = [(".\\dataset\\NUOVO\\val",
                                     ".\\dataset\\NUOVO\\val\\output.csv",
                                     ".\\dataset\\NUOVO\\val\\output_final.csv",
                                     ".\\dataset\\NUOVO\\val\\output_segmentation.csv"),
                                    (".\\dataset\\NUOVO\\train",
                                     ".\\dataset\\NUOVO\\train\\output.csv",
                                     ".\\dataset\\NUOVO\\train\\output_final.csv",
                                     ".\\dataset\\NUOVO\\train\\output_segmentation.csv"),
                                    (".\\dataset\\NUOVO\\test",
                                     ".\\dataset\\NUOVO\\test\\output.csv",
                                     ".\\dataset\\NUOVO\\test\\output_final.csv",
                                     ".\\dataset\\NUOVO\\test\\output_segmentation.csv")
                                    ]

IMG_SIZE = 224
# Il resto deve dare 0
BLOCKS_NUM = 7
BLOCK_SIZE = IMG_SIZE / BLOCKS_NUM


def calculate_box(x, y, w, h):
    left_top = (x, y)
    bottom_right = (x + w, y + h)
    # output = []
    output = ""
    for i in range(BLOCKS_NUM):
        for j in range(BLOCKS_NUM):
            if i * BLOCK_SIZE <= bottom_right[0] and ((i + 1) * BLOCK_SIZE) >= left_top[0] and j * BLOCK_SIZE <= \
                    bottom_right[1] and ((j + 1) * BLOCK_SIZE) >= left_top[1]:
                new_x = left_top[0] if i * BLOCK_SIZE < left_top[0] else i * BLOCK_SIZE
                new_y = left_top[1] if j * BLOCK_SIZE < left_top[1] else j * BLOCK_SIZE

                width = bottom_right[0] - new_x if (i + 1) * BLOCK_SIZE > bottom_right[0] else (
                                                                                                           i + 1) * BLOCK_SIZE - new_x
                height = bottom_right[1] - new_y if (j + 1) * BLOCK_SIZE > bottom_right[1] else (
                                                                                                            j + 1) * BLOCK_SIZE - new_y

                # output.append([1, int(new_x + (width / 2)), int(new_y + (height / 2)), round(width / BLOCK_SIZE, 2), round(height / BLOCK_SIZE, 2)])
                output += "1," + str(int(new_x + (width / 2))) + "," + str(int(new_y + (height / 2))) + "," + str(
                    round(width / BLOCK_SIZE, 2)) + "," + str(round(height / BLOCK_SIZE, 2)) + ","

            else:
                # output.append([0, 0, 0, 0, 0])
                output += "0,0,0,0,0,"

    return output.strip(",")


if __name__ == '__main__':

    for el in image_paths_and_description_file:
        file = open(el[3], "w")

        #new_df = pd.DataFrame(columns=["image_id", "label"])

        lineToWrite = "image_id,"
        for i in range(BLOCKS_NUM * BLOCKS_NUM * 5):
            lineToWrite += str(i) + ","
        file.write(lineToWrite.strip(",") + "\n")

        #for n in range(int(BLOCKS_NUM * BLOCKS_NUM * 5)):
        #    new_df[str(n)] = 0

        f = []
        df = pd.read_csv(el[2])
        for index, row in df.iterrows():
            if index % 100 == 0 and index != 0:
                print("Processed " + str(index) + " images")

            lineToWrite = row["image_id"] + ","
            lineToWrite += calculate_box(row["x_1"], row["y_1"], row["width"], row["height"])
            file.write(lineToWrite + "\n")

            #output = calculate_box(row["x_1"], row["y_1"], row["width"], row["height"])
            #actual_dict = {"image_id": row["image_id"]}
            #flat_output = [item for sublist in output for item in sublist]
            # for el in range(len(flat_output)):
            #    actual_dict[str(el)] = flat_output[el]
            # new_df = new_df.append(actual_dict, ignore_index=True)

            # im = Image.open(".\\dataset\\NUOVO\\val\\" + row["image_id"], 'r')
            # draw = ImageDraw.Draw(im)
            #
            # left_top = (row["x_1"], row["y_1"])
            # bottom_right = (row["x_1"] + row["width"], row["y_1"] + row["height"])
            # for i in range(BLOCKS_NUM):
            #     for j in range(BLOCKS_NUM):
            #
            #         #draw.point((new_x + (width / 2), new_y + (height / 2)), fill="White")
            #         #draw.text((new_x + (width / 2), new_y + (height / 2)), text)
            #         index = int((i * BLOCKS_NUM) + j)
            #         if output[index][0] == 1:
            #             draw.rectangle(((i * BLOCK_SIZE, j * BLOCK_SIZE),
            #                             (i * BLOCK_SIZE + BLOCK_SIZE, j * BLOCK_SIZE + BLOCK_SIZE)),
            #                            outline="Green")
            #             draw.point((output[index][1], output[index][2]), fill="White")
            #         else:
            #             draw.rectangle(((i * BLOCK_SIZE, j * BLOCK_SIZE),
            #                             (i * BLOCK_SIZE + BLOCK_SIZE, j * BLOCK_SIZE + BLOCK_SIZE)),
            #                            outline="Black")
            #
            # im.show()
        #new_df.to_csv(el[3])
