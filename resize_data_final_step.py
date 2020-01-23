from PIL import Image, ImageDraw
import pandas as pd
import ast
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_paths_and_description_file = [(".\\dataset\\train",
                                     ".\\dataset\\train\\output.csv",
                                     ".\\dataset\\train\\output_final.csv"),
                                    (".\\dataset\\test",
                                     ".\\dataset\\test\\output.csv",
                                     ".\\dataset\\test\\output_final.csv"),
                                    (".\\dataset\\val",
                                     ".\\dataset\\val\\output.csv",
                                     ".\\dataset\\val\\output_final.csv")
                                    ]

WIDTH_SCALE, HEIGHT_SCALE = 224, 224

IMG_SIZE = 224
IMG_SIZE_pre_processing = 256

PADDING_IMAGE = 5000
FINAL_IMG_SIZE = 224


def reduce_image_size_and_return_reduced_labels(im, label_fixed, new_width, new_height):
    width, height = im.size
    # print(width)
    # print(height)
    scaling_factor_width, scaling_factor_height = width / new_width, height / new_height
    # print(scaling_factor_width)
    # print(scaling_factor_height)
    # Get labels info about the image
    # label = image_path.loc[image_path["filename"] == "0_Parade_marchingband_1_356.jpg"]["labels"]
    label_x, label_y, label_width, label_height = label_fixed[0], label_fixed[1], label_fixed[2], label_fixed[3]

    # print(label_x)
    # print(label_y)
    # print(label_width)
    # print(label_height)
    # try to resize
    new_label_x, new_label_y, new_label_width, new_label_height = round(label_x / scaling_factor_width), round(
        label_y / scaling_factor_height), round(label_width / scaling_factor_width), round(
        label_height / scaling_factor_height)

    im = im.resize((new_width, new_height))

    # draw = ImageDraw.Draw(im)
    # draw.rectangle(((new_label_x, new_label_y), (new_label_x + new_label_width, new_label_y + new_label_height)),
    #    outline="Red")
    # im.save("test.jpg")
    # img = mpimg.imread('test.jpg')
    # imgplot = plt.imshow(img)
    # plt.show()

    return im, [new_label_x, new_label_y, new_label_width, new_label_height]


def pad_image(im, row):
    width, height = im.size
    width_padd = (PADDING_IMAGE - width) / 2
    height_padd = (PADDING_IMAGE - height) / 2

    new_im = Image.new("RGB", (PADDING_IMAGE, PADDING_IMAGE))
    new_im.paste(im, (int(width_padd), int(height_padd)))

    row[0] = int(row[0]) + int(width_padd)
    row[1] = int(row[1]) + int(height_padd)

    return new_im, row


def crop_image_centered_face(img, row):
    width_bb = row[2]
    height_bb = row[3]

    padding_width = round(random.uniform(0.2, 2), 2)
    padding_height = round(random.uniform(0.2, 2), 2)
    width_magin = round(random.uniform(0.2, 0.8), 2)
    height_margin = round(random.uniform(0.2, 0.8), 2)

    padd_width_left = int((width_bb * padding_width) * width_magin)
    padd_width_right = int((width_bb * padding_width) * (1 - width_magin))

    padd_height_top = int((height_bb * padding_height) * height_margin)
    padd_height_bot = int((height_bb * padding_height) * (1 - height_margin))

    if padd_width_left + padd_width_right + width_bb < FINAL_IMG_SIZE:
        padd_width_left = int((FINAL_IMG_SIZE - width_bb) * width_magin)
        padd_width_right = int((FINAL_IMG_SIZE - width_bb) * (1 - width_magin))

    if padd_height_top + padd_height_bot + height_bb < FINAL_IMG_SIZE:
        padd_height_top = int((FINAL_IMG_SIZE - height_bb) * height_margin)
        padd_height_bot = int((FINAL_IMG_SIZE - height_bb) * (1 - height_margin))

    left_top = (row[0] - padd_width_left, row[1] - padd_height_top)
    right_bot = (row[0] + row[2] + padd_width_right, row[1] + row[3] + padd_height_bot)

    img = img.crop((left_top[0], left_top[1], right_bot[0], right_bot[1]))
    row[0] = padd_width_left
    row[1] = padd_height_top

    return img, row


def flip(img, row):
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        height = img.size[1]
        row[1] = height - row[1] - row[3]
    else:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        width = img.size[0]
        row[0] = width - row[0] - row[2]

    return img, row


def compare_and_add_data_augmentation_to_dataframe(image_path, df, old_row, new_row, new_img, suffix):
    if not new_row == old_row:
        file_name = row["filename"].split(".")[0] + suffix + ".jpg"
        img_reduced, new_labels = reduce_image_size_and_return_reduced_labels(new_img, new_row, WIDTH_SCALE,
                                                                              HEIGHT_SCALE)
        dict_to_append = {"filename": file_name, "x_start": new_labels[0],
                          "y_start": new_labels[1], "width": new_labels[2], "height": new_labels[3]}
        df = df.append(dict_to_append, ignore_index=True)
        img_reduced.save(image_path + "//" + file_name)  # this is a pillow image

    return df


def save_image(img, file_name, image_path, df, labels):
    img_reduced, new_labels = reduce_image_size_and_return_reduced_labels(img, labels, WIDTH_SCALE, HEIGHT_SCALE)
    dict_to_append = {"filename": file_name, "x_start": new_labels[0], "y_start": new_labels[1], "width": new_labels[2], "height": new_labels[3]}
    df = df.append(dict_to_append, ignore_index=True)

    #draw = ImageDraw.Draw(img_reduced)
    #draw.rectangle(((new_labels[0], new_labels[1]), (new_labels[0] + new_labels[2], new_labels[1] + new_labels[3])), outline="Red")

    img_reduced.save(image_path + "//" + file_name)  # this is a pillow image
    return df


if __name__ == '__main__':
    for el in image_paths_and_description_file:
        print("Processing " + el[0])
        data_frame_finale = pd.DataFrame(columns=["filename", "labels", "x_start", "y_start", "width", "height"])
        data_frame_description_step2 = pd.read_csv(el[1])
        data_frame_description_step2["x_start"] = 0
        data_frame_description_step2["y_start"] = 0
        data_frame_description_step2["width"] = 0
        data_frame_description_step2["height"] = 0
        for index, row in data_frame_description_step2.iterrows():
            if index % 100 == 0:
                print("Processed " + str(index) + " images")
            image_path = el[0] + "//" + row["filename"]

            #default image
            default_img = Image.open(image_path, 'r')
            default_labels = ast.literal_eval(row["labels"])
            data_frame_finale = save_image(default_img, row["filename"], el[0], data_frame_finale, default_labels)

            #flipped image
            flipped_img, flipped_labels = flip(default_img, default_labels.copy())
            cropped_img, cropped_labels = crop_image_centered_face(flipped_img, flipped_labels.copy())
            new_file_name = row["filename"].split(".")[0] + "_flipped.jpg"
            data_frame_finale = save_image(cropped_img, new_file_name, el[0], data_frame_finale, cropped_labels)

            #cropped image
            padded_img, padded_labels = pad_image(default_img, default_labels.copy())
            cropped_img, cropped_labels = crop_image_centered_face(padded_img, padded_labels.copy())
            new_file_name = row["filename"].split(".")[0] + "_cropped_.jpg"
            data_frame_finale = save_image(cropped_img, new_file_name, el[0], data_frame_finale, cropped_labels)


        del data_frame_finale["labels"]
        data_frame_finale.to_csv(el[2])