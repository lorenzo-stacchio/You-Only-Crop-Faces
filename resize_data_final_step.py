from PIL import Image, ImageDraw
import pandas as pd
import ast
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

WIDTH_SCALE, HEIGHT_SCALE = 300, 300


def reduce_image_size_and_return_reduced_labels(image_path, labels, new_width, new_height):
    im = Image.open(image_path, 'r')
    width, height = im.size
    # print(width)
    # print(height)
    scaling_factor_width, scaling_factor_height = width / new_width, height / new_height
    # print(scaling_factor_width)
    # print(scaling_factor_height)
    # Get labels info about the image
    # label = image_path.loc[image_path["filename"] == "0_Parade_marchingband_1_356.jpg"]["labels"]
    label_fixed = ast.literal_eval(labels)  # to threat the string as tuple
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

    #draw = ImageDraw.Draw(im)
    #draw.rectangle(((new_label_x, new_label_y), (new_label_x + new_label_width, new_label_y + new_label_height)),
    #    outline="Red")
    #im.save("test.jpg")
    #img = mpimg.imread('test.jpg')
    #imgplot = plt.imshow(img)
    #plt.show()


    return im, [new_label_x, new_label_y, new_label_width, new_label_height]


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
            img_reduced, new_labels = reduce_image_size_and_return_reduced_labels(image_path, row["labels"],
                                                                                  WIDTH_SCALE, HEIGHT_SCALE)
            img_reduced.save(image_path)  # this is a pillow image
            row["x_start"] = new_labels[0]
            row["y_start"] = new_labels[1]
            row["width"] = new_labels[2]
            row["height"] = new_labels[3]
            data_frame_finale = data_frame_finale.append(row, ignore_index=False)

        del data_frame_finale["labels"]
        data_frame_finale.to_csv(el[2])
