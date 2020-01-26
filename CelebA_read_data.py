import os
import re
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

bb_file_path = ".\\dataset\\NUOVO\\CelebA\\Anno\\"
bb_file_name = "list_bbox_celeba.csv"
bb_new_file_name = "parsed_list_bbox_celeba.csv"
image_path = ".\\dataset\\NUOVO\\CelebA\\Img\\img_align_celeba"


def prepare_bb_txt_file():
    with open(bb_file_path + "\\" + bb_file_name, "r") as fr:
        with open(bb_file_path + "\\" + bb_new_file_name, "w") as fw:
            for line in fr:
                tmp = re.sub("\s+", ",", line)
                tmp = tmp.rstrip(',') + "\n"
                fw.writelines(tmp)


def prepare_images_and_csv():
    dfs = pd.read_csv(bb_file_path + "\\" + bb_new_file_name)
    train, test = train_test_split(dfs, test_size=0.1)
    train, val = train_test_split(train, test_size=0.1)

    count = 0

    for df, path in zip([val, train, test], [".\\dataset\\NUOVO\\val", ".\\dataset\\NUOVO\\train", ".\\dataset\\NUOVO\\test"]):
        for index, row in df.iterrows():
            count += 1
            if count % 100 == 0:
                print("Computed", count, " over ", len(dfs))

            shutil.copy(image_path + "\\" + row['image_id'], path + "\\" + row['image_id'])

        df.to_csv(path + "\\" + "output.csv")



if __name__ == '__main__':
    if os.path.exists(".\\dataset\\NUOVO\\val"):
        shutil.rmtree(".\\dataset\\NUOVO\\val")
        os.makedirs(".\\dataset\\NUOVO\\val")
    else:
        os.makedirs(".\\dataset\\NUOVO\\val")

    if os.path.exists(".\\dataset\\NUOVO\\test"):
        shutil.rmtree(".\\dataset\\NUOVO\\test")
        os.makedirs(".\\dataset\\NUOVO\\test")
    else:
        os.makedirs(".\\dataset\\NUOVO\\test")

    if os.path.exists(".\\dataset\\NUOVO\\train"):
        shutil.rmtree(".\\dataset\\NUOVO\\train")
        os.makedirs(".\\dataset\\NUOVO\\train")
    else:
        os.makedirs(".\\dataset\\NUOVO\\train")

    #prepare_bb_txt_file()
    prepare_images_and_csv()
