import scipy.io
from os import listdir
from os.path import isfile, join
import shutil
import csv


def get_images_with_one_face_train_val(image_path, mat_path, output_path_val, output_path_train):
    mat2 = scipy.io.loadmat(mat_path)

    # Print caratteristiche file mat
    # for k in mat2:
    #    print(k)
    # Con la chiave stringa si accede al gruppo di cartelle che contengono quella proprietà: Es. bbx_list Con la
    # seconda chiave si accede alla cartella desiderata, ricorda che windows ordina gli indici con 0,1,10,11...19,
    # 20 Con il terzo indice si accede a tutte le caratteristiche effettive delle foto contenute nella cartella Con
    # il quarto indice alle feature della singola foto
    # Noi siamo interessati solo alle feature bbx, che si
    onlyDir = [f for f in listdir(image_path) if not isfile(join(image_path, f))]
    list_all = zip(mat2["face_bbx_list"], mat2["file_list"], onlyDir)

    #list_training = list_all[:(int((len(onlyDir) - 1) * 0.8))]
    list_training = [
        [zip(mat2["face_bbx_list"][:(int((len(onlyDir) - 1) * 0.8))], mat2["file_list"][:(int((len(onlyDir) - 1) * 0.8))], onlyDir[:(int((len(onlyDir) - 1) * 0.8))]), output_path_train],
        [zip(mat2["face_bbx_list"][(int((len(onlyDir) - 1) * 0.8)):len(onlyDir)-1], mat2["file_list"][(int((len(onlyDir) - 1) * 0.8)):len(onlyDir)-1], onlyDir[(int((len(onlyDir) - 1) * 0.8)):len(onlyDir)-1]), output_path_val]
    ]

    for zip_training in list_training:
        # filenames_single_photos = np.array([])
        # all_labels_nparray = np.array([])
        file_description = []
        for bbx, f_l, dir_name in zip_training[0]:  # bbx is nested, f_l is directly connected to filenames of the folder
            pair_photo_filename = zip(bbx[0], f_l[0])
            for bbx_single_photo, single_photo_filename in pair_photo_filename:
                if len(bbx_single_photo[0]) == 1:
                    # print(bbx_single_photo)
                    file_description.append((str(single_photo_filename[0][0]) + ".jpg"
                                             , list(list(bbx_single_photo[0])[0])))
                    shutil.copy(image_path + "\\" + dir_name + "\\" + str(single_photo_filename[0][0]) + ".jpg",
                                zip_training[1])
        print(len(file_description))
        with open(zip_training[1] + "\\output.csv", "w", newline='') as f:
            csv_out = csv.writer(f)
            csv_out.writerow(['filename', 'labels'])
            for row in file_description:
                csv_out.writerow(row)


def get_images_with_one_face_test(image_path, mat_path, output_path):
    mat2 = scipy.io.loadmat(mat_path)

    # Print caratteristiche file mat
    # for k in mat2:
    #    print(k)
    # Con la chiave stringa si accede al gruppo di cartelle che contengono quella proprietà: Es. bbx_list Con la
    # seconda chiave si accede alla cartella desiderata, ricorda che windows ordina gli indici con 0,1,10,11...19,
    # 20 Con il terzo indice si accede a tutte le caratteristiche effettive delle foto contenute nella cartella Con
    # il quarto indice alle feature della singola foto
    # Noi siamo interessati solo alle feature bbx, che si
    onlyDir = [f for f in listdir(image_path) if not isfile(join(image_path, f))]
    list_all = zip(mat2["face_bbx_list"], mat2["file_list"], onlyDir)
    # filenames_single_photos = np.array([])
    # all_labels_nparray = np.array([])
    file_description = []
    for bbx, f_l, dir_name in list_all:  # bbx is nested, f_l is directly connected to filenames of the folder
        pair_photo_filename = zip(bbx[0], f_l[0])
        for bbx_single_photo, single_photo_filename in pair_photo_filename:
            if len(bbx_single_photo[0]) == 1:
                # print(bbx_single_photo)
                file_description.append((str(single_photo_filename[0][0]) + ".jpg"
                                         , list(list(bbx_single_photo[0])[0])))
                shutil.copy(image_path + "\\" + dir_name + "\\" + str(single_photo_filename[0][0]) + ".jpg", output_path)
    print(len(file_description))
    with open(output_path + "\\output.csv", "w", newline='') as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['filename', 'labels'])
        for row in file_description:
            csv_out.writerow(row)


if __name__ == '__main__':
    get_images_with_one_face_train_val(".\\dataset\\WIDER_train\\images", ".\\dataset\\wider_face_split\\wider_face_train.mat",
                                       ".\\dataset\\val", ".\\dataset\\train")
    get_images_with_one_face_test(".\\dataset\\WIDER_val\\images", ".\\dataset\\wider_face_split\\wider_face_val.mat", ".\\dataset\\test")
