import scipy.io
from os import listdir
from os.path import isfile, join
import shutil
import csv

if __name__ == '__main__':
    image_path = 'C:\\Users\\Lorenzo Stacchio\\Downloads\\WIDER_val\\images'
    mat2 = scipy.io.loadmat('C:\\Users\\Lorenzo Stacchio\\Desktop\\Machine_learning_project\\dataset\\test_set'
                            '\\wider_face_val.mat')
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
    #filenames_single_photos = np.array([])
    #all_labels_nparray = np.array([])
    file_description = []
    for bbx, f_l, dir_name in list_all:  # bbx is nested, f_l is directly connected to filenames of the folder
        pair_photo_filename = zip(bbx[0], f_l[0])
        for bbx_single_photo, single_photo_filename in pair_photo_filename:
            if len(bbx_single_photo[0]) == 1:
                #print(bbx_single_photo)
                file_description.append((image_path + "\\" + dir_name + "\\" + str(single_photo_filename[0][0]) + ".jpg"
                                         , bbx_single_photo[0]))
                # print(image_path + "\\" + dir_name + "\\" + str(single_photo_filename[0][0]) + ".jpg")
    print(len(file_description))
    #print(file_description)
    #with open("C:\\Users\\Lorenzo Stacchio\\Desktop\\Machine_learning_project\\dataset\\train_set\\output.csv", "w", newline='') as f:
    #    csv_out = csv.writer(f)
    #    csv_out.writerow(['filename', 'labels'])
    #    for row in file_description:
    #        csv_out.writerow(row)
    # Copia e incolla tutte le foto su un'altra directory a mia scelta
    # Copy all the files to another directory
    if True: #made to execute or no execute this code
        for file_name,values in file_description:
            shutil.copy(file_name, 'C:\\Users\\Lorenzo Stacchio\\Desktop\\Machine_learning_project\\dataset\\test_set\\images')

    # Write file description
