import SimpleITK as sitk
import numpy as np
import os

site = "Abdominal"
Root = r"4D_0_90_{}".format(site,site)

dir_list = []

def cal_centroid(array_X, voxel_spacings):
    index_X = np.where(array_X != 0)
    array_index_X = np.transpose(np.array(index_X))
    array_index_X_SI = array_index_X[:, 0]
    array_index_X_AP = array_index_X[:, 1]
    array_index_X_LR = array_index_X[:, 2]
    centroid_X_SI = np.mean(array_index_X_SI) * voxel_spacings[2]
    centroid_X_AP = np.mean(array_index_X_AP) * voxel_spacings[0]
    centroid_X_LR = np.mean(array_index_X_LR) * voxel_spacings[1]
    return centroid_X_SI, centroid_X_AP, centroid_X_LR

for root, dirs, files in os.walk(Root):
    for dir_i in dirs:
        dirs_path = os.path.join(root, dir_i)
        if len(dirs_path) == (len(Root) + 9):
            dir_list.append(dirs_path)


for phase in ['Phase0', 'Phase50']:
    ID_list = []
    spacing_0_list = []
    spacing_2_list = []
    size_0_list = []
    size_2_list = []

    GTV_index_X_list_0 = []
    GTV_index_X_list_1 = []
    GTV_index_Y_list_0 = []
    GTV_index_Y_list_1 = []
    GTV_index_Z_list_0 = []
    GTV_index_Z_list_1 = []

    GTV_volume_list = []

    Centroid_SI_list = []
    Centroid_AP_list = []
    Centroid_LR_list = []

    for path_toprocess in dir_list:
        ID = path_toprocess[-8:]
        ID_list.append(ID)

        file_toprocess_GTV = os.path.join(os.path.join(path_toprocess, "structs_{}".format(phase[5:])), "mask_GTV.nii")

        spacing = sitk.ReadImage(file_toprocess_GTV).GetSpacing()
        spacing_0 = spacing[0]
        spacing_0_list.append(spacing_0)
        spacing_2 = spacing[2]
        spacing_2_list.append(spacing_2)

        size = sitk.ReadImage(file_toprocess_GTV).GetSize()
        size_0 = size[0]
        size_0_list.append(size_0)
        size_2 = size[2]
        size_2_list.append(size_2)

        array_GTV = sitk.GetArrayFromImage(sitk.ReadImage(file_toprocess_GTV))

        GTV_volume = (np.sum(array_GTV != 0)) * (spacing_0) * (spacing_0) * (spacing_2)
        GTV_volume_list.append(GTV_volume)

        centroid_SI, centroid_AP, centroid_LR = cal_centroid(array_GTV, spacing)
        Centroid_SI_list.append(centroid_SI)
        Centroid_AP_list.append(centroid_AP)
        Centroid_LR_list.append(centroid_LR)

        mask_GTV_indexes = np.where(array_GTV != 0)

        GTV_index_X_0 = np.min(mask_GTV_indexes[2])
        GTV_index_X_list_0.append(GTV_index_X_0)
        GTV_index_X_1 = np.max(mask_GTV_indexes[2])
        GTV_index_X_list_1.append(GTV_index_X_1)
        GTV_index_Y_0 = np.min(mask_GTV_indexes[1])
        GTV_index_Y_list_0.append(GTV_index_Y_0)
        GTV_index_Y_1 = np.max(mask_GTV_indexes[1])
        GTV_index_Y_list_1.append(GTV_index_Y_1)
        GTV_index_Z_0 = np.min(mask_GTV_indexes[0])
        GTV_index_Z_list_0.append(GTV_index_Z_0)
        GTV_index_Z_1 = np.max(mask_GTV_indexes[0])
        GTV_index_Z_list_1.append(GTV_index_Z_1)

        print("******ID:{}******".format(ID))

    if phase == 'Phase0':
        GTV_volume_array_0 = np.around(np.array(GTV_volume_list), 2)
        Centroid_SI_array_0 = np.around(np.array(Centroid_SI_list), 2)
        Centroid_AP_array_0 = np.around(np.array(Centroid_AP_list), 2)
        Centroid_LR_array_0 = np.around(np.array(Centroid_LR_list), 2)
    if phase == "Phase50":
        GTV_volume_array_50 = np.around(np.array(GTV_volume_list), 2)
        Centroid_SI_array_50 = np.around(np.array(Centroid_SI_list), 2)
        Centroid_AP_array_50 = np.around(np.array(Centroid_AP_list), 2)
        Centroid_LR_array_50 = np.around(np.array(Centroid_LR_list), 2)

        delta_SI = Centroid_SI_array_0 - Centroid_SI_array_50
        delta_AP = Centroid_AP_array_0 - Centroid_AP_array_50
        delta_LR = Centroid_LR_array_0 - Centroid_LR_array_50
        Amplitude = np.power((Centroid_SI_array_0 - Centroid_SI_array_50), 2) + np.power(
            (Centroid_AP_array_0 - Centroid_AP_array_50), 2) + np.power((Centroid_LR_array_0 - Centroid_LR_array_50), 2)
        Amplitude_root = np.power(Amplitude, 0.5)

        delta_volume = GTV_volume_array_0 - GTV_volume_array_50
        Centroids_nottrans = np.stack([delta_SI, delta_AP, delta_LR, Amplitude_root], axis=0)
        delta_centriods = np.transpose(Centroids_nottrans)

        np.savetxt("./Changes_of_Centroid_and_Volume/delta_volume_{}.txt".format(site), delta_volume)
        np.savetxt("./Changes_of_Centroid_and_Volume/delta_centriods_{}.txt".format(site), delta_centriods)

        with open('./Changes_of_Centroid_and_Volume/summary_{}.txt'.format(site), 'a') as f:
            f.write("delta_SI: " + "min " + str(np.min(delta_SI)) + " max " + str(np.max(delta_SI))+ " median " + str(np.median(delta_SI)) + "\n" +
                    "delta_AP: " + "min " + str(np.min(delta_AP)) + " max " + str(np.max(delta_AP))+ " median " + str(np.median(delta_AP)) + "\n" +
                    "delta_LR: " + "min " + str(np.min(delta_LR)) + " max " + str(np.max(delta_LR)) + " median " + str(np.median(delta_LR)) + "\n" +
                    "Amplitude_root: " + "min " + str(np.min(Amplitude_root)) + " max " + str(np.max(Amplitude_root))+ " median " + str(np.median(Amplitude_root)) + "\n" +
                    "delta_volume: " + "min " + str(np.min(delta_volume)) + " max " + str(np.max(delta_volume)) + " median " + str(np.median(delta_volume)))
