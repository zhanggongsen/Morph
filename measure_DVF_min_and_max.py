import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

Root = "Augmented_4DCT"
Patient_IDs = os.listdir(Root)
for Patient_ID in Patient_IDs:
    Patient_path = os.path.join(Root, Patient_ID)
    print("**** Patient_ID:{} **** ".format(Patient_ID))
    Augmented_IDs = os.listdir(Patient_path)
    min_list_patient,max_list_patient,mean_list_patient=[],[],[]
    for Augmented_ID in Augmented_IDs:
        Augmented_path = os.path.join(Patient_path, Augmented_ID)
        Augmented_file_path = os.path.join(Augmented_path, "DVF.nii")
        Augmented_image = sitk.ReadImage(Augmented_file_path)
        Augmented_DVF = sitk.GetArrayFromImage(Augmented_image)
        min_list_patient.append(np.min(Augmented_DVF))
        max_list_patient.append(np.max(Augmented_DVF))
        mean_list_patient.append(np.mean(Augmented_DVF))
        Augmented_DVF_X = sitk.GetArrayFromImage(Augmented_image)[:,:,:,0]
        Augmented_DVF_Y = sitk.GetArrayFromImage(Augmented_image)[:, :, :, 1]
        Augmented_DVF_Z = sitk.GetArrayFromImage(Augmented_image)[:, :, :, 2]

        print("    Augmented_ID:{} image_size:{} array_Shape:{}".format(Augmented_ID, Augmented_image.GetSize(),
                                                                        Augmented_DVF.shape))
        print("        DVF min:{} max:{} mean:{}".format(np.min(Augmented_DVF), np.max(Augmented_DVF),
                                                     np.mean(Augmented_DVF)))
        print("            DVF_X min:{} max:{} mean:{}".format(np.min(Augmented_DVF_X), np.max(Augmented_DVF_X),
                                                     np.mean(Augmented_DVF_X)))
        print("            DVF_Y min:{} max:{} mean:{}".format(np.min(Augmented_DVF_Y), np.max(Augmented_DVF_Y),
                                                     np.mean(Augmented_DVF_Y)))
        print("            DVF_Z min:{} max:{} mean:{}".format(np.min(Augmented_DVF_Z), np.max(Augmented_DVF_Z),
                                                  np.mean(Augmented_DVF_Z)))
    print("**** DVF min:{} max:{} mean:{}".format(np.min(np.array(min_list_patient)), np.max(np.array(max_list_patient)),
                                                   np.mean(np.array(mean_list_patient))))


