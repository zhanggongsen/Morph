import os.path
import random
# from data.base_dataset import BaseDataset, get_params, get_transform
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms
from data.image_folder import make_dataset
import SimpleITK as sitk
import numpy as np
import torch


class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        # self.degrees = self.opt.rotation_degrees
        self.degrees = self.opt.rotation_degrees
        self.ifRotate_n90 = self.opt.ifRotate_n90
        self.ifCrop = self.opt.ifCrop


    def __getitem__(self, index):
        # read a image given a random integer index
        AB_path = self.AB_paths[index]

        AB_nii = sitk.ReadImage(AB_path)

        AB_np1 = sitk.GetArrayFromImage(AB_nii)  # <class 'numpy.ndarray'>


        if self.ifCrop == 1:
            # origin_index_x = random.randint(10, self.opt.load_size - self.opt.crop_size-10)
            origin_index_x = random.randint(0, self.opt.load_size - self.opt.crop_size)
            end_index_x = origin_index_x + self.opt.crop_size
            # origin_index_y = random.randint(10, self.opt.load_size - self.opt.crop_size-10)
            origin_index_y = random.randint(0, self.opt.load_size - self.opt.crop_size)
            end_index_y = origin_index_y + self.opt.crop_size
            # origin_index_z = random.randint(10, self.opt.load_size - self.opt.crop_size-10)
            origin_index_z = random.randint(0, self.opt.load_size - self.opt.crop_size)

            end_index_z = origin_index_z + self.opt.crop_size
            AB_np1 = AB_np1[:, origin_index_x:end_index_x, origin_index_y:end_index_y, origin_index_z:end_index_z]
        else:
            pass
        # if self.ifCrop == 1:
        #
        #     AB_np1 = AB_np1[:, 0:96, 0:96, 0:96]
        # else:
        #     pass
        # if self.ifRotate_n90 == 1:
        #     k_n90 = random.randint(0, 3)
        #     if k_n90 == 0:
        #         pass
        #     else:
        #         AB_np1 = np.rot90(AB_np1.swapaxes(1, 3), k_n90, axes=(1, 2)).swapaxes(1, 3)
        # else:
        #     pass

        if self.ifRotate_n90 == 1:
            ifRotate = random.randint(0, 1)
            if ifRotate == 0:
                pass
            else:
                AB_np1 = np.rot90(AB_np1.swapaxes(1, 3), 2, axes=(1, 2)).swapaxes(1, 3)
        else:
            pass


        A_np2 = AB_np1[0, :, :, :]
        B_np2 = AB_np1[1, :, :, :]

        C_np2 = AB_np1[2, :, :, :]
        # D_np2 = AB_np1[3, :, :, :]

        E_1_np2 = AB_np1[5, :, :, :]
        E_2_np2 = AB_np1[6, :, :, :]
        E_3_np2 = AB_np1[7, :, :, :]

        F_np2 = AB_np1[8, :, :,:]

        # G_np2 = AB_np1[9, :, :, :]

        # H_np2 = AB_np1[10, :, :, :]

        I_1_np2 = AB_np1[11, :, :, :]
        I_2_np2 = AB_np1[12, :, :, :]
        I_3_np2 = AB_np1[13, :, :, :]
        J_np2 = AB_np1[14, :, :, :]
        K_np2 = AB_np1[15, :, :, :]


        A_np3 = np.tile(A_np2.astype(np.float32), (1, 1, 1, 1))
        B_np3 = np.tile(B_np2.astype(np.float32), (1, 1, 1, 1))
        C_np3 = np.tile(C_np2.astype(np.float32), (1, 1, 1, 1))
        # D_np3 = np.tile(D_np2.astype(np.float32), (1, 1, 1, 1))
        E_1_np3 = np.tile(E_1_np2.astype(np.float32), (1, 1, 1, 1))
        E_2_np3 = np.tile(E_2_np2.astype(np.float32), (1, 1, 1, 1))
        E_3_np3 = np.tile(E_3_np2.astype(np.float32), (1, 1, 1, 1))
        F_np3 = np.tile(F_np2.astype(np.float32), (1, 1, 1, 1))
        # G_np3 = np.tile(G_np2.astype(np.float32), (1, 1, 1, 1))
        # H_np3 = np.tile(H_np2.astype(np.float32), (1, 1, 1, 1))
        I_1_np3 = np.tile(I_1_np2.astype(np.float32), (1, 1, 1, 1))
        I_2_np3 = np.tile(I_2_np2.astype(np.float32), (1, 1, 1, 1))
        I_3_np3 = np.tile(I_3_np2.astype(np.float32), (1, 1, 1, 1))
        J_np3 = np.tile(J_np2.astype(np.float32), (1, 1, 1, 1))
        K_np3 = np.tile(K_np2.astype(np.float32), (1, 1, 1, 1))

        # dataB_gray_high = self.dataB_gray_high
        # dataB_gray_low = self.dataB_gray_low
        # dataA_gray_high = self.dataA_gray_high
        # dataA_gray_low = self.dataA_gray_low

        A_tensor = Scale_and_Normalize_toTensor(A_np3, scale=False,  normalize=False).to(
            torch.float)
        B_tensor = Scale_and_Normalize_toTensor(B_np3, scale=False,  normalize=False).to(
            torch.float)
        C_tensor = Scale_and_Normalize_toTensor(C_np3, scale=False,  normalize=False).to(
            torch.float)
        # D_tensor = Scale_and_Normalize_toTensor(D_np3, scale=False,  normalize=False).to(
        #     torch.float)
        E_1_tensor = Scale_and_Normalize_toTensor(E_1_np3, scale=False,  normalize=False).to(
            torch.float)
        E_2_tensor = Scale_and_Normalize_toTensor(E_2_np3, scale=False, normalize=False).to(
            torch.float)
        E_3_tensor = Scale_and_Normalize_toTensor(E_3_np3, scale=False, normalize=False).to(
            torch.float)
        F_tensor = Scale_and_Normalize_toTensor(F_np3, scale=False,  normalize=False).to(
            torch.float)
        # G_tensor = Scale_and_Normalize_toTensor(G_np3, scale=False, normalize=False).to(
        #     torch.float)
        # H_tensor = Scale_and_Normalize_toTensor(H_np3, scale=False, normalize=False).to(
        #     torch.float)
        I_1_tensor = Scale_and_Normalize_toTensor(I_1_np3, scale=False, normalize=False).to(
            torch.float)
        I_2_tensor = Scale_and_Normalize_toTensor(I_2_np3, scale=False, normalize=False).to(
            torch.float)
        I_3_tensor = Scale_and_Normalize_toTensor(I_3_np3, scale=False, normalize=False).to(
            torch.float)
        J_tensor = Scale_and_Normalize_toTensor(J_np3, scale=False, normalize=False).to(
            torch.float)
        K_tensor = Scale_and_Normalize_toTensor(K_np3, scale=False, normalize=False).to(
            torch.float)

        return {'A': A_tensor, 'B': B_tensor, 'C': C_tensor, 'E_1': E_1_tensor, 'E_2': E_2_tensor, 'E_3': E_3_tensor, 'F': F_tensor, 'I_1': I_1_tensor, 'I_2': I_2_tensor, 'I_3': I_3_tensor,'J': J_tensor,'K': K_tensor,'A_paths': AB_path,
                'B_paths': AB_path, 'C_paths': AB_path, 'D_paths': AB_path, 'E_1_paths': AB_path,'E_2_paths': AB_path,'E_3_paths': AB_path, 'F_paths': AB_path,'G_paths': AB_path,'I_1_paths': AB_path,'I_2_paths': AB_path,'I_3_paths': AB_path,'J_paths': AB_path,'K_paths': AB_path}

    def __len__(self):

        return len(self.AB_paths)

def Scale_and_Normalize_toTensor(input_np, gray_low=0, gray_high=1, scale=False, normalize=False):
    if isinstance(input_np, np.ndarray):
        input_np = input_np.astype(np.float32)
        if scale == True:
            output_np = (input_np - gray_low) * (1 - 0) / (gray_high - gray_low)
        else:
            output_np=input_np
        float__tensor = torch.from_numpy(output_np)
        if normalize == True:
            # normalize [0,1] tensor to [-1,1].
            output_tensor = (float__tensor - 0.5) / 0.5
        else:
            output_tensor = float__tensor
    return output_tensor
