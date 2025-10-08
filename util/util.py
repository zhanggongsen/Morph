from __future__ import print_function
import torch
import numpy as np
# from PIL import Image
import os
import SimpleITK as sitk


def tensor2im(input_image,  label="none", imtype=np.float32):

    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        # print("********************************", image_tensor.cpu().float().numpy().shape)
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (1, 1, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling   因为要映射回float 所以不能用255
        # if "A" in label:
        #     # image_numpy = (np.transpose(image_numpy, (1, 2, 3, 0)) + 1) / 2.0 * (
        #     #         gray_high_dataA - gray_low_dataA) + gray_low_dataA  # post-processing: tranpose and scaling back to dataA
        #     image_numpy = ((np.transpose(image_numpy, (1, 2, 3, 0))) * (
        #             gray_high_dataA - gray_low_dataA)) + gray_low_dataA
        #     # print("******************************** A.shape", image_numpy.shape)
        # else:
        #     # image_numpy = (np.transpose(image_numpy, (1, 2, 3, 0)) + 1) / 2.0 * (
        #     #         gray_high_dataB - gray_low_dataB) + gray_low_dataB  # post-processing: tranpose and scaling back to dataA
        #     image_numpy = ((np.transpose(image_numpy, (1, 2, 3, 0))) * (
        #             gray_high_dataB - gray_low_dataB)) + gray_low_dataB

        # if "B" in label:
        #     # image_numpy = (np.transpose(image_numpy, (1, 2, 3, 0)) + 1) / 2.0 * (
        #     #         gray_high_dataB - gray_low_dataB) + gray_low_dataB  # post-processing: tranpose and scaling back to dataA
        #     image_numpy = ((np.transpose(image_numpy, (1, 2, 3, 0))) * (
        #             gray_high_dataB - gray_low_dataB)) + gray_low_dataB
        # if "C" in label:
        #     # image_numpy = (np.transpose(image_numpy, (1, 2, 3, 0)) + 1) / 2.0 * (
        #     #         gray_high_dataB - gray_low_dataB) + gray_low_dataB  # post-processing: tranpose and scaling back to dataA
        #     image_numpy = ((np.transpose(image_numpy, (1, 2, 3, 0))) * (gray_high_dataB - gray_low_dataB)) + gray_low_dataB
        # if "D" in label:
        #     # image_numpy = (np.transpose(image_numpy, (1, 2, 3, 0)) + 1) / 2.0 * (
        #     #         gray_high_dataB - gray_low_dataB) + gray_low_dataB  # post-processing: tranpose and scaling back to dataA
        #     image_numpy = ((np.transpose(image_numpy, (1, 2, 3, 0))) * (gray_high_dataB - gray_low_dataB)) + gray_low_dataB
        # if "E" in label:
        #     # image_numpy = (np.transpose(image_numpy, (1, 2, 3, 0)) + 1) / 2.0 * (
        #     #         gray_high_dataB - gray_low_dataB) + gray_low_dataB  # post-processing: tranpose and scaling back to dataA
        #     image_numpy = ((np.transpose(image_numpy, (1, 2, 3, 0))) * (gray_high_dataB - gray_low_dataB)) + gray_low_dataB

        image_numpy = np.transpose(image_numpy, (1, 2, 3, 0))
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):

    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path,img_spacing):

    img = sitk.GetImageFromArray(image_numpy)
    img.SetSpacing(img_spacing)
    writer = sitk.ImageFileWriter()
    # writer.SetImageIO("GDCMImageIO")
    writer.SetFileName(image_path)
    writer.Execute(img)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
