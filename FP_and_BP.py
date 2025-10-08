import SimpleITK as sitk
import numpy as np
import astra
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import SimpleITK as sitk
import odl
import time


def calc_space(image):
    size = np.array(image.GetSize())
    spacing = np.array(image.GetSpacing())
    length = size * spacing / 1000.0  # in meters
    # print(size, spacing, length)
    # length and size should be reversed
    length = length[::-1]
    size = size[::-1]
    space = odl.uniform_discr([-l / 2 for l in length], [l / 2 for l in length], size, dtype='float32')
    return space


def convert_hu_to_linear_attenuation(hu_array_0,
                                     MU_WATER=20, MU_AIR=0.02, MU_MAX=None):
    hu_array = np.array(hu_array_0)
    if MU_MAX is None:
        # MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER
        MU_MAX = 1500 * (MU_WATER - MU_AIR) / 1000 + MU_WATER
    # convert values
    hu_array *= (MU_WATER - MU_AIR) / 1000
    hu_array += MU_WATER
    hu_array /= MU_MAX
    np.clip(hu_array, 0., 1., out=hu_array)
    return hu_array * MU_MAX


def convert_linear_attenuation_to_hu(linear_array_0,
                                     MU_WATER=20, MU_AIR=0.02, MU_MAX=None):
    linear_array = np.array(linear_array_0)
    if MU_MAX is None:
        # MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER
        MU_MAX = 1500 * (MU_WATER - MU_AIR) / 1000 + MU_WATER
    linear_array = np.clip(linear_array, 0., MU_MAX)
    # convert values
    hu_array = 1000.0 * (linear_array - MU_WATER) / (MU_WATER - MU_AIR)
    return hu_array


def add_poisson_noise_to_projection(data_0, num_photons=1e5, dtype=np.float32):
    data = np.array(data_0)
    data = np.exp(-data) * num_photons
    data = np.random.poisson(data) / num_photons
    # data_test = np.array(data)
    # data_test[data < 0.1 / num_photons] = 0
    data = np.maximum(0.1 / num_photons, data)
    data = -np.log(data)
    data = data.astype(dtype)
    return data

def add_gaussian_noise_to_projection(proj, I0=1e5, sigma=10, clip_min=1e-3):
    I = I0 * np.exp(-proj)
    noise = np.random.normal(0, sigma, I.shape)
    I_noisy = I + noise
    I_noisy = np.clip(I_noisy, clip_min, None)
    return -np.log(I_noisy / I0)


def Resampling(img, NEW_spacing, lable=False):
    original_size = img.GetSize()
    original_spacing = img.GetSpacing()
    new_spacing = NEW_spacing

    new_size = [int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
                int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
                int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))]
    resampleSliceFilter = sitk.ResampleImageFilter()
    if lable == False:
        Resampleimage = resampleSliceFilter.Execute(img, new_size, sitk.Transform(), sitk.sitkBSpline,
                                                    img.GetOrigin(), new_spacing, img.GetDirection(), 0,
                                                    img.GetPixelIDValue())
        # Resampleimage = resampleSliceFilter.Execute(img, new_size, sitk.Transform(), sitk.sitkNearestNeighbor,
        #                                              img.GetOrigin(), new_spacing, img.GetDirection(), 0,
        #                                              img.GetPixelIDValue())
        # ResampleimageArray = sitk.GetArrayFromImage(Resampleimage)
        Resampleimage = resampleSliceFilter.Execute(img, new_size, sitk.Transform(), sitk.sitkNearestNeighbor,
                                                    img.GetOrigin(), new_spacing, img.GetDirection(), 0,
                                                    img.GetPixelIDValue())
    # ResampleimageArray = sitk.GetArrayFromImage(Resampleimage)
    return Resampleimage


def Resampling_2D(img, NEW_spacing, lable=False):
    original_size = img.GetSize()
    original_spacing = img.GetSpacing()
    new_spacing = NEW_spacing

    # new_size = [int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
    #             int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
    #             int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))]
    new_size = [int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
                int(round(original_size[1] * (original_spacing[1] / new_spacing[1])))]
    resampleSliceFilter = sitk.ResampleImageFilter()
    if lable == False:
        Resampleimage = resampleSliceFilter.Execute(img, new_size, sitk.Transform(), sitk.sitkBSpline,
                                                    img.GetOrigin(), new_spacing, img.GetDirection(), 0,
                                                    img.GetPixelIDValue())
        # Resampleimage = resampleSliceFilter.Execute(img, new_size, sitk.Transform(), sitk.sitkNearestNeighbor,
        #                                              img.GetOrigin(), new_spacing, img.GetDirection(), 0,
        #                                              img.GetPixelIDValue())
        # ResampleimageArray = sitk.GetArrayFromImage(Resampleimage)
        # ResampleimageArray[ResampleimageArray < 0] = 0
    else:
        Resampleimage = resampleSliceFilter.Execute(img, new_size, sitk.Transform(), sitk.sitkNearestNeighbor,
                                                    img.GetOrigin(), new_spacing, img.GetDirection(), 0,
                                                    img.GetPixelIDValue())
    return Resampleimage


def CropImage(image0, cropsize, pad_value=0):
    array0 = sitk.GetArrayFromImage(image0)
    # (filepath, filename) = os.path.split(Masks_file_toProcess)
    Shape0 = array0.shape
    if Shape0[0] >= cropsize[0]:
        array0 = array0
    else:
        array0 = np.pad(array0, (((cropsize[0] - Shape0[0]) // 2, (cropsize[0] - Shape0[0]) // 2 + 1), (0, 0), (0, 0)),
                        "constant", constant_values=pad_value)
    if Shape0[1] >= cropsize[1]:
        array0 = array0
    else:
        array0 = np.pad(array0, ((0, 0), ((cropsize[1] - Shape0[1]) // 2, (cropsize[1] - Shape0[1]) // 2 + 1), (0, 0)),
                        "constant", constant_values=pad_value)
    if Shape0[2] >= cropsize[2]:
        array0 = array0
    else:
        array0 = np.pad(array0, ((0, 0), (0, 0), ((cropsize[2] - Shape0[2]) // 2, (cropsize[2] - Shape0[2]) // 2 + 1)),
                        "constant", constant_values=pad_value)
    Shape1 = array0.shape
    array_cropped = array0[((Shape1[0] - cropsize[0]) // 2):(((Shape1[0] + cropsize[0]) // 2)),
                    ((Shape1[1] - cropsize[1]) // 2):(((Shape1[1] + cropsize[1]) // 2)),
                    ((Shape1[2] - cropsize[2]) // 2):(((Shape1[2] + cropsize[2]) // 2))]
    if array_cropped.dtype == "float32":
        array_float32 = array_cropped
    else:
        array_float32 = array_cropped.astype("float32")
    image_out = sitk.GetImageFromArray(array_float32)
    image_out.SetSpacing(image0.GetSpacing())
    image_out.SetOrigin(image0.GetOrigin())
    return image_out


Read_path = r"Augmented_4DCT"
dir_list = []

for root, dirs, files in os.walk(Read_path):
    for dir_i in dirs:
        dirs_path = os.path.join(root, dir_i)
        if len(dirs_path) == (len(Read_path) + 9):
            dir_list.append(dirs_path)

processed_datanum = 0
for path_toProcess in dir_list:
    time_start_for_one_patient = time.time()
    patient_num = path_toProcess.split('\\')[-1]

    Phase_list = [f for f in os.listdir(path_toProcess)]
    for i in Phase_list:
        # print("  Phase:{}".format(i))
        path_read = os.path.join(path_toProcess, i)
        # path_save = path_read
        # if not os.path.exists(path_save):
        #     os.mkdir(path_save)

        phantom_origin = sitk.ReadImage(os.path.join(path_read, "image.nii"))
        phantom_array = sitk.GetArrayFromImage(phantom_origin).astype(np.float32)

        phantom_array = convert_hu_to_linear_attenuation(phantom_array,
                                                         MU_WATER=20, MU_AIR=0.02, MU_MAX=None)
        phantom = sitk.GetImageFromArray(phantom_array)
        phantom.SetOrigin(phantom_origin.GetOrigin())
        phantom.SetSpacing(phantom_origin.GetSpacing())
        sitk.WriteImage(phantom,os.path.join(path_read, "image_miu.nii"))

        detector_side = [0.397, 0.298]
        detector_size = [1024, 768]
        phantom_space = calc_space(phantom)

        # angle_partition = odl.nonuniform_partition([0, 0.5 * np.pi])
        num_angles = 1
        angle_partition = odl.uniform_partition(min_pt=-np.pi, max_pt=np.pi, shape=num_angles)

        detector_partition = odl.uniform_partition([-detector_side[0] / 2, -detector_side[1] / 2],
                                                   [detector_side[0] / 2, detector_side[1] / 2], detector_size)
        sid = 3.7
        sad = 2.7
        geometry = odl.tomo.ConeFlatGeometry(angle_partition, detector_partition, src_radius=sad, det_radius=sid - sad,
                                             axis=(1, 0, 0))

        proj_op = odl.tomo.RayTransform(phantom_space, geometry)
        # projections_origin = proj_op(phantom_array).asarray()
        projections = proj_op(phantom_array)
        projections_origin = np.array(projections)
        # print("Projections' max value:{},mean value:{}, min value:{}, dtype:{}".format(np.max(projections_origin),
        #                                                                                np.mean(projections_origin),
        #                                                                                np.min(projections_origin),
        #                                                                                projections_origin.dtype))

        projections_without_noise = np.rot90(projections_origin, 1, axes=(1, 2))
        img_drr_without_noise = sitk.GetImageFromArray(projections_without_noise)
        img_drr_without_noise.SetOrigin(phantom.GetOrigin())
        img_drr_without_noise.SetSpacing([(detector_side[0] * 1000 / detector_size[0]) * sad / sid,
                                          (detector_side[1] * 1000 / detector_size[1]) * sad / sid, 1])
        sitk.WriteImage(img_drr_without_noise, os.path.join(path_read, 'DRR_no_noise.nii'))


        projections_noise_poisson = add_poisson_noise_to_projection(projections_without_noise, num_photons=100000,
                                                            dtype=np.float32)

        img_drr_noise_poisson = sitk.GetImageFromArray(projections_noise_poisson)
        img_drr_noise_poisson.SetOrigin(phantom.GetOrigin())
        img_drr_noise_poisson.SetSpacing([(detector_side[0] * 1000 / detector_size[0]) * sad / sid,
                                  (detector_side[1] * 1000 / detector_size[1]) * sad / sid, 1])
        sitk.WriteImage(img_drr_noise_poisson, os.path.join(path_read, 'DRR_poisson_noise.nii'))


        projections_noise_all= add_gaussian_noise_to_projection(projections_noise_poisson, I0=1e5, sigma=10, clip_min=1e-3)

        img_drr_noise_all = sitk.GetImageFromArray(projections_noise_all)
        img_drr_noise_all.SetOrigin(phantom.GetOrigin())
        img_drr_noise_all.SetSpacing([(detector_side[0] * 1000 / detector_size[0]) * sad / sid,
                                  (detector_side[1] * 1000 / detector_size[1]) * sad / sid, 1])
        sitk.WriteImage(img_drr_noise_all, os.path.join(path_read, 'DRR_all_noise.nii'))

        back_projector_without_noise = proj_op.adjoint
        time_start_for_BP_without_noise=time.time()
        reconstruction_without_noise = back_projector_without_noise(projections).asarray()
        time_end_for_BP_without_noise = time.time()
        print("Patient_num: {} Augmented_Phase: {} BP_time: {}".format(patient_num,i,time_end_for_BP_without_noise-time_start_for_BP_without_noise))
        print("projections's max value:{}, mean value:{}, min value:{}, dtype:{}".format(
            np.max(projections),
            np.mean(projections),
            np.min(projections),
            projections.dtype))
        print("Reconstruction_without_noise's max value:{}, mean value:{}, min value:{}, dtype:{}".format(
            np.max(reconstruction_without_noise),
            np.mean(reconstruction_without_noise),
            np.min(reconstruction_without_noise),
            reconstruction_without_noise.dtype))
        img_Recons_without_noise = sitk.GetImageFromArray(reconstruction_without_noise)
        img_Recons_without_noise.SetSpacing(phantom.GetSpacing())
        img_Recons_without_noise.SetOrigin(phantom.GetOrigin())
        sitk.WriteImage(img_Recons_without_noise, os.path.join(path_read, "recons_no_noise.nii"))

        back_projector_poisson_noise = proj_op.adjoint
        # time_start_for_FBP_noise=time.time()
        reconstruction_poisson_noise = back_projector_poisson_noise(np.rot90(projections_noise_poisson, 3, axes=(1, 2))).asarray()
        # time_end_for_FBP_noise = time.time()
        # recon_ray_trafo_noise = odl.tomo.RayTransform(phantom_space, geometry)
        # print("FBP_noise_time: {}".format(time_start_for_FBP_noise-time_end_for_FBP_noise))
        # back_projector_noise = odl.tomo.fbp_op(recon_ray_trafo_noise, filter_type='Hann',
        #                                                frequency_scaling=0.8)
        # reconstruction_noise = back_projector_noise(np.rot90(projections_noise, 3, axes=(1, 2))).asarray()
        img_Recons_poisson_noise = sitk.GetImageFromArray(reconstruction_poisson_noise)
        img_Recons_poisson_noise.SetSpacing(phantom.GetSpacing())
        img_Recons_poisson_noise.SetOrigin(phantom.GetOrigin())
        sitk.WriteImage(img_Recons_poisson_noise, os.path.join(path_read, "recons_poisson_noise.nii"))

        back_projector_all_noise = proj_op.adjoint
        # time_start_for_FBP_noise=time.time()
        reconstruction_all_noise = back_projector_all_noise(np.rot90(projections_noise_all, 3, axes=(1, 2))).asarray()
        img_Recons_all_noise = sitk.GetImageFromArray(reconstruction_all_noise)
        img_Recons_all_noise.SetSpacing(phantom.GetSpacing())
        img_Recons_all_noise.SetOrigin(phantom.GetOrigin())
        sitk.WriteImage(img_Recons_all_noise, os.path.join(path_read, "recons_all_noise.nii"))

    time_end_for_one_patient = time.time()
    processed_datanum += 1
    print("Processed data_num: {}/{}, time spent:{}".format(processed_datanum, len(dir_list),
                                                            time_end_for_one_patient - time_start_for_one_patient))
