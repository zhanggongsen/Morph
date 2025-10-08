import SimpleITK as sitk
import numpy as np
import os
import random
import copy
from scipy.interpolate import make_interp_spline


def bspline_interpolate_dvf_3frames(dvf_paths, alpha):
    assert len(dvf_paths) == 3, "3 DVFs at least"
    assert 0.0 <= alpha <= 1.0, "alpha should in range of [0, 1]"

    dvf_imgs = [sitk.ReadImage(p, sitk.sitkVectorFloat32) for p in dvf_paths]
    arrays = [sitk.GetArrayFromImage(img) for img in dvf_imgs]

    t = np.array([0.0, 0.5, 1.0])
    y = np.stack(arrays, axis=0)  # Shape: (3, D, H, W, 3)

    spline = make_interp_spline(t, y, axis=0, k=2)
    interp_array = spline(alpha)  # Shape: (D, H, W, 3)

    interp_dvf = sitk.GetImageFromArray(interp_array.astype(np.float32))
    interp_dvf.CopyInformation(dvf_imgs[1])

    return interp_dvf


def find_DVF_pair(string_list):

    valid_strings = [s for s in string_list if len(s) >= 2]

    # ends_with__1 = [s for s in valid_strings if s.endswith('_1')]  # 以'_1'结尾的组,PCA+Setup增强
    # not_ends_with__1 = [s for s in valid_strings if not s.endswith('_1')]  # 不以'_1'结尾的组,PCA增强
    # has__1_=[s for s in valid_strings if "_1_" in s]  # 含'_1_'的组,对于此批数据，选定一对（PCA+Setup）和PCA，使用SVF增强
    # if not ends_with__1 or not not_ends_with__1 or not has__1_:
    #     return None  # 无法满足条件(1)
    #
    # a = random.choice(ends_with__1)
    #
    # # b_list=[s for s in not_ends_with__1 if s[:2]==a[:2]]
    # b = random.choice(not_ends_with__1)
    #
    # # c_list=[s for s in has__1_ if s[:2]==a[:2]]
    # c=random.choice(has__1_)
    a = random.choice(valid_strings)
    b_list=[s for s in valid_strings if not s==a]
    b = random.choice(b_list)
    c_list=[s for s in b_list if not s==b]
    c=random.choice(c_list)

    return a, b, c


def apply_dvf_to_image(image, dvf, is_mask=False):
    dvf=sitk.Cast(dvf, sitk.sitkVectorFloat64)
    transform = sitk.DisplacementFieldTransform(dvf)

    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    default_value = 0 if is_mask else -1000  # CT: typical air HU value

    resampled = sitk.Resample(image, image, transform, interpolator, default_value, image.GetPixelID())

    if is_mask:
        resampled = sitk.Cast(resampled, sitk.sitkUInt8)
    else:
        resampled = sitk.Cast(resampled, sitk.sitkInt16)

    return resampled

def smooth_dvf_until_valid(dvf, max_iterations=30, sigma=0.5, threshold=0.1, verbose=True):
    current_dvf = copy.deepcopy(dvf)
    for i in range(max_iterations):
        jac = sitk.DisplacementFieldJacobianDeterminant(current_dvf)
        jac_array = sitk.GetArrayFromImage(jac)
        num_folded = np.sum(jac_array <= threshold)

        if verbose:
            print(f"      [smooth iter {i}] folding voxels: {num_folded}")

        if num_folded == 0:
            if verbose:
                print("      no folding exists")
            return current_dvf

        current_dvf = sitk.SmoothingRecursiveGaussian(current_dvf, sigma)

    if verbose:
        print("      warning folding still exists after smoothing")

    return current_dvf

Processed_root = r"Processed_4DCT"
Augmented_root = r"Augmented_4DCT"
Patient_list = os.listdir(Augmented_root)
sample_number_for_one_origin_phase=5
alpha = 0.5

if __name__ == "__main__":
    for patient in Patient_list:
        print("Processing patient: {}".format(patient))
        patient_root = os.path.join(Augmented_root, patient)
        phase_list_all = os.listdir(patient_root)
        for origin_phase in ["00","10","20","30","40","50","60","70","80","90"]:
            for i in range(sample_number_for_one_origin_phase):
                phase_list = [s for s in phase_list_all if s.startswith('{}_'.format(origin_phase))]
                phase_A,phase_B,phase_C=find_DVF_pair(phase_list)
                print("    Processing Phases: {} ,{} and {}".format(phase_A,phase_B,phase_C))
                path_A=os.path.join(os.path.join(patient_root,phase_A),"DVF.nii")
                path_B = os.path.join(os.path.join(patient_root, phase_B), "DVF.nii")
                path_C = os.path.join(os.path.join(patient_root, phase_C), "DVF.nii")

                paths=[path_A,path_B,path_C]
                dvf_interp= bspline_interpolate_dvf_3frames(paths, alpha)

                save_path = os.path.join(patient_root, "{}_{}_{}_Bsp".format(phase_A,phase_B[3:],phase_C[3:]))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                jacobian = sitk.DisplacementFieldJacobianDeterminant(dvf_interp)
                jacobian_array = sitk.GetArrayFromImage(jacobian)
                has_folding = np.any(jacobian_array <= 0)
                if has_folding:
                    print("    ********** folding exists **********")
                    dvf_interp = smooth_dvf_until_valid(dvf_interp, max_iterations=30, sigma=0.5, threshold=0.1,verbose=True)
                else:
                    print("    ********** no folding **********")
                jacobian_new = sitk.DisplacementFieldJacobianDeterminant(dvf_interp)
                sitk.WriteImage(jacobian_new, os.path.join(save_path, "jacobian.nii"))

                sitk.WriteImage(dvf_interp, os.path.join(save_path, "DVF.nii"))

                origin_phase = phase_A.split('_')[0]
                path_processed = os.path.join(os.path.join(Processed_root, patient), "Cropped_{}".format(origin_phase))
                CT_processed = sitk.ReadImage(os.path.join(path_processed, "image.nii"))
                mask_Body_processed = sitk.ReadImage(os.path.join(path_processed, "mask_Body.nii"))
                mask_Lung_processed = sitk.ReadImage(os.path.join(path_processed, "mask_Lung.nii"))
                mask_GTV_processed = sitk.ReadImage(os.path.join(path_processed, "mask_GTV.nii"))

                # Apply DVF_Compose to CTs and masks, and save all images
                CT_Augmented = apply_dvf_to_image(CT_processed, dvf_interp, is_mask=False)
                sitk.WriteImage(CT_Augmented, os.path.join(save_path, "image.nii"))
                mask_Body_Augmented = apply_dvf_to_image(mask_Body_processed, dvf_interp, is_mask=True)
                sitk.WriteImage(mask_Body_Augmented, os.path.join(save_path, "mask_Body.nii"))
                mask_Lung_Augmented = apply_dvf_to_image(mask_Lung_processed, dvf_interp, is_mask=True)
                sitk.WriteImage(mask_Lung_Augmented, os.path.join(save_path, "mask_Lung.nii"))
                mask_GTV_Augmented = apply_dvf_to_image(mask_GTV_processed, dvf_interp, is_mask=True)
                sitk.WriteImage(mask_GTV_Augmented, os.path.join(save_path, "mask_GTV.nii"))




