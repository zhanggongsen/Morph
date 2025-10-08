import os
import re
import SimpleITK as sitk
import traceback
import copy
import numpy as np


def apply_deformation(orig_img, transform, file_type):

    if file_type == "image":
        interpolator = sitk.sitkLinear
        default_value = -1000.0
    else:
        interpolator = sitk.sitkNearestNeighbor
        default_value = 0.0


    deformed_img = sitk.Resample(
        orig_img,
        orig_img,
        transform,
        interpolator,
        default_value,
        orig_img.GetPixelID()
    )

    return deformed_img

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

augmented_base_dir = r"Augmented_4DCT"
processed_base_dir = r"Processed_4DCT"

if not os.path.exists(augmented_base_dir):
    raise FileNotFoundError(f" path not exists: {augmented_base_dir}")
if not os.path.exists(processed_base_dir):
    raise FileNotFoundError(f" path not exists: {processed_base_dir}")

for subdir in os.listdir(augmented_base_dir):
    subdir_path = os.path.join(augmented_base_dir, subdir)

    if not os.path.isdir(subdir_path):
        continue

    if subdir.endswith("_VAE"):
        match = re.match(r"^(\d{2})", subdir)
        if not match:
            continue
        timepoint = match.group(1)

        orig_dir = os.path.join(processed_base_dir, f"Cropped_{timepoint}")
        if not os.path.exists(orig_dir):
            continue

        dvf_path = os.path.join(subdir_path, "DVF.nii")
        if not os.path.exists(dvf_path):
            continue

        orig_files = {
            "image": os.path.join(orig_dir, "image.nii"),
            "mask_Body": os.path.join(orig_dir, "mask_Body.nii"),
            "mask_GTV": os.path.join(orig_dir, "mask_GTV.nii"),
            "mask_Lung": os.path.join(orig_dir, "mask_Lung.nii")
        }

        missing_files = [name for name, path in orig_files.items() if not os.path.exists(path)]
        if missing_files:
            continue

        try:
            dvf = sitk.ReadImage(dvf_path)
            dvf = sitk.Cast(dvf, sitk.sitkVectorFloat64)

            jacobian = sitk.DisplacementFieldJacobianDeterminant(dvf)
            jacobian_array = sitk.GetArrayFromImage(jacobian)
            has_folding = np.any(jacobian_array <= 0)
            if has_folding:
                print("    ********** folding exists **********")
                dvf = smooth_dvf_until_valid(dvf, max_iterations=30, sigma=0.5, threshold=0.1,
                                                    verbose=True)
            else:
                print("    ********** no folding **********")
            jacobian_new = sitk.DisplacementFieldJacobianDeterminant(dvf)
            sitk.WriteImage(jacobian_new, os.path.join(subdir_path, "jacobian.nii"))

            sitk.WriteImage(dvf, os.path.join(subdir_path, "DVF.nii"))

            transform = sitk.DisplacementFieldTransform(dvf)

            for file_type, orig_path in orig_files.items():
                orig_img = sitk.ReadImage(orig_path)

                deformed_img = apply_deformation(orig_img, transform, file_type)

                output_path = os.path.join(subdir_path, os.path.basename(orig_path))
                sitk.WriteImage(deformed_img, output_path)
                print(f"saving: {output_path}")

        except Exception as e:
            print(f"errors in processing {subdir}: {str(e)}")
            traceback.print_exc()
            continue