import os
import SimpleITK as sitk
import numpy as np
import copy
import shutil

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

# ----------- Main Pipeline -----------------------------------------------
Processed_root = r"Processed_4DCT"
Augmented_root = r"Augmented_4DCT"
Patient_list = os.listdir(Augmented_root)


def main():
    for patient in Patient_list:
        print("Processing patient: {}".format(patient))
        patient_root = os.path.join(Augmented_root, patient)
        phase_list = os.listdir(patient_root)
        for phase in phase_list:
            phase_root = os.path.join(patient_root, phase)


            print("    Augmented phase: {}".format(phase))
            origin_phase=phase.split('_')[0]
            path_processed=os.path.join(os.path.join(Processed_root,patient),"Cropped_{}".format(origin_phase))
            CT_processed=sitk.ReadImage(os.path.join(path_processed,"image.nii"))
            mask_Body_processed = sitk.ReadImage(os.path.join(path_processed, "mask_Body.nii"))

            mask_Lung_processed = sitk.ReadImage(os.path.join(path_processed, "mask_Lung.nii"))

            mask_GTV_processed = sitk.ReadImage(os.path.join(path_processed, "mask_GTV.nii"))

            DVF_origin = sitk.ReadImage(os.path.join(phase_root,"DVF.nii"))  # Assuming stored as a vector field
            DVF_origin = sitk.Cast(DVF_origin, sitk.sitkVectorFloat64)

            DVF_clipped=sitk.Clamp(DVF_origin,lowerBound=-8.0, upperBound=8.0)
            # array_DVF = sitk.GetArrayFromImage(DVF_clipped)

            jacobian = sitk.DisplacementFieldJacobianDeterminant(DVF_clipped)
            jacobian_array=sitk.GetArrayFromImage(jacobian)
            # folding_voxel_number=np.sum(jacobian_array <= 0)
            has_folding = np.any(jacobian_array <= 0)

            if has_folding:
                print("        ********** folding exists **********")
                DVF_clipped = smooth_dvf_until_valid(DVF_clipped, max_iterations=100, sigma=0.5, threshold=0.1, verbose=True)
            else:
                print("        ********** no folding **********")
            jacobian_new = sitk.DisplacementFieldJacobianDeterminant(DVF_clipped)
            jacobian_array_new = sitk.GetArrayFromImage(jacobian_new)
            has_folding_new = np.any(jacobian_array_new <= 0)
            if has_folding_new:
                shutil.rmtree(phase_root)
                continue

            else:
                pass
            sitk.WriteImage(jacobian_new, os.path.join(phase_root, "jacobian.nii"))
            # array_DVF = sitk.GetArrayFromImage(DVF_clipped)

            # Apply DVF_Compose to CTs and masks, and save all images
            CT_Augmented = apply_dvf_to_image(CT_processed, DVF_clipped, is_mask=False)
            sitk.WriteImage(CT_Augmented, os.path.join(phase_root,"image.nii"))

            mask_Body_Augmented = apply_dvf_to_image(mask_Body_processed, DVF_clipped, is_mask=True)
            sitk.WriteImage(mask_Body_Augmented,  os.path.join(phase_root,"mask_Body.nii"))
            mask_Lung_Augmented = apply_dvf_to_image(mask_Lung_processed, DVF_clipped, is_mask=True)
            sitk.WriteImage(mask_Lung_Augmented, os.path.join(phase_root, "mask_Lung.nii"))
            mask_GTV_Augmented = apply_dvf_to_image(mask_GTV_processed, DVF_clipped, is_mask=True)
            sitk.WriteImage(mask_GTV_Augmented, os.path.join(phase_root, "mask_GTV.nii"))

            # Save DVF
            sitk.WriteImage(DVF_clipped, os.path.join(phase_root,"DVF.nii"))


if __name__ == "__main__":
    main()