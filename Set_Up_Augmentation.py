import os

import SimpleITK as sitk
import numpy as np
import random
import copy


# ----------- Generate random translation and rotation ------------
def generate_random_transform():
    # Translation in mm [-2, 2]
    translation = [random.uniform(-1.0, 1.0) for _ in range(3)]

    # Rotation in degrees [-1, 1]
    rotation = [random.uniform(-1, 1) for _ in range(3)]  # Around x, y, z

    print(f"    Random translation (mm): {translation}")
    print(f"    Random rotation (degrees): {rotation}")
    return translation, rotation


# ----------- Build DVF_1 from the transform -----------------------
def build_dvf_from_transform(reference_image, translation, rotation_deg):
    rotation_rad = [np.deg2rad(r) for r in rotation_deg]

    transform = sitk.Euler3DTransform()
    transform.SetCenter(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))
    transform.SetRotation(*rotation_rad)
    transform.SetTranslation(translation)

    displacement_field = sitk.TransformToDisplacementField(transform,
                                                           sitk.sitkVectorFloat64,
                                                           reference_image.GetSize(),
                                                           reference_image.GetOrigin(),
                                                           reference_image.GetSpacing(),
                                                           reference_image.GetDirection())
    return sitk.Cast(displacement_field, sitk.sitkVectorFloat64)


# ----------- Compose DVF_0 and DVF_1 to get DVF_2 ----------------
def compose_dvfs(dvf_0, dvf_1):
    """
    CT_0(x) = CT_1( x + DVF_Compose(x) )
    DVF_Compose(x) = DVF_0(x) + DVF_1( x + DVF_0(x) )复合两个形变矢量场(DVF)
    """

    # Step 1: Convert dvf_0 to a transform
    dvf_0_copy=copy.deepcopy(dvf_0)
    transform = sitk.DisplacementFieldTransform(dvf_0)
    # print("dvf_0_copy size:", dvf_0_copy.GetSize())
    # print("dvf_0_copy pixel type:", dvf_0_copy.GetPixelIDTypeAsString())
    # print("dvf_0_copy component dimension:", dvf_0_copy.GetNumberOfComponentsPerPixel())
    # print("transform size:", transform.GetSize())
    # print("transform pixel type:", transform.GetPixelIDTypeAsString())
    # print("transform component dimension:", transform.GetNumberOfComponentsPerPixel())

    # Step 2: Warp dvf_2 using dvf_1 (sample dvf_2 at x + dvf_1(x))
    dvf_1_warped = sitk.Resample(dvf_1, dvf_0_copy, transform, sitk.sitkLinear,
                                    0.0, dvf_1.GetPixelID())
    # print("dvf_1 size:", dvf_1.GetSize())
    # print("dvf_1 pixel type:", dvf_1.GetPixelIDTypeAsString())
    # print("dvf_1 component dimension:", dvf_1.GetNumberOfComponentsPerPixel())
    # print("dvf_1_warped size:", dvf_1_warped.GetSize())
    # print("dvf_1_warped pixel type:", dvf_1_warped.GetPixelIDTypeAsString())
    # print("dvf_1_warped component dimension:", dvf_1_warped.GetNumberOfComponentsPerPixel())
    # Step 3: Add warped dvf_1 and dvf_0
    composed_dvf = sitk.Add(dvf_0_copy,dvf_1_warped)
    return composed_dvf

# ----------- Step 4: Apply DVF_2 to CT_a ---------------------------------
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
            if not phase_root.endswith("_Setup"):
                print("    Augmented phase: {}".format(phase))
                origin_phase=phase.split('_')[0]
                path_processed=os.path.join(os.path.join(Processed_root,patient),"Cropped_{}".format(origin_phase))
                CT_processed=sitk.ReadImage(os.path.join(path_processed,"image.nii"))
                mask_Body_processed = sitk.ReadImage(os.path.join(path_processed, "mask_Body.nii"))
                mask_Lung_processed = sitk.ReadImage(os.path.join(path_processed, "mask_Lung.nii"))
                mask_GTV_processed = sitk.ReadImage(os.path.join(path_processed, "mask_GTV.nii"))

                DVF_origin = sitk.ReadImage(os.path.join(phase_root,"DVF.nii"))  # Assuming stored as a vector field
                DVF_origin = sitk.Cast(DVF_origin, sitk.sitkVectorFloat64)
                save_path = os.path.join(patient_root, "{}_Setup".format(phase[:-4]))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                # Step 1: Random motion
                translation, rotation = generate_random_transform()
                # Save Random motion to txt file
                with open(os.path.join(save_path, "weights.txt"), 'w') as f:
                    f.write(f"translation: {translation}\nrotation: {rotation}\n")

                # Step 2: Generate DVF_Setup
                DVF_Setup = build_dvf_from_transform(CT_processed, translation, rotation)
                # print("DVF_Setup size:", DVF_Setup.GetSize())
                # print("DVF_Setup pixel type:", DVF_Setup.GetPixelIDTypeAsString())
                # print("DVF_Setup component dimension:", DVF_Setup.GetNumberOfComponentsPerPixel())

                # Step 3: Compose DVF_origin and DVF_Setup -> DVF_Compose

                DVF_Compose = compose_dvfs(DVF_origin, DVF_Setup)
                jacobian = sitk.DisplacementFieldJacobianDeterminant(DVF_Compose)
                jacobian_array=sitk.GetArrayFromImage(jacobian)
                # folding_voxel_number=np.sum(jacobian_array <= 0)
                has_folding = np.any(jacobian_array <= 0)

                if has_folding:
                    print("    ********** folding exists **********")
                    DVF_Compose = smooth_dvf_until_valid(DVF_Compose, max_iterations=100, sigma=0.5, threshold=0.1, verbose=True)
                else:
                    print("    ********** no folding **********")
                jacobian_new = sitk.DisplacementFieldJacobianDeterminant(DVF_Compose)
                sitk.WriteImage(jacobian_new, os.path.join(save_path, "jacobian.nii"))


                # Step 4: Apply DVF_Compose to CTs and masks, and save all images
                CT_Augmented = apply_dvf_to_image(CT_processed, DVF_Compose, is_mask=False)
                sitk.WriteImage(CT_Augmented, os.path.join(save_path,"image.nii"))

                mask_Body_Augmented = apply_dvf_to_image(mask_Body_processed, DVF_Compose, is_mask=True)
                sitk.WriteImage(mask_Body_Augmented,  os.path.join(save_path,"mask_Body.nii"))
                mask_Lung_Augmented = apply_dvf_to_image(mask_Lung_processed, DVF_Compose, is_mask=True)
                sitk.WriteImage(mask_Lung_Augmented, os.path.join(save_path, "mask_Lung.nii"))
                mask_GTV_Augmented = apply_dvf_to_image(mask_GTV_processed, DVF_Compose, is_mask=True)
                sitk.WriteImage(mask_GTV_Augmented, os.path.join(save_path, "mask_GTV.nii"))


                # Save DVF
                sitk.WriteImage(DVF_Setup, os.path.join(save_path,"DVF_Setup.nii"))
                sitk.WriteImage(DVF_Compose, os.path.join(save_path,"DVF.nii"))


if __name__ == "__main__":
    main()