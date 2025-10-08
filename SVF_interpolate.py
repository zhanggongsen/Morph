import SimpleITK as sitk
import numpy as np
import os
import random
import copy

def compose_dvf(dvf1, dvf2):
    if dvf1.GetSpacing() != dvf2.GetSpacing() or dvf1.GetOrigin() != dvf2.GetOrigin():
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(dvf2)
        resampler.SetInterpolator(sitk.sitkLinear)
        dvf1 = resampler.Execute(dvf1)

    warped = sitk.Warp(dvf1, dvf2,
                       outputOrigin=dvf2.GetOrigin(),
                       outputSpacing=dvf2.GetSpacing(),
                       outputDirection=dvf2.GetDirection())

    warped.CopyInformation(dvf2)

    return sitk.Add(dvf2, warped)


def exp_svf(svf, squaring_steps=7):
    disp = sitk.Image(svf.GetSize(), sitk.sitkVectorFloat32)
    disp.CopyInformation(svf)

    svf_array = sitk.GetArrayFromImage(svf)
    scaled_svf_array = svf_array / (2 ** squaring_steps)
    scaled_svf = sitk.GetImageFromArray(scaled_svf_array.astype(np.float32))
    scaled_svf.CopyInformation(svf)

    disp = scaled_svf

    for _ in range(squaring_steps):
        disp = compose_dvf(disp, disp)

    return disp


def vector_negative(vector_image):
    array = sitk.GetArrayFromImage(vector_image)
    negative_array = -array
    negative_image = sitk.GetImageFromArray(negative_array.astype(np.float32))
    negative_image.CopyInformation(vector_image)
    return negative_image


def log_dvf_fixed_point(dvf, max_iter=10, tol=1e-3, squaring_steps=7, verbose=False):
    id_dvf = sitk.Image(dvf.GetSize(), sitk.sitkVectorFloat32)
    id_dvf.CopyInformation(dvf)

    svf = sitk.Image(dvf.GetSize(), sitk.sitkVectorFloat32)
    svf.CopyInformation(dvf)

    for i in range(max_iter):
        # exp_svf_disp = exp_svf(svf, squaring_steps)

        neg_svf = vector_negative(svf)
        exp_neg_svf = exp_svf(neg_svf, squaring_steps)

        # residual_field = dvf ∘ exp(-svf)
        residual_field = compose_dvf(dvf, exp_neg_svf)

        # residual = residual_field - id_dvf
        residual = sitk.Subtract(residual_field, id_dvf)

        # svf = svf + residual
        prev_svf = svf
        svf = sitk.Add(svf, residual)

        diff = sitk.Subtract(svf, prev_svf)
        diff_norm_arr = sitk.GetArrayFromImage(diff)
        max_update = np.max(np.linalg.norm(diff_norm_arr, axis=-1))

        if verbose:
            print(f"            Iteration {i + 1}/{max_iter}: Max update = {max_update:.6f}")

        if max_update < tol:
            if verbose:
                print(f"        Converged after {i + 1} iterations")
            break

    return svf


def svf_interpolate_precise(dvf_path1, dvf_path2, alpha=0.5, squaring_steps=7):
    dvf1 = sitk.Cast(sitk.ReadImage(dvf_path1), sitk.sitkVectorFloat32)
    dvf2 = sitk.Cast(sitk.ReadImage(dvf_path2), sitk.sitkVectorFloat32)

    if dvf1.GetSize() != dvf2.GetSize() or dvf1.GetSpacing() != dvf2.GetSpacing():
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(dvf1)
        resampler.SetInterpolator(sitk.sitkLinear)
        dvf2 = resampler.Execute(dvf2)

    print("        Calculating SVF for first DVF...")
    svf1 = log_dvf_fixed_point(dvf1, squaring_steps=squaring_steps, verbose=True)

    print("        Calculating SVF for second DVF...")
    svf2 = log_dvf_fixed_point(dvf2, squaring_steps=squaring_steps, verbose=True)

    svf1_arr = sitk.GetArrayFromImage(svf1)
    svf2_arr = sitk.GetArrayFromImage(svf2)
    svf_interp_arr = (1 - alpha) * svf1_arr + alpha * svf2_arr

    svf_interp = sitk.GetImageFromArray(svf_interp_arr.astype(np.float32))
    svf_interp.CopyInformation(svf1)

    print("        Calculating interpolated DVF from SVF...")
    dvf_interp = exp_svf(svf_interp, squaring_steps)

    return svf1, svf2, svf_interp, dvf_interp


def find_DVF_pair(string_list):

    a,b=random.sample(string_list,2)

    return a, b


def apply_dvf_to_image(image, dvf, is_mask=False):
    dvf=sitk.Cast(dvf, sitk.sitkVectorFloat64)
    transform = sitk.DisplacementFieldTransform(dvf)

    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    default_value = 0 if is_mask else -1000

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
                phase_list=[s for s in phase_list_all if s.startswith('{}_'.format(origin_phase))]
                phase_A,phase_B=find_DVF_pair(phase_list)
                print("    Processing Phases: {} and {}".format(phase_A,phase_B))
                path_A=os.path.join(os.path.join(patient_root,phase_A),"DVF.nii")
                path_B = os.path.join(os.path.join(patient_root, phase_B), "DVF.nii")

                svf_1, svf_2, svf_Interp, dvf_interp= svf_interpolate_precise(path_A, path_B, alpha)

                save_path = os.path.join(patient_root, "{}_{}_SVF".format(phase_A,phase_B[3:]))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                jacobian = sitk.DisplacementFieldJacobianDeterminant(dvf_interp)
                jacobian_array = sitk.GetArrayFromImage(jacobian)
                # folding_voxel_number=np.sum(jacobian_array <= 0)
                has_folding = np.any(jacobian_array <= 0)
                if has_folding:
                    print("    ********** folding exists **********")
                    dvf_interp = smooth_dvf_until_valid(dvf_interp, max_iterations=30, sigma=0.5, threshold=0.1,verbose=True)
                else:
                    print("    ********** no folding **********")
                jacobian_new = sitk.DisplacementFieldJacobianDeterminant(dvf_interp)
                sitk.WriteImage(jacobian_new, os.path.join(save_path, "jacobian.nii"))

                sitk.WriteImage(svf_1, os.path.join(save_path, "svf_1.nii"))
                sitk.WriteImage(svf_2, os.path.join(save_path, "svf_2.nii"))
                sitk.WriteImage(svf_Interp, os.path.join(save_path, "svf_Interp.nii"))
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




