import SimpleITK as sitk
import numpy as np


def log_dvf(dvf_img):
    inv_dvf = sitk.InvertDisplacementField(dvf_img,
                                           maximumNumberOfIterations=50,
                                           meanErrorToleranceThreshold=1e-5)

    inv_array = sitk.GetArrayFromImage(inv_dvf)
    svf_array = -1.0 * inv_array

    svf = sitk.GetImageFromArray(svf_array.astype(np.float32))
    svf.CopyInformation(dvf_img)
    return svf


def exp_svf(svf, squaring_steps=6):
    svf_array = sitk.GetArrayFromImage(svf)
    scaled_svf_array = svf_array / (2 ** squaring_steps)

    scaled_svf = sitk.GetImageFromArray(scaled_svf_array.astype(np.float32))
    scaled_svf.CopyInformation(svf)

    disp = scaled_svf

    for _ in range(squaring_steps):
        warped = sitk.Warp(disp, disp, interpolator=sitk.sitkLinear)

        warped.CopyInformation(disp)

        disp = sitk.Add(disp, warped)

    return disp



def svf_interpolate_precise(dvf_path1, dvf_path2, alpha=0.5, squaring_steps=6):

    dvf1 = sitk.Cast(sitk.ReadImage(dvf_path1, sitk.sitkVectorFloat32), sitk.sitkVectorFloat32)
    dvf2 = sitk.Cast(sitk.ReadImage(dvf_path2, sitk.sitkVectorFloat32), sitk.sitkVectorFloat32)

    # 1. log
    svf1 = log_dvf(dvf1)
    svf2 = log_dvf(dvf2)
    # svf1 = log_dvf_fixed_point(dvf1, max_iter=20, tol=1e-4, squaring_steps=6, verbose=True)
    # svf2 = log_dvf_fixed_point(dvf2, max_iter=20, tol=1e-4, squaring_steps=6, verbose=True)

    # 2. interp
    svf1_array = sitk.GetArrayFromImage(svf1)
    svf2_array = sitk.GetArrayFromImage(svf2)
    svf_interp_array = (1 - alpha) * svf1_array + alpha * svf2_array

    svf_interp = sitk.GetImageFromArray(svf_interp_array.astype(np.float32))
    svf_interp.CopyInformation(svf1)

    # 3. exp
    dvf_interp = exp_svf(svf_interp, squaring_steps)

    # 4. Jacobian check
    jacobian = sitk.DisplacementFieldJacobianDeterminant(dvf_interp)
    has_folding = np.any(sitk.GetArrayFromImage(jacobian) <= 0)

    return svf1,svf2,svf_interp, dvf_interp, jacobian, has_folding


if __name__ == "__main__":
    dvf_path1 = r"DVF_0.nii"
    dvf_path2 = r"DVF_1.nii"

    alpha = 0.5

    svf_1,svf_2,svf_Interp, dvf_interp, jac, folding = svf_interpolate_precise(dvf_path1, dvf_path2, alpha)

    sitk.WriteImage(svf_1, r"svf_1.nii")
    sitk.WriteImage(svf_2, r"svf_2.nii")
    sitk.WriteImage(svf_Interp, r"svf_Interp.nii")
    sitk.WriteImage(dvf_interp, r"DVF_interp.nii")
    sitk.WriteImage(jac, r"Jacobian_interp.nii")

    if folding:
        print("Folding exists (Jacobian ≤ 0)")
    else:
        print("Success, no folding")
