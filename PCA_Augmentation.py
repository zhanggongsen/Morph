import os
import numpy as np
import SimpleITK as sitk
from sklearn.decomposition import PCA
import random
import string
from tqdm import tqdm
import copy

# Configuration
raw_data_root = r"E:\VI\KV2CT\4D_Augmentation\Processed_4DCT"
augmented_data_root = r"E:\VI\KV2CT\4D_Augmentation\Augmented_4DCT"
phases = [f"Phase{i}" for i in ["00","10","20","30","40","50","60","70","80","90"]]  # Phase0 to Phase90
num_augmentations = 5 # Number of augmentations per phase
pca_components = 3  # Number of PCA components to keep
hu_range = [-1000, 1500]  # Expected HU value range


def load_dvfs(patient_path, floating_phase):
    """Load all DVFs for a given floating phase using SimpleITK"""
    dvfs = []
    dvf_metadata = None
    cropped_dir = f"Cropped_{floating_phase.split('Phase')[-1]}"
    phase_path = os.path.join(patient_path, cropped_dir)

    # Load all DVFs where this phase is the floating phase
    for fixed_phase in phases:
        if fixed_phase == floating_phase:
            continue
        dvf_filename = f"DVF_{floating_phase}_{fixed_phase}.nii"
        dvf_path = os.path.join(phase_path, dvf_filename)

        if os.path.exists(dvf_path):
            dvf_img = sitk.ReadImage(dvf_path)
            # Convert to vector of float64 if needed
            if dvf_img.GetPixelID() != sitk.sitkVectorFloat64:
                dvf_img = sitk.Cast(dvf_img, sitk.sitkVectorFloat64)
            dvf_data = sitk.GetArrayFromImage(dvf_img)
            dvfs.append(dvf_data)

            # Store metadata from first DVF
            if dvf_metadata is None:
                dvf_metadata = {
                    'origin': dvf_img.GetOrigin(),
                    'spacing': dvf_img.GetSpacing(),
                    'direction': dvf_img.GetDirection()
                }

    return dvfs, dvf_metadata


def apply_dvf_to_image(image, dvf, is_mask=False, hu_range=None):
    """Apply DVF to an image using SimpleITK with HU value preservation"""
    # Convert numpy arrays to SimpleITK images
    if is_mask:
        image_sitk = sitk.GetImageFromArray(image.astype(np.int16))
    else:
        image_sitk = sitk.GetImageFromArray(image.astype(np.float32))

    # Create displacement field image with proper type
    dvf_sitk = sitk.GetImageFromArray(dvf.astype(np.float64), isVector=True)
    dvf_sitk.CopyInformation(image_sitk)

    # Create displacement field transform
    transform = sitk.DisplacementFieldTransform(dvf_sitk)

    # Apply the transform with appropriate interpolation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_sitk)
    resampler.SetInterpolator(sitk.sitkLinear if not is_mask else sitk.sitkNearestNeighbor)
    resampler.SetTransform(transform)

    # For CT images, set default pixel value to -1000 (air)
    if not is_mask and hu_range is not None:
        resampler.SetDefaultPixelValue(hu_range[0])

    deformed_image_sitk = resampler.Execute(image_sitk)
    deformed_array = sitk.GetArrayFromImage(deformed_image_sitk)

    # Clip HU values to valid range for CT images
    if not is_mask and hu_range is not None:
        deformed_array = np.clip(deformed_array, hu_range[0], hu_range[1])

    return deformed_array


def generate_random_weights():
    """Generate random weights in specified ranges"""
    w0 = random.uniform(0.95, 1.05)
    w1 = random.uniform(-1, 1)
    w2 = random.uniform(-1, 1)
    w3 = random.uniform(-1, 1)
    return w0, w1, w2, w3

def smooth_dvf_until_valid(dvf, max_iterations=30, sigma=0.5, threshold=0.1, verbose=True):
    """
    对DVF进行逐步平滑，直到Jacobian determinant > threshold（避免折叠）

    参数:
        dvf (SimpleITK.Image): 输入DVF（形变矢量场）
        max_iterations (int): 最大平滑次数
        sigma (float): 高斯平滑的初始sigma
        threshold (float): 判断折叠的Jacobian下界（默认0）
        verbose (bool): 是否打印每次迭代信息

    返回:
        SimpleITK.Image: 平滑修复后的DVF
    """

    current_dvf = copy.deepcopy(dvf)
    for i in range(max_iterations):
        # 计算Jacobian determinant
        jac = sitk.DisplacementFieldJacobianDeterminant(current_dvf)
        jac_array = sitk.GetArrayFromImage(jac)
        num_folded = np.sum(jac_array <= threshold)

        if verbose:
            print(f"      [平滑迭代 {i}] 折叠体素数量: {num_folded}")

        # 若没有折叠，直接返回
        if num_folded == 0:
            if verbose:
                print("      平滑后形变场有效，无折叠")
            return current_dvf

        # 若仍有折叠，执行平滑
        current_dvf = sitk.SmoothingRecursiveGaussian(current_dvf, sigma)

    if verbose:
        print("      警告：超过最大迭代次数，仍存在折叠体素")

    return current_dvf

def process_patient(patient_id):
    """Process a single patient"""
    patient_path = os.path.join(raw_data_root, patient_id)

    for floating_phase in phases:
        print("  moving phase: {}".format(floating_phase))
        # Step 1: Load DVFs and perform PCA
        dvfs, dvf_metadata = load_dvfs(patient_path, floating_phase)

        save_path_DVF_mean=os.path.join(patient_path,f"Cropped_{floating_phase.split('Phase')[-1]}")


        if not dvfs or dvf_metadata is None:
            print(f"No DVFs found for patient {patient_id} phase {floating_phase}")
            continue

        # Convert DVFs to 2D array (samples x features)
        original_shape = dvfs[0].shape
        dvf_array = np.array([dvf.reshape(-1) for dvf in dvfs])

        # Calculate mean DVF (DVF0)
        dvf0 = np.mean(dvf_array, axis=0)
        img_DVF0_mean=sitk.GetImageFromArray(dvf0.reshape(original_shape), isVector=True)
        img_DVF0_mean.SetOrigin(dvf_metadata['origin'])
        img_DVF0_mean.SetSpacing(dvf_metadata['spacing'])
        img_DVF0_mean.SetDirection(dvf_metadata['direction'])
        sitk.WriteImage(img_DVF0_mean, os.path.join(save_path_DVF_mean, "DVF_mean.nii"))



        # Center data for PCA
        centered_dvfs = dvf_array - dvf0
        # centered_dvfs = dvf_array

        # Perform PCA
        pca = PCA(n_components=pca_components)
        principal_components = pca.fit_transform(centered_dvfs) # shape: (9, pca_components)
        principal_dvfs = pca.components_  # shape: (pca_components, N); These are DVF1, DVF2, DVF3
        std_principal_components = np.std(principal_components, axis=0)

        # Get representative DVF for each principal component
        # Option 1: Scale by average projection
        component_dvfs = []
        scale_std=2.0
        for i in range(pca_components):
            dvf = (scale_std* std_principal_components[i]) * principal_dvfs[i] + dvf0
            dvf = dvf.reshape(original_shape)

            # avg_score = np.mean(principal_components[:, i])  # scalar
            # pc_vector = principal_dvfs[i]  # shape: (N,)
            # dvf = (avg_score * pc_vector + dvf0).reshape(original_shape)  # add back mean

            img_DVF = sitk.GetImageFromArray(dvf, isVector=True)
            img_DVF.SetOrigin(dvf_metadata['origin'])
            img_DVF.SetSpacing(dvf_metadata['spacing'])
            img_DVF.SetDirection(dvf_metadata['direction'])
            sitk.WriteImage(img_DVF, os.path.join(save_path_DVF_mean, "DVF_principal_{}.nii".format(i)))

            component_dvfs.append(dvf)

        # 保存平均 Principal DVF
        dvfs_array=np.array(component_dvfs)
        # print(dvfs_array.shape)
        dvfs_mean=np.mean(dvfs_array, axis=0)
        img_DVF_mean=sitk.GetImageFromArray(dvfs_mean.reshape(original_shape), isVector=True)
        img_DVF_mean.SetOrigin(dvf_metadata['origin'])
        img_DVF_mean.SetSpacing(dvf_metadata['spacing'])
        img_DVF_mean.SetDirection(dvf_metadata['direction'])
        sitk.WriteImage(img_DVF_mean, os.path.join(save_path_DVF_mean, "DVF_principal_mean.nii"))
        #
        # # 保存每个 Principal DVF
        # for i in range(len(principal_dvfs)):
        #     img_DVF = sitk.GetImageFromArray(principal_dvfs[i].reshape(original_shape), isVector=True)
        #     img_DVF.SetOrigin(dvf_metadata['origin'])
        #     img_DVF.SetSpacing(dvf_metadata['spacing'])
        #     img_DVF.SetDirection(dvf_metadata['direction'])
        #     sitk.WriteImage(img_DVF, os.path.join(save_path_DVF_mean, "DVF_principal_{}.nii".format(i)))



        # Create output directory structure
        cropped_dir = f"Cropped_{floating_phase.split('Phase')[-1]}"
        phase_path = os.path.join(patient_path, cropped_dir)

        # Load original images
        image_path = os.path.join(phase_path, "image.nii")
        mask_gtv_path = os.path.join(phase_path, "mask_GTV.nii")
        mask_body_path = os.path.join(phase_path, "mask_Body.nii")
        mask_lung_path = os.path.join(phase_path, "mask_Lung.nii")

        if not all(os.path.exists(p) for p in [image_path, mask_gtv_path, mask_body_path, mask_lung_path]):
            print(f"Missing files for patient {patient_id} phase {floating_phase}")
            continue

        # Load images with proper data types
        image_sitk = sitk.ReadImage(image_path)
        mask_gtv_sitk = sitk.ReadImage(mask_gtv_path)
        mask_body_sitk = sitk.ReadImage(mask_body_path)
        mask_lung_sitk = sitk.ReadImage(mask_lung_path)

        # Convert to numpy arrays
        image_data = sitk.GetArrayFromImage(image_sitk)
        mask_gtv_data = sitk.GetArrayFromImage(mask_gtv_sitk)
        mask_body_data = sitk.GetArrayFromImage(mask_body_sitk)
        mask_lung_data = sitk.GetArrayFromImage(mask_lung_sitk)

        # Verify HU range
        if np.min(image_data) < hu_range[0] or np.max(image_data) > hu_range[1]:
            print(f"Warning: Patient {patient_id} phase {floating_phase} has HU values outside expected range")

        # Store metadata from first image
        image_metadata = {
                'origin': image_sitk.GetOrigin(),
                'spacing': image_sitk.GetSpacing(),
                'direction': image_sitk.GetDirection(),
                'dtype': image_sitk.GetPixelID()
        }

        # Create augmented data for this phase
        for aug_idx in range(num_augmentations):
            # Step 2: Generate random weights
            w0, w1, w2, w3 = generate_random_weights()

            # Step 3: Compute DVF(t)
            dvf_t = w0 * dvf0
            for i, w in enumerate([w1, w2, w3]):
                dvf_t += w * (component_dvfs[i].reshape(-1))

            dvf_t = dvf_t.reshape(original_shape)

            # Apply DVF to images with appropriate settings
            deformed_image = apply_dvf_to_image(image_data, dvf_t, hu_range=hu_range)
            deformed_gtv = apply_dvf_to_image(mask_gtv_data, dvf_t, is_mask=True)
            deformed_body = apply_dvf_to_image(mask_body_data, dvf_t, is_mask=True)
            deformed_lung = apply_dvf_to_image(mask_lung_data, dvf_t, is_mask=True)

            # Step 4: Save augmented data
            # Generate unique 6-digit random string
            while True:
                random_str = ''.join(random.choices(string.digits, k=6))
                aug_dir_name = f"{floating_phase.split('Phase')[-1]}_{random_str}_PCA"
                aug_patient_path = os.path.join(augmented_data_root, patient_id)
                aug_phase_path = os.path.join(aug_patient_path, aug_dir_name)
                if not os.path.exists(aug_phase_path):
                    break

            os.makedirs(aug_phase_path, exist_ok=True)

            # Save weights to txt file
            with open(os.path.join(aug_phase_path, "weights.txt"), 'w') as f:
                f.write(f"w0: {w0}\nw1: {w1}\nw2: {w2}\nw3: {w3}\n")

                # # 检查 Jacobian 是否出现 folding（值 ≤ 0 表示不可逆）
                # jacobian = sitk.DisplacementFieldJacobianDeterminant(dvf_sitk)
                # has_folding = np.any(sitk.GetArrayFromImage(jacobian) <= 0)
                # if has_folding:
                #     print("    ********** {} has folding **********".format(aug_phase_path))
                #     print("    ********** folding exists **********")
                #     dvf_sitk = smooth_dvf_until_valid(dvf_sitk, max_iterations=30, sigma=0.5, threshold=0.1,
                #                                          verbose=True)
                # else:
                #     print("    {} no folding.".format(aug_phase_path))
                #
                # jacobian_new = sitk.DisplacementFieldJacobianDeterminant(dvf_sitk)
                # sitk.WriteImage(jacobian_new, os.path.join(aug_phase_path, "jacobian.nii"))

            # Save DVF(t)
            dvf_sitk = sitk.GetImageFromArray(dvf_t.astype(np.float64), isVector=True)
            dvf_sitk.SetOrigin(dvf_metadata['origin'])
            dvf_sitk.SetSpacing(dvf_metadata['spacing'])
            dvf_sitk.SetDirection(dvf_metadata['direction'])
            sitk.WriteImage(dvf_sitk, os.path.join(aug_phase_path, "DVF.nii"))


            # Save deformed images with original data types
            def save_as_nii(data, metadata, path, is_mask=False):
                if is_mask:
                    img = sitk.GetImageFromArray(data.astype(np.uint8))
                else:
                    # Preserve original CT data type
                    if metadata['dtype'] == sitk.sitkInt16:
                        img = sitk.GetImageFromArray(data.astype(np.int16))
                    else:
                        img = sitk.GetImageFromArray(data.astype(np.float32))

                img.SetOrigin(metadata['origin'])
                img.SetSpacing(metadata['spacing'])
                img.SetDirection(metadata['direction'])
                sitk.WriteImage(img, path)

            save_as_nii(deformed_image, image_metadata, os.path.join(aug_phase_path, "image.nii"))
            save_as_nii(deformed_gtv, image_metadata, os.path.join(aug_phase_path, "mask_GTV.nii"), is_mask=True)
            save_as_nii(deformed_body, image_metadata, os.path.join(aug_phase_path, "mask_Body.nii"), is_mask=True)
            save_as_nii(deformed_lung, image_metadata, os.path.join(aug_phase_path, "mask_Lung.nii"), is_mask=True)



def main():
    # Get all patient directories
    patient_ids = [d for d in os.listdir(raw_data_root)
                   if os.path.isdir(os.path.join(raw_data_root, d)) and d.isdigit() and len(d) == 8]

    # Create output directory if it doesn't exist
    os.makedirs(augmented_data_root, exist_ok=True)

    # Process each patient
    for patient_id in tqdm(patient_ids, desc="Processing patients"):
        print("Processing {}".format(patient_id))
        process_patient(patient_id)


if __name__ == "__main__":
    main()