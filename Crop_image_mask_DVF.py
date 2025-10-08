import os
import SimpleITK as sitk

def crop_image(image, lower_crop, upper_crop):
    return sitk.Crop(image, lower_crop, upper_crop)

def process_nii_file(filepath, is_vector=False):
    image = sitk.ReadImage(filepath, sitk.sitkVectorFloat64 if is_vector else sitk.sitkFloat32)
    size = image.GetSize()

    # DVF: shape = [x, y, z, 3]
    if is_vector:
        # For vector images (e.g., DVF), last dim is vector length (not spatial)
        lower_crop = [4, 4, 4]
        upper_crop = [4, 4, 4]
        cropped = sitk.Crop(image, lower_crop, upper_crop)
    else:
        lower_crop = [4, 4, 4]
        upper_crop = [4, 4, 4]
        cropped = sitk.Crop(image, lower_crop, upper_crop)

    sitk.WriteImage(cropped, filepath, True)  # Overwrite

def main():
    root_dir = r"Processed_4DCT"
    time_points = [f"Cropped_{i}" for i in range(0, 100, 10)]

    for patient_id in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue

        for phase in time_points:
            phase_path = os.path.join(patient_path, phase)
            if not os.path.isdir(phase_path):
                continue

            for file in os.listdir(phase_path):
                file_path = os.path.join(phase_path, file)
                if not file.endswith(".nii"):
                    continue

                if file.startswith("DVF_") and file.endswith(".nii"):
                    # DVF data has vector components
                    print(f"Cropping vector image (DVF): {file_path}")
                    process_nii_file(file_path, is_vector=True)
                elif file in ["image.nii", "mask_Body.nii", "mask_GTV.nii", "mask_Lung.nii"]:
                    print(f"Cropping scalar image: {file_path}")
                    process_nii_file(file_path, is_vector=False)

if __name__ == "__main__":
    main()