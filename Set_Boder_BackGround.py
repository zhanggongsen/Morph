import os
import SimpleITK as sitk
import numpy as np
# Boder_Background_number=5

root_dir = r"Processed_4DCT"

# time_phases = [str(i) for i in range(0, 100, 10)]
time_phases = ["00","10","20","30","40","50","60","70","80","90"]
mask_files = ["mask_Body.nii", "mask_GTV.nii", "mask_Lung.nii"]
background_values = {
    "image.nii": -1000,
    "mask_Body.nii": 0,
    "mask_GTV.nii": 0,
    "mask_Lung.nii": 0,
}


def set_border_to_background(image_array, bg_value):
    array = image_array.copy()
    array[:4, :, :] = bg_value
    array[-4:, :, :] = bg_value
    array[:, :4, :] = bg_value
    array[:, -4:, :] = bg_value
    array[:, :, :4] = bg_value
    array[:, :, -4:] = bg_value
    return array


def process_image(file_path, bg_value):
    image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(image)
    modified_array = set_border_to_background(array, bg_value)
    modified_image = sitk.GetImageFromArray(modified_array)
    modified_image.SetSpacing(image.GetSpacing())
    modified_image.SetOrigin(image.GetOrigin())
    modified_image.SetDirection(image.GetDirection())
    sitk.WriteImage(modified_image, file_path)



for patient_id in os.listdir(root_dir):
    patient_path = os.path.join(root_dir, patient_id)
    if not os.path.isdir(patient_path):
        continue

    for phase in time_phases:
        phase_dir = os.path.join(patient_path, f"Cropped_{phase}")
        if not os.path.isdir(phase_dir):
            continue

        img_file = os.path.join(phase_dir, "image.nii")
        if os.path.exists(img_file):
            print(f"Processing {img_file}")
            process_image(img_file, background_values["image.nii"])

        for mask_name in mask_files:
            mask_file = os.path.join(phase_dir, mask_name)
            if os.path.exists(mask_file):
                print(f"Processing {mask_file}")
                process_image(mask_file, background_values[mask_name])
