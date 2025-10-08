import os
import pydicom
import numpy as np
from pathlib import Path


def get_ct_info(dicom_dir):

    try:

        dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
        if not dicom_files:
            return None, None, None, None

        first_dicom = pydicom.dcmread(os.path.join(dicom_dir, dicom_files[0]), stop_before_pixels=True)
        pixel_spacing = first_dicom.PixelSpacing if hasattr(first_dicom, 'PixelSpacing') else [1.0, 1.0]
        slice_thickness = first_dicom.SliceThickness if hasattr(first_dicom, 'SliceThickness') else 1.0
        spacing = [float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness)]

        rows = first_dicom.Rows if hasattr(first_dicom, 'Rows') else 0
        columns = first_dicom.Columns if hasattr(first_dicom, 'Columns') else 0

        num_slices = len(dicom_files)
        size = [int(rows), int(columns), int(num_slices)]

        manufacturer = first_dicom.Manufacturer if hasattr(first_dicom, 'Manufacturer') else "Unknown"
        model = first_dicom.ManufacturerModelName if hasattr(first_dicom, 'ManufacturerModelName') else "Unknown"

        return spacing, size, manufacturer, model
    except Exception as e:
        print(f"Error processing {dicom_dir}: {str(e)}")
        return None, None, None, None


def main(root_dir):
    root_path = Path(r"anon")

    for patient_dir in root_path.iterdir():
        if patient_dir.is_dir():
            print(f"\nPatient: {patient_dir.name}")

            for phase_dir in patient_dir.iterdir():
                if phase_dir.is_dir():
                    print(f"  Phase: {phase_dir.name}")
                    spacing, size, manufacturer, model = get_ct_info(phase_dir)

                    if spacing and size:
                        print(f"    Spacing (x, y, z): {spacing} mm")
                        print(f"    Size (rows, columns, slices): {size}")
                        print(f"    Manufacturer: {manufacturer}")
                        print(f"    Model: {model}")
                    else:
                        print(f"    No valid CT data found in {phase_dir}")


if __name__ == "__main__":
    root_directory = "data_directory"
    main(root_directory)