import os
import pydicom

def read_patient_info_from_ct_series(series_dir):
    patient_id = None
    patient_age = None

    for filename in os.listdir(series_dir):
        filepath = os.path.join(series_dir, filename)
        try:
            ds = pydicom.dcmread(filepath, stop_before_pixels=True)
            if patient_id is None and hasattr(ds, 'PatientID'):
                patient_id = ds.PatientID
            if patient_age is None and hasattr(ds, 'PatientAge'):
                patient_age = ds.PatientAge

            if patient_id is not None and patient_age is not None:
                break
        except Exception:
            continue

    return patient_id, patient_age


series_path = "00000001/0"
pid, age = read_patient_info_from_ct_series(series_path)
print("Patient ID:", pid)
print("Patient AGE:", age if age is not None else "NA")
