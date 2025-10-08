import os
import shutil
import pydicom


def parse_phase(series_desc):
    if not series_desc:
        return None
    desc = series_desc.strip().upper()
    if desc in ["MIP", "MINIP", "AVG"]:
        return desc
    if ".0% AMP" in desc:
        try:
            # Extract the part before '.0% AMP'
            num_str = desc.split(".0% AMP")[0].strip()
            # Validate it's a valid phase like 0, 10, 20, ..., 90
            if num_str.isdigit() and int(num_str) % 10 == 0 and 0 <= int(num_str) <= 90:
                return num_str
        except:
            pass
    return None


def process_patient_folder(patient_dir):
    print(f"Processing folder: {patient_dir}")
    dcm_files = [f for f in os.listdir(patient_dir) if f.lower().endswith('.dcm')]

    phase_to_uid = {}
    uid_to_phase = {}

    ct_files_by_phase = {}

    rtstruct_files = []

    for filename in dcm_files:
        filepath = os.path.join(patient_dir, filename)
        try:
            ds = pydicom.dcmread(filepath, stop_before_pixels=True)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        if ds.Modality == "CT":
            series_desc = getattr(ds, 'SeriesDescription', None)
            phase = parse_phase(series_desc)
            if phase:
                series_uid = getattr(ds, 'SeriesInstanceUID', None)
                if series_uid:
                    if phase not in phase_to_uid:
                        phase_to_uid[phase] = series_uid
                        uid_to_phase[series_uid] = phase
                    # Group the file
                    if phase not in ct_files_by_phase:
                        ct_files_by_phase[phase] = []
                    ct_files_by_phase[phase].append(filename)

        elif ds.Modality == "RTSTRUCT":
            rtstruct_files.append(filename)

    # Create phase subfolders and move CT files
    for phase in phase_to_uid:
        phase_subdir = os.path.join(patient_dir, phase)
        os.makedirs(phase_subdir, exist_ok=True)

        for ct_filename in ct_files_by_phase.get(phase, []):
            src = os.path.join(patient_dir, ct_filename)
            dst = os.path.join(phase_subdir, ct_filename)
            shutil.move(src, dst)
            print(f"Moved CT file {ct_filename} to {phase}")

    for filename in rtstruct_files:
        filepath = os.path.join(patient_dir, filename)
        try:
            ds = pydicom.dcmread(filepath, stop_before_pixels=True)

            # Attempt to extract the referenced Series Instance UID
            # Assuming the first ReferencedFrameOfReferenceSequence
            if hasattr(ds, 'ReferencedFrameOfReferenceSequence') and ds.ReferencedFrameOfReferenceSequence:
                ref_frame_seq = ds.ReferencedFrameOfReferenceSequence[0]
                if hasattr(ref_frame_seq, 'RTReferencedStudySequence') and ref_frame_seq.RTReferencedStudySequence:
                    rt_study_seq = ref_frame_seq.RTReferencedStudySequence[0]
                    if hasattr(rt_study_seq, 'RTReferencedSeriesSequence') and rt_study_seq.RTReferencedSeriesSequence:
                        rt_series_seq = rt_study_seq.RTReferencedSeriesSequence[0]
                        series_uid = getattr(rt_series_seq, 'SeriesInstanceUID', None)
                        if series_uid:
                            phase = uid_to_phase.get(series_uid)
                            if phase:
                                phase_subdir = os.path.join(patient_dir, phase)
                                dst = os.path.join(phase_subdir, filename)
                                shutil.move(filepath, dst)
                                print(f"Moved RT-Struct {filename} to {phase}")
                            else:
                                print(f"No matching phase for RT-Struct {filename} with UID {series_uid}")
                        else:
                            print(f"No SeriesInstanceUID found in RT-Struct {filename}")
                    else:
                        print(f"No RTReferencedSeriesSequence in RT-Struct {filename}")
                else:
                    print(f"No RTReferencedStudySequence in RT-Struct {filename}")
            else:
                print(f"No ReferencedFrameOfReferenceSequence in RT-Struct {filename}")

        except Exception as e:
            print(f"Error processing RT-Struct {filename}: {e}")


if __name__ == "__main__":
    primary_dir = r"\anon"
    patient_folders = [d for d in os.listdir(primary_dir) if os.path.isdir(os.path.join(primary_dir, d))]

    if len(patient_folders) != 10:
        print(f"Warning: Found {len(patient_folders)} folders, expected 10.")

    for folder in patient_folders:
        process_patient_folder(os.path.join(primary_dir, folder))