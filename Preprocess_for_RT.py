import os
import time
import SimpleITK as sitk
from datetime import datetime
import numpy as np

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

def register_phases(patient_dir):
    for moving_phase in ['AVG']:
        moving_dir = os.path.join(patient_dir, f'structs_{moving_phase}')
        moving_image_path = os.path.join(moving_dir, 'image.nii')

        if not os.path.exists(moving_image_path):
            continue

        moving_image = sitk.ReadImage(moving_image_path)

        for fixed_phase in ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90']:
            if fixed_phase == moving_phase:
                continue

            fixed_dir = os.path.join(patient_dir, f'structs_{fixed_phase}')
            fixed_image_path = os.path.join(fixed_dir, 'image.nii')

            if not os.path.exists(fixed_image_path):
                continue

            print(f"Patient: {os.path.basename(patient_dir)}, "
                  f"moving phase: Phase{moving_phase}, "
                  f"fixed phase: Phase{fixed_phase}", end="", flush=True)

            start_time = time.time()
            fixed_image = sitk.ReadImage(fixed_image_path)

            elastix = sitk.ElastixImageFilter()
            elastix.LogToConsoleOff()
            elastix.LogToFileOff()
            elastix.SetOutputDirectory("")
            elastix.SetFixedImage(fixed_image)
            elastix.SetMovingImage(moving_image)

            # ==================================================================
            # Euler Transform
            # ==================================================================
            param_rigid = sitk.GetDefaultParameterMap("rigid")

            param_rigid["FixedImageDimension"] = ["3"]
            param_rigid["MovingImageDimension"] = ["3"]
            param_rigid["FixedInternalImagePixelType"] = ["float"]
            param_rigid["MovingInternalImagePixelType"] = ["float"]
            param_rigid["DefaultPixelValue"] = ["-1000"]

            param_rigid["NumberOfResolutions"] = ["4"]
            param_rigid["ImagePyramidSchedule"] = ["8 8 4", "4 4 2", "2 2 1", "1 1 1"]

            param_rigid["MaximumNumberOfIterations"] = ["512"]
            param_rigid["NewSamplesEveryIteration"] = ["true"]

            param_rigid["Metric"] = ["AdvancedMattesMutualInformation"]
            param_rigid["NumberOfHistogramBins"] = ["32"]
            param_rigid["NumberOfSpatialSamples"] = ["2048"]

            param_rigid["LogToConsole"] = ["false"]
            param_rigid["LogToFile"] = ["false"]

            # ==================================================================
            # Affine Transform
            # ==================================================================
            param_affine = sitk.GetDefaultParameterMap("affine")
            param_affine["FixedImageDimension"] = ["3"]
            param_affine["MovingImageDimension"] = ["3"]
            param_affine["FixedInternalImagePixelType"] = ["float"]
            param_affine["MovingInternalImagePixelType"] = ["float"]
            param_affine["DefaultPixelValue"] = ["-1000"]

            param_affine["NumberOfResolutions"] = ["4"]
            param_affine["ImagePyramidSchedule"] = param_rigid["ImagePyramidSchedule"]
            param_affine["MaximumNumberOfIterations"] = ["512"]
            param_affine["NumberOfSpatialSamples"] = ["2048"]

            param_affine["LogToConsole"] = ["false"]
            param_affine["LogToFile"] = ["false"]
            # ==================================================================
            # Non-rigid
            # ==================================================================
            param_bspline = sitk.GetDefaultParameterMap("bspline")

            param_bspline["FixedImageDimension"] = ["3"]
            param_bspline["MovingImageDimension"] = ["3"]
            param_bspline["FixedInternalImagePixelType"] = ["float"]
            param_bspline["MovingInternalImagePixelType"] = ["float"]
            param_bspline["DefaultPixelValue"] = ["-1000"]

            param_bspline["FinalGridSpacingInPhysicalUnits"] = ["10 10 20"]
            param_bspline["GridSpacingSchedule"] = ["16.0 16.0 8.0", "8.0 8.0 4.0", "4.0 4.0 2.0", "2.0 2.0 1.0"]

            param_bspline["Optimizer"] = [
                "AdaptiveStochasticGradientDescent"]
            param_bspline["NumberOfResolutions"] = ["4"]
            param_bspline["ImagePyramidSchedule"] = [
                "8", "8", "2",
                "4", "4", "2",
                "2", "2", "1",
                "1", "1", "1"
            ]
            param_bspline["MaximumNumberOfIterations"] = ["1024"]
            param_bspline["NumberOfSpatialSamples"] = ["2048"]

            param_bspline["FinalBSplineInterpolationOrder"] = ["3"]
            param_bspline["BSplineTransformSplineOrder"] = ["3"]
            param_bspline["Metric1Weight"] = ["1.0"]
            param_bspline["Metric2Weight"] = ["0"]
            # param_bspline["SizeBoundaryPenalty"] = ["0.5"]
            # param_bspline["HowToCombineTransforms"] = ["Compose"]

            # param_bspline["WriteResultImage"] = ["false"]
            param_bspline["LogToConsole"] = ["false"]
            param_bspline["LogToFile"] = ["false"]

            elastix.SetParameterMap(param_rigid)
            elastix.AddParameterMap(param_affine)
            elastix.AddParameterMap(param_bspline)

            elastix.Execute()

            result_image = elastix.GetResultImage()
            transform_parameters = elastix.GetTransformParameterMap()

            # registered_image_path = os.path.join(
            #     moving_dir,
            #     f'Registered_image_Phase{moving_phase}_Phase{fixed_phase}.nii'
            # )
            # sitk.WriteImage(result_image, registered_image_path)

            transformix = sitk.TransformixImageFilter()
            transformix.LogToConsoleOff()
            transformix.LogToFileOff()
            # transformix.SetOutputDirectory("")
            transformix.SetTransformParameterMap(transform_parameters)
            transformix.ComputeDeformationFieldOn()
            transformix.SetMovingImage(moving_image)
            transformix.Execute()

            dvf = transformix.GetDeformationField()
            # dvf_path = os.path.join(
            #     moving_dir,
            #     f'DVF_Phase{moving_phase}_Phase{fixed_phase}.nii'
            # )
            # sitk.WriteImage(dvf, dvf_path)
            GTV_image=sitk.ReadImage(os.path.join(moving_dir, 'mask_GTV.nii'))
            GTV_image_out=apply_dvf_to_image(GTV_image, dvf, is_mask=True)
            sitk.WriteImage(GTV_image_out, os.path.join(fixed_dir, 'mask_GTV.nii'))

            elapsed_time = time.time() - start_time
            print(f", Times: {elapsed_time:.2f} seconds")


def Regis_process_all_patients(root_dir):
    for patient_id in os.listdir(root_dir):
        patient_dir = os.path.join(root_dir, patient_id)
        for phase_i in ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90',"AVG"]:
            if os.path.exists(os.path.join(patient_dir,"structs_{}".format(phase_i), 'mask_External.nii')):
                # path_read_new=path_read.replace('BODY','Body')
                os.rename(os.path.join(patient_dir,"structs_{}".format(phase_i), 'mask_External.nii'),os.path.join(patient_dir,"structs_{}".format(phase_i), 'mask_Body.nii'))
            if os.path.exists(os.path.join(patient_dir,"structs_{}".format(phase_i), 'mask_PTV.nii')):
                # path_read_new=path_read.replace('BODY','Body')
                os.rename(os.path.join(patient_dir,"structs_{}".format(phase_i), 'mask_External.nii'),os.path.join(patient_dir,"structs_{}".format(phase_i), 'mask_GTV.nii'))

        if os.path.isdir(patient_dir) and patient_id.isdigit() and len(patient_id) == 8:
            print(f"\n Patient Processing: {patient_id}")
            register_phases(patient_dir)


if __name__ == "__main__":
    root_dir = r"4DCT"
    Regis_process_all_patients(root_dir)