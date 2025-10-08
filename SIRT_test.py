import os
import numpy as np
import SimpleITK as sitk
import astra
import time

HU_WATER = 0.0
MU_WATER = 0.02
I0 = 1e5
GAUSSIAN_NOISE_SIGMA = 5

SID = 3.7 * 1000
SAD = 2.7 * 1000
detector_size = [0.397 * 1000, 0.298 * 1000]
detector_resolution = [1024, 768]

base_dir = r"Augmented_4DCT"


# def hu_to_mu(hu_array):
#     return (hu_array / 1000 + 1) * MU_WATER
def convert_hu_to_linear_attenuation(hu_array_0,
                                     MU_WATER=0.02, MU_AIR=0.00002, MU_MAX=None):
    hu_array = np.array(hu_array_0)
    if MU_MAX is None:
        # MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER
        MU_MAX = 1500 * (MU_WATER - MU_AIR) / 1000 + MU_WATER
    # convert values
    hu_array *= (MU_WATER - MU_AIR) / 1000
    hu_array += MU_WATER
    hu_array /= MU_MAX
    np.clip(hu_array, 0., 1., out=hu_array)
    return hu_array * MU_MAX

# def add_poisson_noise(projection):
#     transmission = np.exp(-projection)
#     photon_count = I0 * transmission
#     noisy_photon = np.random.poisson(photon_count)
#     noisy_photon[noisy_photon == 0] = 1  # 避免除零
#     return -np.log(noisy_photon / I0)


# def add_gaussian_noise(projection):
#     return projection + np.random.normal(0, GAUSSIAN_NOISE_SIGMA, projection.shape)


def add_poisson_noise_to_projection(data_0, num_photons=1e5, dtype=np.float32):
    data = np.array(data_0)
    data = np.exp(-data) * num_photons
    data = np.random.poisson(data) / num_photons
    # data_test = np.array(data)
    # data_test[data < 0.1 / num_photons] = 0
    data = np.maximum(0.1 / num_photons, data)
    data = -np.log(data)
    data = data.astype(dtype)
    return data

def add_gaussian_noise_to_projection(proj, I0=1e5, sigma=10, clip_min=1e-3):
    I = I0 * np.exp(-proj)
    noise = np.random.normal(0, sigma, I.shape)
    I_noisy = I + noise
    I_noisy = np.clip(I_noisy, clip_min, None)
    return -np.log(I_noisy / I0)

def create_astra_geometry(vol_shape, vol_spacing):

    vol_size_phy = [vol_shape[0] * vol_spacing[0],
                    vol_shape[1] * vol_spacing[1],
                    vol_shape[2] * vol_spacing[2]]

    vol_geom = astra.create_vol_geom(
        vol_shape[1], vol_shape[0], vol_shape[2],  # ASTRA y,x,z
        -vol_size_phy[1] / 2, vol_size_phy[1] / 2,
        -vol_size_phy[0] / 2, vol_size_phy[0] / 2,
        -vol_size_phy[2] / 2, vol_size_phy[2] / 2
    )

    det_spacing = [detector_size[0] / detector_resolution[0],
                   detector_size[1] / detector_resolution[1]]

    angles = np.deg2rad([0, 90])
    proj_geom = astra.create_proj_geom(
        'cone',
        det_spacing[1], det_spacing[0],
        detector_resolution[1], detector_resolution[0],
        angles,
        SAD, SID
    )

    return vol_geom, proj_geom


def project_volume(vol_data, vol_geom, proj_geom):
    vol_id = astra.data3d.create('-vol', vol_geom, vol_data)
    proj_id = astra.data3d.create('-proj3d', proj_geom)

    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['VolumeGeometry'] = vol_geom
    cfg['ProjectionGeometry'] = proj_geom
    cfg['VolumeDataId'] = vol_id
    cfg['ProjectionDataId'] = proj_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    proj_data = astra.data3d.get(proj_id)

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(vol_id)
    astra.data3d.delete(proj_id)

    return proj_data


def reconstruct_sirt(proj_data, vol_geom, proj_geom, iterations=100):
    proj_id = astra.data3d.create('-proj3d', proj_geom, proj_data)
    rec_id = astra.data3d.create('-vol', vol_geom)

    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ProjectionGeometry'] = proj_geom
    cfg['VolumeGeometry'] = vol_geom
    cfg['ProjectionDataId'] = proj_id
    cfg['ReconstructionDataId'] = rec_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, iterations)
    rec_data = astra.data3d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(rec_id)

    return rec_data


def reconstruct_sart(proj_data, vol_geom, proj_geom, iterations=100):
    proj_id = astra.data3d.create('-proj3d', proj_geom, proj_data)
    rec_id = astra.data3d.create('-vol', vol_geom)

    cfg = astra.astra_dict('SART3D_CUDA')
    cfg['ProjectionGeometry'] = proj_geom
    cfg['VolumeGeometry'] = vol_geom
    cfg['ProjectionDataId'] = proj_id
    cfg['ReconstructionDataId'] = rec_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, iterations)
    rec_data = astra.data3d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(rec_id)

    return rec_data


def save_as_nii(data, output_path, reference_img=None):
    img = sitk.GetImageFromArray(data)
    if reference_img:
        img.CopyInformation(reference_img)
    sitk.WriteImage(img, output_path)


for patient in os.listdir(base_dir):
    patient_dir = os.path.join(base_dir, patient)
    if not os.path.isdir(patient_dir):
        continue

    for phase in os.listdir(patient_dir):
        phase_dir = os.path.join(patient_dir, phase)
        if not os.path.isdir(phase_dir):
            continue

        ct_path = os.path.join(phase_dir, 'image.nii')
        if not os.path.exists(ct_path):
            continue

        print(f"Processing: {patient}/{phase}")

        ct_img = sitk.ReadImage(ct_path)
        ct_array = sitk.GetArrayFromImage(ct_img)  # [z,y,x]
        mu_array = convert_hu_to_linear_attenuation(ct_array,
                                     MU_WATER=0.02, MU_AIR=0.00002, MU_MAX=None)

        mu_img = sitk.GetImageFromArray(mu_array)
        mu_img.CopyInformation(ct_img)
        sitk.WriteImage(mu_img, os.path.join(phase_dir, 'image_miu.nii'))

        # vol_shape = mu_array.shape  # (z,y,x)
        # vol_spacing = ct_img.GetSpacing()[::-1]

        # vol_geom, proj_geom = create_astra_geometry(vol_shape, vol_spacing)
        vol_geom, proj_geom = create_astra_geometry(mu_img.GetSize(), ct_img.GetSpacing())

        # print(mu_array.shape,vol_geom)
        proj_data = project_volume(mu_array, vol_geom, proj_geom)
        print(proj_data.shape)

        drr_array = np.transpose(proj_data, (1, 0, 2))
        print(drr_array.shape)
        save_as_nii(drr_array, os.path.join(phase_dir, 'DRR.nii'))

        drr_poi = add_poisson_noise_to_projection(proj_data, num_photons=100000,dtype=np.float32)
        print(np.max(proj_data),np.min(proj_data),np.max(drr_poi),np.min(drr_poi))
        drr_poi_array = np.transpose(drr_poi, (1, 0, 2))
        save_as_nii(drr_poi_array, os.path.join(phase_dir, 'DRR_Poi.nii'))

        drr_poi_gau = add_gaussian_noise_to_projection(drr_poi, I0=1e5, sigma=10, clip_min=1e-3)
        drr_poi_gau_array = np.transpose(drr_poi_gau, (1, 0, 2))
        save_as_nii(drr_poi_gau_array, os.path.join(phase_dir, 'DRR_Poi_Gau.nii'))

        time_0=time.time()
        rec_original = reconstruct_sirt(proj_data, vol_geom, proj_geom)
        time_1 = time.time()

        print(time_1-time_0)
        save_as_nii(rec_original, os.path.join(phase_dir, 'recons.nii'), ct_img)

        rec_poi = reconstruct_sirt(drr_poi, vol_geom, proj_geom)
        save_as_nii(rec_poi, os.path.join(phase_dir, 'recons_Poi.nii'), ct_img)

        rec_poi_gau = reconstruct_sirt(drr_poi_gau, vol_geom, proj_geom)
        save_as_nii(rec_poi_gau, os.path.join(phase_dir, 'recons_Poi_Gau.nii'), ct_img)

print("Processing completed!")