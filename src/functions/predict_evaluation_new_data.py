import argparse
import os
import shutil
import subprocess
import multiprocessing
import pandas as pd
import numpy as np
import SimpleITK as sitk
import pydicom as dicom
from ..utils import data_utils
from ..utils import evaluation_utils
from . import data_generation as dg
from typing import Tuple, Union, Dict
from warnings import warn
from multiprocessing import Pool
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, load_json, save_json, save_pickle
from nnunetv2.evaluation.evaluate_predictions import compute_metrics, save_summary_json
from nnunetv2.postprocessing.remove_connected_components import remove_homolateral_healthy_brain, morphological_opening_by_reconstruction
from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager    
from nnunetv2.utilities.json_export import recursive_fix_for_json_export

# TODO: function for predicting in batch and just one file


def identify_pet_dicom_series(folder: str, keep_only_first: bool = True) -> Union[Tuple[str], Dict[str, Tuple[str]]]:
    """
    Identify PET DICOM series in a directory tree and return the file paths for the matching series.

    This function scans the provided folder (recursively) for DICOM files using SimpleITK's
    ImageSeriesReader. For each DICOM series found it reads the first file of the series
    with pydicom and selects series whose Modality tag equals 'PT' (PET studies). The function
    uses the series' SeriesDescription (with spaces replaced by underscores) as the dictionary key
    when returning multiple series.

    Parameters
    ----------
    folder : str
        Path to the root directory to search for DICOM files. The search is recursive.
    keep_only_first : bool, optional
        If True (default) and multiple PET series are found, only the first found series
        (arbitrary order determined by the filesystem/reader) is returned as a sequence of file
        paths. If False and multiple PET series are found, a dictionary mapping series descriptions
        to their file path lists is returned.

    Returns
    -------
    list[str] | dict[str, list[str]] | dict
        - If one PET series is found, returns a list of file paths (strings) for that series.
        - If multiple PET series are found:
            - If keep_only_first is True: returns a list of file paths for the first series kept.
            - If keep_only_first is False: returns a dict mapping SeriesDescription (spaces
              replaced by underscores) to lists of file paths for all detected PET series.
        - If no PET series is found: returns an empty dict ({}).
        Note: although the function's annotation uses Tuple, the actual returned container for
        file collections is a sequence (list) of file path strings.
    """
    image_series_reader = sitk.ImageSeriesReader()
    # Get all images path in root directory
    all_images_path = image_series_reader.GetGDCMSeriesFileNames(folder, recursive=True)
    # Retrieve unique folders with dicom files
    dicom_folders_list = []
    for image_path in all_images_path:
        image_folder = os.path.dirname(image_path)
        if image_folder not in dicom_folders_list:
            dicom_folders_list.append(image_folder)
    # Look which dicom files we are interested in
    matching_series = {}
    for dicom_folder in dicom_folders_list:
        series_ids_list = image_series_reader.GetGDCMSeriesIDs(dicom_folder)
        for series_id in series_ids_list:
            series_id_files = image_series_reader.GetGDCMSeriesFileNames(dicom_folder, seriesID=series_id)
            ds = dicom.read_file(series_id_files[0])
            if ds.Modality == 'PT': # Should we select on other dicom tags ? Radiopharmaceutical if consistent ?
                matching_series[ds.SeriesDescription.replace(' ', '_')] = series_id_files
    if matching_series:
        if len(matching_series.keys()) > 1:
            warn('More than one DICOM series was found for patient %s: %s.' % (folder, matching_series))
            if keep_only_first:
                series_to_keep = list(matching_series.keys())[0]
                warn('Only following series was kept: %s' % series_to_keep)
                return matching_series[series_to_keep]
        else:
            series_to_keep = list(matching_series.keys())[0]
            return matching_series[series_to_keep]
    else:
        warn('No folder with DICOM files was found for patient %s.' % folder)
    return matching_series
    

def process_patient_data(patient, patient_ID, image_dir, label_dir, n_labels,
                         ref_spacing, ref_size, skull_stripping, brain_centering):
    """
    Process and save image and label data for a single patient.

    Loads PET image and optional segmentation mask for a patient using
    data_utils.get_TEP_data_mask, optionally applies skull stripping and
    brain centering as requested, and writes the resulting image components
    and label file(s) to disk.

    Parameters
    ----------
    patient : str
        Path to the patient data file or directory.
    patient_ID : str
        Short string identifier for the patient used when creating output
        filenames.
    image_dir : str
        Directory path where image files will be written. Must be writable.
    label_dir : str
        Directory path where label (mask) files will be written. Must be writable.
    n_labels : int
        Number of segmentation labels expected (used when converting mask to
        nnU-Net format).
    skull_stripping : bool
        If True, request skull stripping when loading the data from the
        data utility.
    brain_centering : bool
        If True, request brain-centering (re-orientation/cropping) when
        loading the data from the data utility.

    Returns
    -------
    None
        Images and labels are written to disk as side effects; nothing is returned.
    """
    # Load image with the options specified
    image, mask = data_utils.get_TEP_data_mask(patient, plane='Axial', n_labels=n_labels,
                                                ref_spacing=ref_spacing, ref_size=ref_size,
                                                skull_stripping=skull_stripping, brain_centering=brain_centering)
    # Save images
    if image.GetNumberOfComponentsPerPixel() > 1:
        for i in range(image.GetNumberOfComponentsPerPixel()):
            img_i = sitk.VectorIndexSelectionCast(image, i)
            sitk.WriteImage(img_i, os.path.join(image_dir, 
                                                'PETseg_%s_%s.nii.gz' % (patient_ID, str(i).zfill(4))))
    else:
        sitk.WriteImage(image, os.path.join(image_dir, 'PETseg_%s_0000.nii.gz' % patient_ID))
    if mask is not None:
        # Convert mask from one hot encoded to required format
        final_mask = data_utils.convert_mask_nnUNet_format(mask, n_labels)
        sitk.WriteImage(final_mask, os.path.join(label_dir, 'PETseg_%s.nii.gz' % patient_ID))



def compute_metrics_on_folder_only_existing_ids(folder_ref: str, folder_pred: str, 
                                                dataset_json_file: str, plans_file: str,  
                                                output_file: str = None,  
                                                num_processes: int = default_num_processes) -> dict:
    """
    Reproduce nnunetv2.evaluation.evaluate_predictions.compute_metrics_on_folder but in a version
    where we only compute metrics on files that are present in both ref and pred.
    Also merge with compute_metrics_on_folder2
    output_file must end with .json; can be None
    """
    # === This is from nnunetv2.evaluation.evaluate_predictions.compute_metrics_on_folder2 ===
    dataset_json = load_json(dataset_json_file)
    # get file ending
    file_ending = dataset_json['file_ending']

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    image_reader_writer = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    regions_or_labels = lm.foreground_regions if lm.has_regions else lm.foreground_labels
    ignore_label = lm.ignore_label
    
    # === This is from nnunetv2.evaluation.evaluate_predictions.compute_metrics_on_folder ===
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    all_files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    all_files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    # Modification is here where we take the intersection of the ids to ensure images exists in both folders
    files_ref = [join(folder_ref, i) for i in set(all_files_pred).intersection(all_files_ref)]
    files_pred = [join(folder_pred, i) for i in set(all_files_pred).intersection(all_files_ref)]
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        # for i in list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred))):
        #     compute_metrics(*i)
        results = pool.starmap(
            compute_metrics,
            list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred),
                     [ignore_label] * len(files_pred)))
        )

    # mean metric per class
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    means = {}
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            means[r][m] = np.nanmean([i['metrics'][r][m] for i in results])

    # foreground mean
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue
            values.append(means[k][m])
        foreground_mean[m] = np.mean(values)

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    recursive_fix_for_json_export(foreground_mean)
    result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}
    if output_file is not None:
        save_summary_json(result, output_file)
    return result 


def create_empty_pp_files(save_dir):
    """
    Create empty postprocessing files in the specified directory.

    Parameters
    ----------
    save_dir : str or os.PathLike
        Path to the directory where postprocessing files will be created.

    Returns
    -------
    None
    """
    save_json({"postprocessing_fns": [], "postprocessing_kwargs": []}, os.path.join(save_dir, 'postprocessing.json'))
    save_pickle(([], []), os.path.join(save_dir, 'postprocessing.pkl'))
    

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Inference on new data')
    parser.add_argument("--input_dir", type=str,
                        required=True,
                        help='Root directory of images.\nWe exepect a directory with each sample in its own subdirectory. '
                        'For each sample a folder containing the DICOM PET series required by the dataset.' 
                        'If several series from the same modalities are given, only the first will be considered.' 
                        'Images can be supplied in other formats readble by SimpleITK but it this case their name should include "PET".' 
                        'Optionnaly the ground truth VOI(s) will be loaded if they are nifti files in a subfolder named ROI and their names'
                        'match those of the labels in the dataset.json')
    parser.add_argument("--output_dir", help='Root directory of output files (seg + results)', type=str)    
    parser.add_argument("--dataset", help='Dataset name or dataset number', required=True, type=str)
    parser.add_argument("--nnUNet_trainer", help='nnUNet trainer name', type=str, required=True)
    parser.add_argument("--nnUNet_plans", help='nnUNet plan name', type=str, required=True)
    parser.add_argument("--configuration", help='Model configuration', type=str, required=True)
    parser.add_argument("--force_postprocessing", 
                        help="On the sidelines of nnunet best post-processing, apply a force post-processing with "
                        "morphological_opening_by_reconstruction on tumor and remove_homolateral_healthy_brain on brain if present", 
                        action='store_true', default=False)
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    if args.dataset.isdigit():
        full_dataset_name = data_utils.get_dataset_fullname(dataset_number=args.dataset)
    else:
        full_dataset_name = data_utils.get_dataset_fullname(dataset_name=args.dataset)
        
    # Set tolerance on coordinates difference
    sitk.ProcessObject_SetGlobalDefaultCoordinateTolerance(1e-2)
        
    # Load database creation arguments and label information
    dataset_model_dir = os.path.join(os.environ.get('nnUNet_results'), full_dataset_name, 
                                '%s__%s__%s' % (args.nnUNet_trainer, args.nnUNet_plans, args.configuration))
    database_args = load_json(os.path.join(dataset_model_dir, 'dataset_commandline_args.txt'))
    dataset_json = load_json(os.path.join(dataset_model_dir, 'dataset.json'))
    label_link = {label_name.lower(): label for label_name, label in dataset_json['labels'].items() if label_name != 'background'}
    n_labels = len(label_link.keys())
    
    if database_args['skull_stripping']:
        # fix threading correctly
        env = os.environ.copy()
        env["MKL_THREADING_LAYER"] = "GNU"          # avoid conflict with libgomp
        env.pop("MKL_SERVICE_FORCE_INTEL", None)    # just in case this is set somewhere
        # find models path
        mri_synthstrip_candidate_dirs = list(set([folder for folder in os.environ["PATH"].split(os.pathsep) if 'mri_synthstrip' in folder]))
        if len(mri_synthstrip_candidate_dirs) == 1:
            # model with csf in brain boundary, otherwise model is synthstrip.nocsf.1.pt
            model_path = os.path.join(mri_synthstrip_candidate_dirs[0], 'synthstrip.1.pt')
            if not os.path.isfile(model_path):
                raise FileNotFoundError("model not found in its supposed place %s" % model_path)
            # check than we can execute the script
            script_path = os.path.join(mri_synthstrip_candidate_dirs[0], 'mri_synthstrip')
            if not os.path.isfile(script_path):
                raise FileNotFoundError("mri_synthstrip script not found in its supposed place %s" % script_path)
            else:
                if not os.access(script_path, os.X_OK):
                    raise RuntimeError('mri_synthstrip script can not be executed (permission problem).')
        else:
            if not mri_synthstrip_candidate_dirs:
                raise FileNotFoundError("mri_synthstrip dir not found")
            else:
                raise RuntimeError("several entries found in PATH for mri_synthstrip. Ensure there is only one.")
    
    patient_names_list = [_ for _ in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, _))]
    # 1. Preprocess DICOM data as in data_generation.py
    # TODO: parallelize this ? could be problematic with skullstripping. need test for time diff
    image_file_reader = sitk.ImageFileReader()
    for patient in patient_names_list:
        patient_folder = os.path.join(input_dir, patient)
        # Identify directory with DICOM PET images
        identified_series_files = identify_pet_dicom_series(patient_folder, keep_only_first=True)
        if identified_series_files:
            # Load DICOM image
            static_image, missing_static_instances = dg.read_dicom_image(series_img_list=identified_series_files)
        else:
            # Might be another format ? Try to load first file containing 'PET'
            pet_files = [_ for _ in os.listdir(patient_folder) if 'PET' in _ and os.path.isfile(os.path.join(patient_folder, _))]
            for file in pet_files:
                image_io = image_file_reader.GetImageIOFromFileName(file)
                if image_io:
                    image_file_reader.SetFileName(file)
                    static_image = image_file_reader.Execute()
                    missing_static_instances = []
                    break
        # Just check if ROI folder exists and load rois needed for dataset
        # TODO : what if rois are in DICOM RT STRUCT format ? Handle this
        # utiliser Media Storage Standard SOP Classes pour savoir le type de fichier dicom https://dicom.nema.org/dicom/2013/output/chtml/part04/sect_i.4.html
        if os.path.exists(os.path.join(patient_folder, 'ROI')):
            roi_dic = dg.load_rois(os.path.join(patient_folder, 'ROI'))
            
            for roi in roi_dic:
                if not roi_dic[roi].IsSameImageGeometryAs(static_image):
                    # We resample ROI geometry to image ones, assuming that this is the case
                    # TODO : maybe check that we still have some tumor in the image after resampling if there was initially otherwise it might be that the roi is not associated with the good image and we need to raise an error
                    roi_dic[roi] = sitk.Resample(roi_dic[roi], static_image, sitk.Transform(), sitk.sitkLabelLinear)
                else:
                    # Correct the case where loading saved image results in different origin (approx 1e-5 differences)
                    roi_dic[roi] = dg.correct_origin(static_image, roi_dic[roi], tol=1e-2)
            roi_dic_keys_lower = {_.lower(): _ for _ in roi_dic}
            # If the user only supply TumorBox should we perform TBR1.6 seg here
            if 'tumorbox' in roi_dic_keys_lower and 'brain' in roi_dic_keys_lower:
                label_shape = sitk.LabelShapeStatisticsImageFilter()
                threshold = 1.6
                roi, _ = dg.tumor_TBR_segmentation(static_image, roi_dic[roi_dic_keys_lower['tumorbox']], 
                                                   roi_dic[roi_dic_keys_lower['brain']], threshold_factor=threshold)
                _, roi_resampled = data_utils.IBSI_resampling(static_image, roi, resampledPixelSpacing=(1 ,1, 1))
                processed_roi_resampled = dg.segmentation_post_treatment(roi_resampled, [1, 1, 1], method='OpeningReconstruction', use_image_spacing=False, select_biggest=False)
                processed_roi = sitk.Resample(processed_roi_resampled, roi, sitk.Transform(), sitk.sitkLabelLinear)
                label_shape.Execute(processed_roi)
                if 1 in label_shape.GetLabels():
                    # Accept processed roi only if we still have a volume after
                    roi = processed_roi
                roi_dic['Tumor'] = roi
                roi_dic_keys_lower['tumor'] = 'Tumor'
            # Keep only rois needed for this dataset
            roi_dic = {_: roi_dic[roi_dic_keys_lower[_]] for _ in roi_dic_keys_lower if _ in label_link}
            # If missing slices in image also apply in ROI
            if missing_static_instances:
                for roi in roi_dic:
                    if roi_dic[roi].GetSize()[-1] != static_image.GetSize()[-1]:
                        roi_dic[roi] = dg.interpolate_roi_missing_slices(static_image, roi_dic[roi], missing_static_instances)
        else:
            roi_dic = {}
        
        # Save all
        if roi_dic:
            os.makedirs(os.path.join(output_dir, 'Generated_data', patient.replace(' ', '_'), 'ROI'), exist_ok=True)
        else:
            os.makedirs(os.path.join(output_dir, 'Generated_data', patient.replace(' ', '_')), exist_ok=True)
        sitk.WriteImage(static_image, os.path.join(output_dir, 'Generated_data', patient.replace(' ', '_'), 'Static_PET.nii.gz'))
        for roi in roi_dic:
            if roi.lower() == 'tumor':
                matching_roi_name = 'Tumor_TBR1,6_seg.nii.gz'
            elif roi.lower() == 'brain':
                matching_roi_name = 'Brain.nii.gz'
            else:
                raise ValueError('roi name not recognized: %s.' % roi)
            sitk.WriteImage(sitk.Cast(roi_dic[roi], sitk.sitkUInt8), os.path.join(output_dir, 'Generated_data', patient.replace(' ', '_'), 'ROI', '%s.nii.gz' % matching_roi_name))
        
        # Run skull stripping if needed
        if database_args['skull_stripping']:
            input_image_path = os.path.join(output_dir, 'Generated_data', patient.replace(' ', '_'), 'Static_PET.nii.gz')
            output_image_path = os.path.join(output_dir, 'Generated_data', patient.replace(' ', '_'), 'Static_PET_skullstripped.nii.gz')
            output_mask_path = os.path.join(output_dir, 'Generated_data', patient.replace(' ', '_'), 'Static_PET_skullstripped_mask.nii.gz')
            subprocess.run(['mri_synthstrip', '-i', input_image_path, '-o', output_image_path, '-m', output_mask_path,
                            '--model', model_path], check=True, env=env)
            # Ensure that values are equal to 0 outside of skullstrip mask. Sometimes we add small values
            skullstrip_mask = sitk.ReadImage(os.path.join(output_dir, 'Generated_data', patient.replace(' ', '_'), 'Static_PET_skullstripped_mask.nii.gz'))
            skullstrip_image = sitk.ReadImage(os.path.join(output_dir, 'Generated_data', patient.replace(' ', '_'), 'Static_PET_skullstripped.nii.gz'))
            sitk.WriteImage(sitk.Mask(skullstrip_image, skullstrip_mask), os.path.join(output_dir, 'Generated_data', patient.replace(' ', '_'), 'Static_PET_skullstripped.nii.gz'))
    
    # 2. Generate network input data as in generate_nnunet_databse.py
    ref_spacing = database_args['ref_spacing']
    ref_size = database_args['ref_size']
    # TODO: parallelize this ? need test for time diff. Add threads arg if needed
    input_nnunet_image_dir = os.path.join(output_dir, 'nnUNet_data', 'imagesTs')
    os.makedirs(input_nnunet_image_dir, exist_ok=True)
    input_nnunet_label_dir = os.path.join(output_dir, 'nnUNet_data', 'labelsTs')
    os.makedirs(input_nnunet_label_dir, exist_ok=True)
    # Process image of each patient
    for i, patient in enumerate(patient_names_list):
        patient_folder = os.path.join(output_dir, 'Generated_data', patient.replace(' ', '_'))
        patient_ID = str(i).zfill(len(str(len(patient_names_list))))
        process_patient_data(patient_folder, patient_ID, input_nnunet_image_dir, input_nnunet_label_dir, n_labels,
                             ref_spacing, ref_size,
                             skull_stripping=database_args['skull_stripping'], brain_centering=database_args['brain_centering'])
        
    # Generate link between nnUnet identifier and original identifier (name of original folder)
    patient_link = {'Original ID': [], 'nnUNet ID': []}
    for i, patient in enumerate(patient_names_list):
        patient_link['Original ID'].append(patient)
        patient_link['nnUNet ID'].append(str(i).zfill(len(str(len(patient_names_list)))))
    patient_link_df = pd.DataFrame(patient_link)
    patient_link_df.to_csv(os.path.join(output_dir, 'nnUNet_data', 'Patient_link.csv'))
    
    print("===== Prediction =====")    
    # 3. Run prediction
    
    # Generate predictions as usual as no ensembling is needed
    pred_output = os.path.join(output_dir, 'nnUNet_results')
    # Run prediction
    os.system('nnUNetv2_predict -d %s -i %s -o %s -f 0 1 2 3 4 -p %s -c %s -tr %s -npp 10 -nps 10 -device cuda' % (full_dataset_name, 
                                                                                                                    os.path.join(output_dir, 'nnUNet_data', 'imagesTs'),
                                                                                                                    pred_output,
                                                                                                                    args.nnUNet_plans, args.configuration, args.nnUNet_trainer))
    # Run post processing if needed
    print('==== Post processing ====')
    crossval_results_dir = os.path.join(dataset_model_dir, 'crossval_results_folds_0_1_2_3_4')
    if os.path.exists(os.path.join(crossval_results_dir, 'postprocessing.json')):
        pp_json = load_json(os.path.join(crossval_results_dir, 'postprocessing.json'))
        if pp_json["postprocessing_fns"]:
            os.system('nnUNetv2_apply_postprocessing -i %s -o %s -pp_pkl_file %s -np 8 -plans_json %s' % (pred_output, 
                                                                                                        os.path.join(pred_output, 'postprocessed'),
                                                                                                        os.path.join(crossval_results_dir, 'postprocessing.pkl'), 
                                                                                                        os.path.join(pred_output, 'plans.json')))
            # TODO : is this really needed ? 
            # Copy needed files to match structure after find_best_configuration
            shutil.copy(os.path.join(pred_output, 'dataset.json'), os.path.join(pred_output, 'postprocessed', 'dataset.json'))
            shutil.copy(os.path.join(pred_output, 'plans.json'), os.path.join(pred_output, 'postprocessed', 'plans.json'))
        shutil.copy(os.path.join(crossval_results_dir, 'postprocessing.json'), os.path.join(pred_output, 'postprocessing.json'))
        shutil.copy(os.path.join(crossval_results_dir, 'postprocessing.pkl'), os.path.join(pred_output, 'postprocessing.pkl'))
        pp_pkl_file = os.path.join(pred_output, 'postprocessing.pkl')
    else:
        pp_pkl_file = None
        
    # Run forced post processing if needed
    if args.force_postprocessing:
        print('==== Forced post processing ====')
        # Create a postprocessing.pkl with functions we want to force postprocessing according to our will
        pp_fns = [morphological_opening_by_reconstruction]
        pp_fn_kwargs = [{'need_props': True, 'kernel_size': 1, 'use_image_spacing': False, 'labels_or_regions': label_link['tumor']}]
        if 'brain' in label_link:
            pp_fns.append(remove_homolateral_healthy_brain)
            pp_fn_kwargs.append({'labels_or_regions': [label_link['brain'], label_link['tumor']], 'need_props': True, 'label_dict': label_link})
        pp_pkl_file_path = os.path.join(pred_output, 'forced_postprocessing.pkl')
        save_pickle((pp_fns, pp_fn_kwargs), pp_pkl_file_path)
        # Call nnUNetv2_apply_postprocessing
        os.system('nnUNetv2_apply_postprocessing -i %s -o %s -pp_pkl_file %s -np 8 -plans_json %s' % (pred_output,  
                                                                                                      os.path.join(pred_output, 'forced_postprocessed'), 
                                                                                                      os.path.join(pred_output, 'forced_postprocessing.pkl'),  
                                                                                                      os.path.join(pred_output, 'plans.json')))
        # Create associated json
        # Normally pp.json contains information on performance of post processing compare to no post processing.
        # However at inference time, this information is not needed and we might not have the GT predictions.
        # Moreover, it will be computed later in GT are available. In consequence we construction the json without this information.
        # Code below is from the end of nnunetv2.postprocessing.remove_connected_components.determine_postprocessing
        tmp = {
        'postprocessing_fns': [i.__name__ for i in pp_fns],
        'postprocessing_kwargs': pp_fn_kwargs,
        }
        # did I already say that I hate json? "TypeError: Object of type int64 is not JSON serializable"
        recursive_fix_for_json_export(tmp)
        save_json(tmp, os.path.join(pred_output, 'forced_postprocessing.json'))
        # Copy needed files to match structure after find_best_configuration
        shutil.copy(os.path.join(pred_output, 'dataset.json'), os.path.join(pred_output, 'forced_postprocessed', 'dataset.json'))
        shutil.copy(os.path.join(pred_output, 'plans.json'), os.path.join(pred_output, 'forced_postprocessed', 'plans.json'))
    
    print("===== Analyze results =====")
    # 4. Analyze results 
    # If GT VOI(s) are given, analysis with segmentation metrics + quantitative results on GT + pred
    # Else quantitative on pred only
    if os.listdir(input_nnunet_label_dir):
        # Get results with nnunet evaluation in summary.json
        for output in ['raw', 'postprocessed', 'forced_pp']:
            if output == 'raw':
                summary_dir = os.path.join(output_dir, 'nnUNet_results')
            elif output == 'forced_pp':
                if not args.force_postprocessing:
                    continue
                summary_dir = os.path.join(output_dir, 'nnUNet_results', 'forced_postprocessed')
            else:
                if pp_pkl_file is None:
                    continue
                summary_dir = os.path.join(output_dir, 'nnUNet_results', 'postprocessed')
            compute_metrics_on_folder_only_existing_ids(input_nnunet_label_dir,  
                                                        summary_dir, os.path.join(summary_dir, 'dataset.json'),  
                                                        os.path.join(summary_dir, 'plans.json'),  
                                                        os.path.join(summary_dir, 'summary.json'))
        with pd.ExcelWriter(os.path.join(output_dir, 'nnUNet_results', 'Global_results.xlsx')) as writer:
            # Get results by samples
            global_results_dic = evaluation_utils.extract_global_results(full_dataset_name, patient_link_df.set_index('nnUNet ID'), 
                                                                   os.path.join(output_dir, 'nnUNet_results'), 
                                                                   os.path.join(output_dir, 'nnUNet_data', 'imagesTs'),
                                                                   force_postprocessing=args.force_postprocessing)
            # Compute overall and fold results
            for output in global_results_dic:
                # Get overall results and add to recap_results_file (bypass fold results by doing as if there is only one fold)
                test_overall_mean_std = evaluation_utils.extract_fold_results(global_results_dic[output].drop('Original ID', axis=1), 
                                                                              pd.DataFrame({'Fold': np.zeros_like(patient_link_df['nnUNet ID']), 
                                                                                            'nnUNet ID': patient_link_df['nnUNet ID']}).set_index('nnUNet ID'))
                # Write results by samples and fold
                test_overall_mean_std.transpose().to_excel(writer, sheet_name='%s_Fold' % output)
                global_results_dic[output].to_excel(writer, sheet_name='%s_Sample' % output)
    else:
        global_results_dic = {}
        for output in ['raw', 'processed', 'forced_pp']:
            if output == 'raw':
                pred_output = os.path.join(output_dir, 'nnUNet_results')
            elif output == 'forced_pp':
                if not args.force_postprocessing:
                    continue
                pred_output = os.path.join(output_dir, 'nnUNet_results', 'forced_postprocessed')
            else:
                if pp_pkl_file is None:
                    continue
                pred_output = os.path.join(output_dir, 'nnUNet_results', 'postprocessed')
            output_dic = {'nnUnet ID': [], 'Original ID': []}
            # get file ending
            file_ending = dataset_json['file_ending']
            all_files_pred = subfiles(pred_output, suffix=file_ending, join=True)
            for file_pred in all_files_pred:
                nnUNet_patient_id = os.path.basename(file_pred).split('.nii.gz')[0].split('PETseg_')[1]
                original_patient_id = patient_link_df.set_index('nnUNet ID').loc[nnUNet_patient_id, 'Original ID']
                pred_mask = sitk.ReadImage(file_pred)
                input_image = sitk.ReadImage(os.path.join(output_dir, 'nnUNet_data', 'imagesTs', 'PETseg_%s_0000.nii.gz' % nnUNet_patient_id))
                patient_results = evaluation_utils.compute_quantitative(input_image, pred_mask, label_link, nan_for_nonexisting=True)
                output_dic['nnUnet ID'].append(nnUNet_patient_id)
                output_dic['Original ID'].append(original_patient_id)
                for key, value in patient_results.items():
                    if not key in output_dic:
                        output_dic[key] = []
                    output_dic[key].append(value)
            global_results_dic[output] = pd.DataFrame(output_dic).set_index('nnUnet ID').sort_index()
        with pd.ExcelWriter(os.path.join(output_dir, 'nnUNet_results', 'Global_results.xlsx')) as writer:
            for output in global_results_dic:
                global_results_dic[output].to_excel(writer, sheet_name='%s_Sample' % output)
    # TODO : if dicom is given in input make dicom RT Struct in output (need resampling to original image if needed)
    # voir https://github.com/qurit/rt-utils/issues/87