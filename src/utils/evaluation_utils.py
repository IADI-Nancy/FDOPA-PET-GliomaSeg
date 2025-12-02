import os
import ast
import numpy as np
import pandas as pd
import SimpleITK as sitk
import sklearn.metrics as skm
from batchgenerators.utilities.file_and_folder_operations import load_json
from surface_distance.metrics import compute_surface_distances, compute_surface_dice_at_tolerance
from joblib import Parallel, delayed


def extract_nnUNet_patient_results(patient_results, label_link):
    """
    Extract per-label evaluation metrics from a single patient's nnU-Net result dictionary.

    Parameters
    ----------
    patient_results : dict[str, dict[str, [float, int]]]
        Dictionary containing nnUNet results for a single patient as stored in summary.json.

    label_link : dict[str, int]
        Dictionary mapping from human-readable label name (str) to label id (int or str) as stored in dataset.json.

    Returns
    -------
    nnUnet_results: dict
        Dictionary containing computed metrics for every label in label_link.
    """
    nnUnet_results = {}
    for label_name, label in label_link.items():
        tp, fp, fn, tn = patient_results[str(label)]['TP'], patient_results[str(label)]['FP'], patient_results[str(label)]['FN'], patient_results[str(label)]['TN']           
        nnUnet_results.update({'nnUNet_Dice_%s' % label_name: patient_results[str(label)]['Dice'],
                               'nnUNet_Accuracy_%s' % label_name: (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else np.nan,
                               'nnUNet_Sensitivity_%s' % label_name: tp / (tp + fn) if (tp + fn) != 0 else np.nan,
                               'nnUNet_Specificity_%s' % label_name: tn / (tn + fp) if (tn + fp) != 0 else np.nan,
                               'nnUNet_PPV_%s' % label_name: tp / (tp + fp) if (tp + fp) != 0 else np.nan,
                               'nnUNet_NPV_%s' % label_name: tn / (tn + fn) if (tn + fn) != 0 else np.nan,
                               'nnUNet_N_voxels_seg_GT_%s' % label_name: patient_results[str(label)]['n_ref'],
                               'nnUNet_N_voxels_seg_model_%s' % label_name: patient_results[str(label)]['n_pred']
                               })
    return nnUnet_results

def surface_hausdorff_distance95(reference_segmentation, seg):
    """
    Compute the 95th-percentile symmetric surface Hausdorff distance (HD95) between two binary segmentations.
    From http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html
    
    Parameters
    ----------
    reference_segmentation : SimpleITK.Image
        Binary reference segmentation image. Non-zero voxels are treated as foreground.
    seg : SimpleITK.Image
        Binary segmentation image to compare against the reference. Non-zero voxels are treated as foreground.
        
    Returns
    -------
    float
        The 95th percentile of the combined surface-to-surface distances (in physical units, respecting image spacing).
        This is computed symmetrically by collecting distances from each surface to the other, padding missing surface
        samples with zeros so each surface contributes equally, and taking the 95th percentile of the pooled distances.
    """
    
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(reference_segmentation, squaredDistance=False, useImageSpacing=True))
    reference_surface = sitk.LabelContour(reference_segmentation)

    statistics_image_filter = sitk.StatisticsImageFilter()
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())

    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(seg, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(seg)

    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances +  list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances +  list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
    all_surface_distances = seg2ref_distances + ref2seg_distances
    return np.percentile(all_surface_distances, 95)

def get_BRATS_results(y_true, y_pred, label_link, nan_for_nonexisting=False):
    """
    Compute BRATS-style evaluation metrics for each label provided in label_link.
    
    Parameters
    ----------
    y_true : SimpleITK.Image
        Ground-truth segmentation image. Expected to be a label image where voxel
        values correspond to integer class labels.
    y_pred : SimpleITK.Image
        Predicted segmentation image. This function will cast y_pred to the pixel
        type of y_true if the types differ. y_pred is expected to be a label image
        (not softmax probabilities).
    label_link : dict[str, int]
        Dictionary mapping from human-readable label name (str) to label id (int or str) as stored in dataset.json.
    nan_for_nonexisting : bool, optional
        If True, metrics for labels that do not exist in both prediction and
        ground truth are set to np.nan. If False (default behavior in this
        function), such labels are treated as perfect matches (Dice=1,
        Hausdorff=0, SurfaceDice@1mm=1).
        
    Returns
    -------
    hm_results: dict[str, float]
        A dictionary containing BRATS-style metrics for each label. For each
        label_name in label_link the following keys are produced:
          - 'HM_Dice_<label_name>' : Dice coefficient
          - 'HM_Surface_Dice_<label_name>' : Surface Dice at 1 mm tolerance
          - 'HM_Hausdorff95_<label_name>' : 95th percentile Hausdorff distance (in mm)
          - 'HM_Sensitivity_<label_name>' : Sensitivity / recall
          - 'HM_Specificity_<label_name>' : Specificity
    """
    hm_results = {}
    # Check whether y_true and y_pred occupy the same physical space        
    if not y_true.IsSameImageGeometryAs(y_pred):
        raise ValueError('y_true and y_pred do not occupy the same physical space')
    
    # Force same type for y_true and y_pred. User must take care that y_pred is a segmentation
    if y_true.GetPixelID() != y_pred.GetPixelID():
        y_pred = sitk.Cast(y_pred, y_true.GetPixelID())
    
    for label_name, label in label_link.items():
        # Softmax --> image shape = (x, y, z) with each voxel being the class of the label
        binary_pred = y_pred == label
        binary_pred_arr = sitk.GetArrayFromImage(binary_pred)
        binary_true = y_true == label
        binary_true_arr = sitk.GetArrayFromImage(binary_true)

        label_intersection = np.logical_and(binary_pred_arr == 1, binary_true_arr == 1).sum()
        bg_intersection = np.logical_and(binary_pred_arr == 0, binary_true_arr == 0).sum()
        union = np.sum(binary_pred_arr) + np.sum(binary_true_arr)
        # Hausdorff function return very big numbers in case of empty mask, set it to max image size in mm
        max_distance = np.linalg.norm(np.asarray(binary_true.GetSize()) * np.asarray(binary_true.GetSpacing()))
        if union == 0:
            if nan_for_nonexisting:
                dice = np.nan
                hausdorff = np.nan
                surface_dice_1mm = np.nan
                
            else:
                # This is different from nnunet that consider this case as nan. 
                # Also used in BRATS http://github.com/rachitsaluja/BraTS-2023-Metrics/blob/main/metrics.py
                dice = 1 
                hausdorff = 0
                surface_dice_1mm = 1
        else:
            dice = 2 * label_intersection / union
            surface_distances = compute_surface_distances(binary_true_arr == 1, binary_pred_arr == 1, y_true.GetSpacing()[::-1])
            surface_dice_1mm = compute_surface_dice_at_tolerance(surface_distances, 1)
            if np.count_nonzero(binary_true_arr) == 0 or np.count_nonzero(binary_pred_arr) == 0:
                if nan_for_nonexisting:
                    hausdorff = np.nan
                else:
                    hausdorff = max_distance
            else:
                hausdorff = surface_hausdorff_distance95(binary_true, binary_pred)
        if hausdorff > max_distance:
            hausdorff = max_distance

        if np.count_nonzero(binary_true_arr) == 0:
            if np.count_nonzero(binary_pred_arr) == 0:
                sensitivity = 1
            else:
                sensitivity = 0
        else:
            sensitivity = label_intersection / np.sum(binary_true_arr == 1)
        specificity = bg_intersection / np.sum(binary_true_arr == 0)
        
        hm_results.update({'HM_Dice_%s' % label_name: dice,
                           'HM_Surface_Dice_%s' % label_name: surface_dice_1mm,
                           'HM_Hausdorff95_%s' % label_name: hausdorff,
                           'HM_Sensitivity_%s' % label_name: sensitivity,
                           'HM_Specificity_%s' % label_name: specificity
                           })
    return hm_results


def compute_quantitative(image, mask, label_link, nan_for_nonexisting=True):
    """
    Compute basic intensity statistics and volume for labeled regions in an image.
    
    Parameters
    ----------
    image : SimpleITK.Image
        Intensity image from which statistics (mean, maximum, standard deviation)
        are computed. Must be spatially aligned with `mask`.
    mask : SimpleITK.Image
        Label image (integer-valued) defining regions of interest. Labels should
        correspond to the integer values referenced by `label_link`.
    label_link : dict[str, int]
        Dictionary mapping from human-readable label name (str) to label id (int or str) as stored in dataset.json.
    nan_for_nonexisting : bool, optional
        If True (default), statistics for labels that are not present in `mask` will be
        set to numpy.nan. If False, missing-label statistics are set to 0.
        Volume for a non-existing label is always returned as 0.
        
    Returns
    -------
    quantif_results: dict[str, float]
        A dictionary with keys in the form:
          - 'HM_Mean_<label_name>' : float
          - 'HM_Max_<label_name>'  : float
          - 'HM_Std_<label_name>'  : float
          - 'HM_Vol_<label_name>'  : float
        For each label_name in `label_link`. Volumes are returned in mL
        assuming the image spacing is expressed in millimeters.
    """
    quantif_results = {}
    label_stats = sitk.LabelStatisticsImageFilter()
    label_stats.Execute(image, mask)
    for label_name, label in label_link.items():
        if label in label_stats.GetLabels():
            mean_mask = label_stats.GetMean(label)
            max_mask = label_stats.GetMaximum(label)
            std_mask = label_stats.GetSigma(label)
            vol_mask_mL = label_stats.GetCount(label) * np.prod(mask.GetSpacing()) / 1000
        else:
            if nan_for_nonexisting:
                mean_mask = np.nan
                max_mask = np.nan
                std_mask = np.nan
            else:
                mean_mask = 0
                max_mask = 0
                std_mask = 0
            vol_mask_mL = 0
        
        quantif_results.update({'HM_Mean_%s' % label_name: mean_mask,
                                'HM_Max_%s' % label_name: max_mask,
                                'HM_Std_%s' % label_name: std_mask,
                                'HM_Vol_%s' % label_name: vol_mask_mL,
                                })
    return quantif_results

def get_quantitative_results(image, y_true, y_pred, label_link, nan_for_nonexisting=True):
    """
    Compute paired quantitative metrics for predicted and ground-truth segmentations.
    This function computes intensity- and volume-based quantitative results for each label
    provided in `label_link` for both the predicted segmentation (`y_pred`) and the
    ground-truth segmentation (`y_true`). It also computes absolute and relative
    (predicted vs true) differences for each metric.
    
    Parameters
    ----------
    image : SimpleITK.Image
        The intensity image from which intensity-based statistics (mean, max, std)
        are computed.
    y_true : SimpleITK.Image
        Ground-truth segmentation image. This must occupy the same physical space
        as `y_pred` (same origin, spacing, direction and size).
    y_pred : SimpleITK.Image
        Predicted segmentation image. If its pixel type differs from `y_true` it
        will be cast to `y_true`'s pixel type before quantitative computation.
    label_link : dict[str, int]
        Dictionary mapping from human-readable label name (str) to label id (int or str) as stored in dataset.json.
    nan_for_nonexisting : bool, optional (default=True)
        Forwarded to `compute_quantitative`. Controls how the helper handles labels
        that do not exist in a segmentation (e.g. return NaN vs. a default value).
        
    Returns
    -------
    quantif_results: dict[str, float]
        Dictionary containing quantitative results for each label in `label_link` for ground-truth and predicted segmentations.
        Also contains absolute and relative differences (predicted vs true) for each metric.
    """
    quantif_results = {}
    
    # Check whether y_true and y_pred occupy the same physical space        
    if not y_true.IsSameImageGeometryAs(y_pred):
        raise ValueError('y_true and y_pred do not occupy the same physical space')
    
    # Force same type for y_true and y_pred. User must take care that y_pred is a segmentation
    if y_true.GetPixelID() != y_pred.GetPixelID():
        y_pred = sitk.Cast(y_pred, y_true.GetPixelID())
    
    quantif_pred_results = compute_quantitative(image, y_pred, label_link, nan_for_nonexisting)
    for label_name in label_link:
       quantif_results.update({'HM_Mean_pred_%s' % label_name: quantif_pred_results['HM_Mean_%s' % label_name], 
                               'HM_Max_pred_%s' % label_name: quantif_pred_results['HM_Max_%s' % label_name], 
                               'HM_Std_pred_%s' % label_name: quantif_pred_results['HM_Std_%s' % label_name], 
                               'HM_Vol_pred_%s' % label_name: quantif_pred_results['HM_Vol_%s' % label_name]})
    quantif_true_results = compute_quantitative(image, y_true, label_link, nan_for_nonexisting)
    for label_name in label_link:
       quantif_results.update({'HM_Mean_true_%s' % label_name: quantif_true_results['HM_Mean_%s' % label_name],
                                'HM_Max_true_%s' % label_name: quantif_true_results['HM_Max_%s' % label_name],
                                'HM_Std_true_%s' % label_name: quantif_true_results['HM_Std_%s' % label_name],
                                'HM_Vol_true_%s' % label_name: quantif_true_results['HM_Vol_%s' % label_name],
                                'HM_Diff_abs_Mean_%s' % label_name: quantif_true_results['HM_Mean_%s' % label_name] - quantif_pred_results['HM_Mean_%s' % label_name],
                                'HM_Diff_abs_Max_%s' % label_name: quantif_true_results['HM_Max_%s' % label_name] - quantif_pred_results['HM_Max_%s' % label_name],
                                'HM_Diff_abs_Std_%s' % label_name: quantif_true_results['HM_Std_%s' % label_name] - quantif_pred_results['HM_Std_%s' % label_name],
                                'HM_Diff_abs_Vol_%s' % label_name: quantif_true_results['HM_Vol_%s' % label_name] - quantif_pred_results['HM_Vol_%s' % label_name],
                                'HM_Diff_rel_Mean_%s' % label_name: (quantif_true_results['HM_Mean_%s' % label_name] - quantif_pred_results['HM_Mean_%s' % label_name]) * 100 / quantif_true_results['HM_Mean_%s' % label_name] if quantif_true_results['HM_Mean_%s' % label_name] != 0 else np.nan,
                                'HM_Diff_rel_Max_%s' % label_name: (quantif_true_results['HM_Max_%s' % label_name] - quantif_pred_results['HM_Max_%s' % label_name]) * 100 / quantif_true_results['HM_Max_%s' % label_name] if quantif_true_results['HM_Max_%s' % label_name] != 0 else np.nan,
                                'HM_Diff_rel_Std_%s' % label_name: (quantif_true_results['HM_Std_%s' % label_name] - quantif_pred_results['HM_Std_%s' % label_name]) * 100 / quantif_true_results['HM_Std_%s' % label_name] if quantif_true_results['HM_Std_%s' % label_name] != 0 else np.nan,
                                'HM_Diff_rel_Vol_%s' % label_name: (quantif_true_results['HM_Vol_%s' % label_name] - quantif_pred_results['HM_Vol_%s' % label_name]) * 100 / quantif_true_results['HM_Vol_%s' % label_name] if quantif_true_results['HM_Vol_%s' % label_name] != 0 else np.nan
                                })
    return quantif_results


def get_patient_res(patient_res, patient_link_df, label_link, input_image_dir):
    nnUNet_patient_id = os.path.basename(patient_res['reference_file']).split('.nii.gz')[0].split('PETseg_')[1]
    original_patient_id = patient_link_df.loc[nnUNet_patient_id, 'Original ID']
    # Get nnUNet results
    patient_results = extract_nnUNet_patient_results(patient_res['metrics'], label_link)
    # Get homemade results with nnUNet inputs (results after resampling/cropping)
    ref_mask = sitk.ReadImage(patient_res['reference_file'])
    pred_mask = sitk.ReadImage(patient_res['prediction_file'])
    patient_results.update(get_BRATS_results(ref_mask, pred_mask, label_link))
    # Load tep image and get quantitative results
    ref_image = sitk.ReadImage(os.path.join(input_image_dir, 'PETseg_%s_0000.nii.gz' % nnUNet_patient_id))
    patient_results.update(get_quantitative_results(ref_image, ref_mask, pred_mask, label_link, nan_for_nonexisting=True))
    return (nnUNet_patient_id, original_patient_id, patient_results)


def extract_global_results(patient_link_df, summary_dir, input_image_dir,
                           force_postprocessing=False, n_jobs=25):
    """Extract the results for all the patients trained for a model dataset and a model type

    Parameters
        ----------
        patient_link_df : pd.DataFrame
            Dataframe with at least two columns : 'nnUnet ID' fpr IDs in nnUNet format 
            and 'Original ID' which is original IDs

        summary_dir : string
            Path to directory where summary.json is stored
        
        input_image_dir : string
            Path to the directory where images given as input are stored

    Returns
        -------
        results_df : pandas DataFrame
            Contains the results of raw data
    """
    
    # Retrieve label information from dataset.json file
    label_link = load_json(os.path.join(summary_dir, 'dataset.json'))['labels']
    label_link = {label_name: label for label_name, label in label_link.items() if label_name != 'background'}
    results_dic = {}
    output_list = ['raw', 'processed', 'forced_pp'] if force_postprocessing else ['raw', 'processed']
    for output in output_list:
        output_dic = {'nnUnet ID': [], 'Original ID': []}
        if output == 'raw':
            summary_path = os.path.join(summary_dir, 'summary.json')
        elif output == 'forced_pp':
            summary_path = os.path.join(summary_dir, 'forced_postprocessed', 'summary.json')
        else:
            # We will check that a postprocessing was applied otherwise no need to analyze it
            # information is stored in postprocessing.json, the field "postprocessing_fns" is empty if no postprocessing was applied
            pp_json = load_json(os.path.join(summary_dir, 'postprocessing.json'))
            if not pp_json["postprocessing_fns"]:
                continue
            summary_path = os.path.join(summary_dir, 'postprocessed', 'summary.json')
        nnunet_summary = load_json(summary_path) 
        out = Parallel(n_jobs=n_jobs)(delayed(get_patient_res)(patient_res, patient_link_df, label_link, input_image_dir) 
                          for patient_res in nnunet_summary["metric_per_case"])
        # Store results
        for out_patient in out:
            nnUNet_patient_id, original_patient_id, patient_results = out_patient 
            output_dic['nnUnet ID'].append(nnUNet_patient_id)
            output_dic['Original ID'].append(original_patient_id)
            for key, value in patient_results.items():
                if not key in output_dic:
                    output_dic[key] = []
                output_dic[key].append(value)
        results_dic[output] = pd.DataFrame(output_dic).set_index('nnUnet ID').sort_index()
    return results_dic

def extract_fold_results(global_results, fold_information):
    """
    Compute overall and per-fold summary statistics for numeric columns in a results DataFrame.

    This function selects numeric (non-object) columns from `global_results` and computes the
    following summary statistics: mean, standard deviation, median, 1st quartile (25th
    percentile) and 3rd quartile (75th percentile). It always returns the overall statistics
    and, if more than one unique fold is present in `fold_information['Fold']`, it also
    computes the same statistics separately for each fold and concatenates them into a single
    DataFrame.

    Parameters
    ----------
    global_results : pandas.DataFrame
        DataFrame containing evaluation results (rows are samples/cases, columns are metrics).
        Only non-object dtypes are considered for the numeric summaries.
    fold_information : pandas.DataFrame or pandas.Series-like
        Object containing fold assignments with a column or Series named 'Fold'. The function
        uses fold_information['Fold'] == fold to select rows corresponding to each fold.
        The fold information must align with `global_results` (i.e., it should have the same
        index or be the same length and in the same order as `global_results`).

    Returns
    -------
    fold_results: pandas.DataFrame
        A DataFrame of aggregated statistics. Rows correspond to the numeric metric column
        names from `global_results`. Columns are labeled with the statistic names:
          - For overall statistics: 'Mean overall', 'Std overall', 'Median overall',
            'Q1 overall', 'Q3 overall'.
          - For each fold (when multiple folds exist): 'Mean fold {fold}', 'Std fold {fold}',
            'Median fold {fold}', 'Q1 fold {fold}', 'Q3 fold {fold}'.
        If there is only one unique fold present in `fold_information['Fold']`, only the
        overall statistics (no per-fold breakdown) are returned.
    """
    global_results_quantitative = global_results.select_dtypes(exclude=['object'])
    overall_mean_std = pd.concat([global_results_quantitative.mean(), global_results_quantitative.std(),
                                  global_results_quantitative.median(), 
                                  global_results_quantitative.quantile(0.25), global_results_quantitative.quantile(0.75)], axis=1,
                                  keys=['Mean overall', 'Std overall', 'Median overall', 'Q1 overall', 'Q3 overall'])
    if len(fold_information['Fold'].unique()) != 1:
        fold_results = [overall_mean_std]
        for fold in fold_information['Fold'].unique():
            current_fold_quantitative_results = global_results_quantitative.loc[fold_information['Fold'] == fold]
            fold_mean_std = pd.concat([current_fold_quantitative_results.mean(), current_fold_quantitative_results.std(),
                                       current_fold_quantitative_results.median(),
                                       current_fold_quantitative_results.quantile(0.25), current_fold_quantitative_results.quantile(0.75)], axis=1,
                                       keys=['Mean fold %d' % fold, 'Std fold %d' % fold, 'Median fold %d' % fold, 
                                             'Q1 fold %d' % fold, 'Q3 fold %d' % fold])
            fold_results.append(fold_mean_std)
        fold_results = pd.concat(fold_results, axis=1)
    else:
        fold_results = overall_mean_std
    return fold_results