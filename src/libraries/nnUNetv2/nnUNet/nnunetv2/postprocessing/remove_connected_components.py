import argparse
import multiprocessing
import shutil
import warnings
from itertools import combinations
from multiprocessing import Pool
from typing import Union, Tuple, List, Callable

import numpy as np
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component, generate_ball, generic_filter_components
from batchgenerators.utilities.file_and_folder_operations import load_json, subfiles, maybe_mkdir_p, join, isfile, \
    isdir, save_pickle, load_pickle, save_json
from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.accumulate_cv_results import accumulate_cv_results
from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask, compute_metrics_on_folder, \
    load_summary_json, label_or_region_to_key
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.file_path_utilities import folds_tuple_to_string
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from skimage.morphology import binary_erosion, reconstruction, ball, disk, binary_closing
from skimage.measure import label, centroid
from scipy.ndimage import distance_transform_edt, binary_erosion


def morphological_opening_by_reconstruction(segmentation: np.ndarray,
                              labels_or_regions: Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]],
                              kernel_size: Union[int, Tuple[int, ...], List[int]],
                              background_label: int = 0,
                              use_image_spacing: bool = False,
                              props: dict = None) -> np.ndarray:
    if not isinstance(kernel_size, (tuple, list)):
        kernel_size = [kernel_size for _ in range(segmentation.ndim)]
    if use_image_spacing:
        if props is not None:
            if 'spacing' in props:
                spacing = props['spacing']
            else:
                raise ValueError('No spacing key with found in props. Keys: %s' % list(props.keys()))
        else:
            warnings.warn('use_image_spacing is True but image properties were not given. Setting spacing to 1 (same behavior as use_image_spacing=False)')
            spacing = [1] * len(kernel_size)
    else:
        spacing = [1] * len(kernel_size)
    mask = np.zeros_like(segmentation, dtype=bool)
    if not isinstance(labels_or_regions, list):
        labels_or_regions = [labels_or_regions]
    for l_or_r in labels_or_regions:
        mask |= region_or_label_to_mask(segmentation, l_or_r)
    ball = generate_ball(kernel_size, spacing)
    mask_keep = reconstruction(binary_erosion(mask, ball), mask, 'dilation', ball).astype(bool)
    ret = np.copy(segmentation)
    ret[mask & ~mask_keep] = background_label
    return ret


def morphological_closing(segmentation: np.ndarray,
                          labels_or_regions: Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]],
                          kernel_size: Union[int, Tuple[int, ...], List[int]],
                          background_label: int = 0,
                          use_image_spacing: bool = False,
                          reverse_order: bool = False,
                          props: dict = None) -> np.ndarray:
    """
    Apply morphological closing operation on all labels
    Make sure there is no overlapping, if yes solve it by choosing the label with the highest or lowest value according to reverse_order (highest for reverse_order = False)
    This is what is used in sitk.LabelUniqueLabelMapFilter (https://simpleitk.org/doxygen/v2_1/html/classitk_1_1simple_1_1LabelUniqueLabelMapFilter.html#details)
    We only consider labels here, region will be treated as independent labels as otherwise 
    we don't know how to impact individual labels from region results
    """
    if not isinstance(kernel_size, (tuple, list)):
        kernel_size = [kernel_size for _ in range(segmentation.ndim)]
    if use_image_spacing:
        if props is not None:
            if 'spacing' in props:
                spacing = props['spacing']
            else:
                raise ValueError('No spacing key with found in props. Keys: %s' % list(props.keys()))
        else:
            warnings.warn('use_image_spacing is True but image properties were not given. Setting spacing to 1 (same behavior as use_image_spacing=False)')
            spacing = [1] * len(kernel_size)
    else:
        spacing = [1] * len(kernel_size)
    ball = generate_ball(kernel_size, spacing)
    # Run morphological closing on each unique label
    unique_labels = np.unique(labels_or_regions)
    mask = np.zeros(segmentation.shape + (len(unique_labels),), dtype=np.uint8)
    for i, label in enumerate(unique_labels):
        label_mask = np.zeros_like(segmentation, dtype=bool)
        label_mask[segmentation == label] = True
        label_mask = binary_closing(label_mask, ball)
        mask[..., i] = label_mask * label # We need to assign the label value in order to find min/max value after. If 1 takes the first occurence
    all_zero_mask = mask.sum(axis=-1) == 0
    if reverse_order:
        # Assign the min label in overlapping voxels
        # Since background is 0 it will be selected as min everytime so the trick is to assign the highest value to background
        mask[mask == 0] = unique_labels.max() + 1
        idx_to_keep = np.argmin(mask, axis=-1)
    else:
        # Assign the max label in overlapping voxels
        idx_to_keep = np.argmax(mask, axis=-1)
    label_to_keep = unique_labels[idx_to_keep]
    label_to_keep[all_zero_mask] = 0
    
    combined_mask = np.zeros_like(segmentation, dtype=bool)
    if not isinstance(labels_or_regions, list):
        labels_or_regions = [labels_or_regions]
    for l_or_r in labels_or_regions:
        combined_mask |= region_or_label_to_mask(segmentation, l_or_r)
    
    ret = np.copy(segmentation)  # do not modify the input!
    ret[combined_mask] = background_label
    # Only add values where no 0 in final seg to avoid conflict
    ret[ret != 0] += label_to_keep[ret != 0]
    return ret


def remove_outside_palate(segmentation: np.ndarray,
                          labels_or_regions: Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]],
                          props: dict,
                          background_label: int = 0,
                          dist_threshold: float = 65, # values in mm extracted from analysis of nancy dataset by measuring the distance between infero-anterior corner ([128,0,0] or [128,0,-1] depending on brain orientation) and limit of the brain (obtained using skullstriped mask)
                          remove_method: str = 'centroid') -> np.ndarray:
    if 'spacing' in props:
        spacing = props['spacing']
    else:
        raise ValueError('No spacing key with found in props. Keys: %s' % list(props.keys()))
    
    if 'sitk_stuff' in props and 'direction' in props['sitk_stuff']:
        direction = props['sitk_stuff']['direction']
    else:
        if not 'sitk_stuff' in props:
            raise ValueError('This function only works with sitk reader. No sitk_stuff key found in props. Keys: %s' % list(props.keys()))
        else:
            raise ValueError('No direction key with found in sitk_stuff. Keys: %s' % list(props['sitk_stuff'].keys()))

    if direction[-3:] == (0.0, 0.0, -1.0):
        brain_z_revert = True
    elif direction[-3:] ==  (0.0, 0.0, 1.0):
        brain_z_revert = False
    else:
        raise ValueError("Can't determine if brain is revert from z orientation: %s" % direction[-3:])

    if remove_method not in ['any', 'all', 'centroid']:
        raise ValueError('remove_method is %s. Must be one of all, any or centroid.' % remove_method)

    mask = np.zeros_like(segmentation, dtype=bool)
    if not isinstance(labels_or_regions, list):
        labels_or_regions = [labels_or_regions]
    for l_or_r in labels_or_regions:
        mask |= region_or_label_to_mask(segmentation, l_or_r)
    
    if remove_method in ['any', 'all']:
        if len(mask.shape) == 3:
            strel = ball(1)
        elif len(mask.shape) == 2:
            strel = disk(1)
        else:
            raise RuntimeError()
    else:
        strel = None

    labeled_image, num_components = label(mask, return_num=True)
    labels_to_keep = []
    for i in range(num_components):
        current_label_image = labeled_image == (i + 1)
        if remove_method == 'centroid':
            # Get centroid
            label_centroid = centroid(current_label_image, spacing=spacing)
            # Compute distance with our point of interest (that could be either in [0, 0, 128] or [-1, 0, 128] depending on brain orientation)
            ref_point_coords = np.multiply([mask.shape[0] - 1, 0, 128], spacing) if brain_z_revert else np.multiply([0, 0, 128])
            distance_to_ref = np.linalg.norm(label_centroid - ref_point_coords)
            # Remove if inferior to threshold
            if distance_to_ref > dist_threshold:
                labels_to_keep.append(i + 1)
        else:
            # Get seg border
            contour = current_label_image.astype(np.uint8) - binary_erosion(np.copy(current_label_image), strel)
            # Get distance map from seg border
            ref_point_mask = np.zeros_like(mask)
            if brain_z_revert:
                ref_point_mask[mask.shape[0] - 1, 0, 128] = 1
            else:
                ref_point_mask[0, 0, 128] = 1
            distance_from_ref = distance_transform_edt(ref_point_mask == 0, spacing)
            distance_from_ref_on_contour = distance_from_ref[contour == 1]
            if remove_method == 'any' and np.all(distance_from_ref_on_contour > dist_threshold):
                # if any point on the seg border is < dist_threshold remove
                labels_to_keep.append(i + 1)
            elif remove_method == 'all' and np.any(distance_from_ref_on_contour > dist_threshold):
                # if all points on the seg border are < dist_threshold remove
                labels_to_keep.append(i + 1)

    mask_keep = np.in1d(labeled_image.ravel(), labels_to_keep).reshape(labeled_image.shape)
    ret = np.copy(segmentation)
    ret[mask & ~mask_keep] = background_label
    return ret


def remove_homolateral_healthy_brain(segmentation: np.ndarray,
                                     labels_or_regions: Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]],
                                     label_dict: dict,
                                     props: dict,
                                     background_label: int = 0):
    if not ('tumor' in label_dict and 'brain' in label_dict):
        raise ValueError('brain and tumor labels are mandatory for this function. Labels: %s' % label_dict)
    
    if 'spacing' in props:
        spacing = props['spacing']
    else:
        raise ValueError('No spacing key with found in props. Keys: %s' % list(props.keys()))
    
    mask = np.zeros_like(segmentation, dtype=bool)
    if not isinstance(labels_or_regions, list):
        labels_or_regions = [labels_or_regions]
    for l_or_r in labels_or_regions:
        mask |= region_or_label_to_mask(segmentation, l_or_r)
    
    tumor_mask = segmentation == label_dict['tumor']
    brain_mask = segmentation == label_dict['brain']
    # Get each component in brain mask with its centroid
    brain_labeled_image, num_components = label(brain_mask, return_num=True)
    if num_components > 1:
        if num_components > 2:
            # Keep two biggest
            filter_fn = lambda x, y: [i for i, j in zip(x, y) if j in sorted(y)[-2:]]
            brain_mask_keep = generic_filter_components(brain_mask, filter_fn)
            brain_mask[~brain_mask_keep] = background_label
            brain_labeled_image, num_components = label(brain_mask, return_num=True)
        component_sizes = {i + 1: j for i, j in enumerate(np.bincount(brain_labeled_image.ravel())[1:])}
        big_diff = (component_sizes[1] > 2 * component_sizes[2]) or (component_sizes[2] > 2 * component_sizes[1])
        if np.any(tumor_mask) and not big_diff:
            brain_component_centroid = {i + 1: centroid(brain_labeled_image == (i+1), spacing=spacing) for i in range(num_components)}
            tumor_centroid = centroid(tumor_mask, spacing=spacing)
            diff_centroid = {label: np.linalg.norm(label_centroid - tumor_centroid) 
                            for label, label_centroid in brain_component_centroid.items()}
            filter_fn = lambda x, y: [i for i, j in zip(x, y) if j == max(y)]
            brain_label_to_keep = filter_fn(list(diff_centroid.keys()), list(diff_centroid.values()))
        else:
            # If no tumor mask or to big diff in size, keep largest label
            filter_fn = lambda x, y: [i for i, j in zip(x, y) if j == max(y)]
            component_sizes = {i + 1: j for i, j in enumerate(np.bincount(brain_labeled_image.ravel())[1:])}
            brain_label_to_keep = filter_fn(list(component_sizes.keys()), list(component_sizes.values()))
    else:
        brain_label_to_keep = [i + 1 for i in range(num_components)] # n_components could be 1 or 0
    mask_keep = np.in1d(brain_labeled_image.ravel(), brain_label_to_keep).reshape(brain_labeled_image.shape)

    ret = np.copy(segmentation)
    ret[ret == label_dict['brain'] & ~mask_keep] = background_label
    return ret

def remove_all_but_largest_component_from_segmentation(segmentation: np.ndarray,
                                                       labels_or_regions: Union[int, Tuple[int, ...],
                                                                                List[Union[int, Tuple[int, ...]]]],
                                                       background_label: int = 0) -> np.ndarray:
    mask = np.zeros_like(segmentation, dtype=bool)
    if not isinstance(labels_or_regions, list):
        labels_or_regions = [labels_or_regions]
    for l_or_r in labels_or_regions:
        mask |= region_or_label_to_mask(segmentation, l_or_r)
    mask_keep = remove_all_but_largest_component(mask)
    ret = np.copy(segmentation)  # do not modify the input!
    ret[mask & ~mask_keep] = background_label
    return ret


def apply_postprocessing(segmentation: np.ndarray, pp_fns: List[Callable], pp_fn_kwargs: List[dict]):
    for fn, kwargs in zip(pp_fns, pp_fn_kwargs):
        segmentation = fn(segmentation, **kwargs)
    return segmentation


def load_postprocess_save(segmentation_file: str,
                          output_fname: str,
                          image_reader_writer: BaseReaderWriter,
                          pp_fns: List[Callable],
                          pp_fn_kwargs: List[dict]):
    seg, props = image_reader_writer.read_seg(segmentation_file)
    for kwargs in pp_fn_kwargs:
        if kwargs.get('need_props'):
            kwargs.pop('need_props')
            kwargs['props'] = props
    seg = apply_postprocessing(seg[0], pp_fns, pp_fn_kwargs)
    image_reader_writer.write_seg(seg, output_fname, props)


POSTPROCESSING_GLOBAL_FUNCTIONS = {
    "remove_largest_component_global": remove_all_but_largest_component_from_segmentation,
    "morphological_opening_by_reconstruction_global": morphological_opening_by_reconstruction,
    "morphological_closing_global": morphological_closing,
    "remove_outside_palate": remove_outside_palate,
    "remove_homolateral_healthy_brain": remove_homolateral_healthy_brain
}

POSTPROCESSING_LABEL_REGION_FUNCTIONS = {
    # No problem of overlapping label here as it will only remove some part. Morphological closing instead could be problematic
    "morphological_opening_by_reconstruction_label_region": morphological_opening_by_reconstruction,
    # We should be careful with morphological closing it could results in overlap. We handle this by assign min/max label value in case of overlap
    # In our case (seg of glioma and brain) it shouldn't be a problem
    "morphological_closing_label_region": morphological_closing,
    "remove_largest_component_label_region": remove_all_but_largest_component_from_segmentation,
}


def determine_postprocessing(folder_predictions: str,
                             folder_ref: str,
                             plans_file_or_dict: Union[str, dict],
                             dataset_json_file_or_dict: Union[str, dict],
                             num_processes: int = default_num_processes,
                             keep_postprocessed_files: bool = True):
    """
    Determines nnUNet postprocessing. Its output is a postprocessing.pkl file in folder_predictions which can be
    used with apply_postprocessing_to_folder.

    Postprocessed files are saved in folder_predictions/postprocessed. Set
    keep_postprocessed_files=False to delete these files after this function is done (temp files will eb created
    and deleted regardless).

    If plans_file_or_dict or dataset_json_file_or_dict are None, we will look for them in input_folder
    
    All combinations of functions in POSTPROCESSING_FUNCTIONS will be tested. 
    """
    output_folder = join(folder_predictions, 'postprocessed')

    if plans_file_or_dict is None:
        expected_plans_file = join(folder_predictions, 'plans.json')
        if not isfile(expected_plans_file):
            raise RuntimeError(f"Expected plans file missing: {expected_plans_file}. The plans files should have been "
                               f"created while running nnUNetv2_predict. Sadge.")
        plans_file_or_dict = load_json(expected_plans_file)
    plans_manager = PlansManager(plans_file_or_dict)

    if dataset_json_file_or_dict is None:
        expected_dataset_json_file = join(folder_predictions, 'dataset.json')
        if not isfile(expected_dataset_json_file):
            raise RuntimeError(
                f"Expected plans file missing: {expected_dataset_json_file}. The plans files should have been "
                f"created while running nnUNetv2_predict. Sadge.")
        dataset_json_file_or_dict = load_json(expected_dataset_json_file)

    if not isinstance(dataset_json_file_or_dict, dict):
        dataset_json = load_json(dataset_json_file_or_dict)
    else:
        dataset_json = dataset_json_file_or_dict

    rw = plans_manager.image_reader_writer_class()
    label_manager = plans_manager.get_label_manager(dataset_json)
    labels_or_regions = label_manager.foreground_regions if label_manager.has_regions else label_manager.foreground_labels

    predicted_files = subfiles(folder_predictions, suffix=dataset_json['file_ending'], join=False)
    ref_files = subfiles(folder_ref, suffix=dataset_json['file_ending'], join=False)
    # we should print a warning if not all files from folder_ref are present in folder_predictions
    if not all([i in predicted_files for i in ref_files]):
        print(f'WARNING: Not all files in folder_ref were found in folder_predictions. Determining postprocessing '
              f'should always be done on the entire dataset!')

    # before we start we should evaluate the imaegs in the source folder
    if not isfile(join(folder_predictions, 'summary.json')):
        compute_metrics_on_folder(folder_ref,
                                  folder_predictions,
                                  join(folder_predictions, 'summary.json'),
                                  rw,
                                  dataset_json['file_ending'],
                                  labels_or_regions,
                                  label_manager.ignore_label,
                                  num_processes)
    baseline_results = load_summary_json(join(folder_predictions, 'summary.json'))

    # Remove pp functions that can't be used in current case + set kwargs
    pp_fns_global_to_test = POSTPROCESSING_GLOBAL_FUNCTIONS.copy()
    pp_fns_global_to_remove = []
    for fn_name in pp_fns_global_to_test:
        if fn_name == 'remove_outside_palate':
            if not isinstance(rw, SimpleITKIO):
                pp_fns_global_to_remove.append(fn_name)
        if fn_name == 'remove_homolateral_healthy_brain':
            if not ('tumor' in label_manager.label_dict and 'brain' in label_manager.label_dict):
                pp_fns_global_to_remove.append(fn_name)
        if fn_name == 'morphological_opening_by_reconstruction_global' or fn_name == 'morphological_closing_global':
            if len(labels_or_regions) > 1:
                pp_fns_global_to_remove.append(fn_name)
    for fn_name in pp_fns_global_to_remove:
        pp_fns_global_to_test.pop(fn_name)
    pp_fns_global_to_test_kwargs = {}
    for fn_name in pp_fns_global_to_test:
        pp_fns_global_to_test_kwargs[fn_name] = {'labels_or_regions': label_manager.foreground_labels}
        if fn_name != 'remove_largest_component_global':
            pp_fns_global_to_test_kwargs[fn_name]['need_props'] = True
            if fn_name == 'morphological_opening_by_reconstruction_global':
                pp_fns_global_to_test_kwargs[fn_name].update({'kernel_size': 1, 'use_image_spacing': False})
            elif fn_name == 'remove_outside_palate':
                pp_fns_global_to_test_kwargs[fn_name].update({'remove_method': 'centroid'})
            elif fn_name == 'remove_homolateral_healthy_brain':
                pp_fns_global_to_test_kwargs[fn_name].update({'label_dict': label_manager.label_dict})
            elif fn_name == 'morphological_closing_global':
                pp_fns_global_to_test_kwargs[fn_name].update({'kernel_size': 3, 'use_image_spacing': False, 'reverse_order': False})
    pp_fns_global_comb = [list(combo) for i in range(1, len(pp_fns_global_to_test) + 1) for combo in combinations(pp_fns_global_to_test.keys(), i)]

    if len(labels_or_regions) > 1:
        pp_fns_label_to_test = POSTPROCESSING_LABEL_REGION_FUNCTIONS.copy()
        pp_fns_label_to_test_kwargs = {'morphological_opening_by_reconstruction_label_region': {'need_props': True, 'kernel_size': 1, 'use_image_spacing': False},
                                       'morphological_closing_label_region': {'need_props': True, 'kernel_size': 3, 'use_image_spacing': False, 'reverse_order': False}}
    else:
        pp_fns_label_to_test = {}
        pp_fns_label_to_test_kwargs = {}
    print('pp labels fns tested: %s' % list(pp_fns_label_to_test.keys()))

    # we save the postprocessing functions in here
    pp_fns = []
    pp_fn_kwargs = []
    source = folder_predictions
    best_results = baseline_results

    # pool party!
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        for fn_name_list in pp_fns_global_comb:
            output_here = join(output_folder, 'temp', '_'.join(fn_name_list))
            maybe_mkdir_p(output_here)

            pool.starmap(
                load_postprocess_save,
                zip(
                    [join(folder_predictions, i) for i in predicted_files],
                    [join(output_here, i) for i in predicted_files],
                    [rw] * len(predicted_files),
                    [[pp_fns_global_to_test[_] for _ in fn_name_list]] * len(predicted_files),
                    [[pp_fns_global_to_test_kwargs[_] for _ in fn_name_list]] * len(predicted_files)
                )
            )
            compute_metrics_on_folder(folder_ref,
                                      output_here,
                                      join(output_here, 'summary.json'),
                                      rw,
                                      dataset_json['file_ending'],
                                      labels_or_regions,
                                      label_manager.ignore_label,
                                      num_processes)
            # now we need to figure out if doing this improved the dice scores. We will implement that defensively in so far
            # that if a single class got worse as a result we won't do this. We can change this in the future but right now I
            # prefer to do it this way            
            pp_results = load_summary_json(join(output_here, 'summary.json'))
            do_this = pp_results['foreground_mean']['Dice'] > best_results['foreground_mean']['Dice']
            if do_this:
                for class_id in pp_results['mean'].keys():
                    if pp_results['mean'][class_id]['Dice'] < best_results['mean'][class_id]['Dice']:
                        do_this = False
                        break
            if do_this:
                print(f'Results were improved with postprocessing with {fn_name_list}. '
                    f'Mean dice before: {round(best_results["foreground_mean"]["Dice"], 5)} '
                    f'after: {round(pp_results["foreground_mean"]["Dice"], 5)}')
                best_results = pp_results
                source = output_here
                pp_fns = [pp_fns_global_to_test[_] for _ in fn_name_list]
                pp_fn_kwargs = [pp_fns_global_to_test_kwargs[_] for _ in fn_name_list]
            else:
                print(f'Postprocessing with {fn_name_list} did not improve results!')

        # in the old nnU-Net we could just apply all-but-largest component removal to all classes at the same time and
        # then evaluate for each class whether this improved results. This is no longer possible because we now support
        # region-based predictions and regions can overlap, causing interactions
        # in principle the order with which the postprocessing is applied to the regions matter as well and should be
        # investigated, but due to some things that I am too lazy to explain right now it's going to be alright (I think)
        # to stick to the order in which they are declared in dataset.json (if you want to think about it then think about
        # region_class_order)
        # 2023_02_06: I hate myself for the comment above. Thanks past me
        # TZAR : Maybe we can do one big comb list with label_or_region pp methods on each label but I am not sure it does the same thing.
        # I think that using something similar to global fns is harder to do here
        if len(labels_or_regions) > 1:
            for fn_name in pp_fns_label_to_test:
                for label_or_region in labels_or_regions:
                    pp_fn = pp_fns_label_to_test[fn_name]
                    kwargs = pp_fns_label_to_test_kwargs[fn_name].copy() if fn_name in pp_fns_label_to_test_kwargs else {}
                    kwargs.update({'labels_or_regions': label_or_region})

                    output_here = join(output_folder, 'temp', fn_name)
                    maybe_mkdir_p(output_here)

                    pool.starmap(
                        load_postprocess_save,
                        zip(
                            [join(source, i) for i in predicted_files],
                            [join(output_here, i) for i in predicted_files],
                            [rw] * len(predicted_files),
                            [[pp_fn]] * len(predicted_files),
                            [[kwargs]] * len(predicted_files)
                        )
                    )
                    compute_metrics_on_folder(folder_ref,
                                              output_here,
                                              join(output_here, 'summary.json'),
                                              rw,
                                              dataset_json['file_ending'],
                                              labels_or_regions,
                                              label_manager.ignore_label,
                                              num_processes)
                    pp_results = load_summary_json(join(output_here, 'summary.json'))
                    do_this = pp_results['mean'][label_or_region]['Dice'] > best_results['mean'][label_or_region]['Dice']
                    if do_this:
                        print(f'Results were improved by postprocessing with {fn_name} for {label_or_region}. '
                            f'Dice before: {round(best_results["mean"][label_or_region]["Dice"], 5)} '
                            f'after: {round(pp_results["mean"][label_or_region]["Dice"], 5)}')
                        if isdir(join(output_folder, 'temp', '%s_currentBest' % fn_name)):
                            shutil.rmtree(join(output_folder, 'temp', '%s_currentBest' % fn_name))
                        shutil.move(output_here, join(output_folder, 'temp', '%s_currentBest' % fn_name), )
                        source = join(output_folder, 'temp', '%s_currentBest' % fn_name)
                        pp_fns.append(pp_fn)
                        pp_fn_kwargs.append(kwargs)
                    else:
                        print(f'Postprocessing with {fn_name} for {label_or_region} did not improve results! '
                            f'Dice before: {round(best_results["mean"][label_or_region]["Dice"], 5)} '
                            f'after: {round(pp_results["mean"][label_or_region]["Dice"], 5)}')
    [shutil.copy(join(source, i), join(output_folder, i)) for i in subfiles(source, join=False)]
    save_pickle((pp_fns, pp_fn_kwargs), join(folder_predictions, 'postprocessing.pkl'))

    # Recompute metrics other prediction_file path is not good in sumarry.json
    compute_metrics_on_folder(folder_ref, output_folder, join(output_folder, 'summary.json'), rw,
                              dataset_json['file_ending'], labels_or_regions, label_manager.ignore_label, num_processes)
    final_results = load_summary_json(join(output_folder, 'summary.json'))
    tmp = {
        'input_folder': {i: baseline_results[i] for i in ['foreground_mean', 'mean']},
        'postprocessed': {i: final_results[i] for i in ['foreground_mean', 'mean']},
        'postprocessing_fns': [i.__name__ for i in pp_fns],
        'postprocessing_kwargs': pp_fn_kwargs,
    }
    # json is very annoying. Can't handle tuples as dict keys.
    tmp['input_folder']['mean'] = {label_or_region_to_key(k): tmp['input_folder']['mean'][k] for k in
                                   tmp['input_folder']['mean'].keys()}
    tmp['postprocessed']['mean'] = {label_or_region_to_key(k): tmp['postprocessed']['mean'][k] for k in
                                    tmp['postprocessed']['mean'].keys()}
    # did I already say that I hate json? "TypeError: Object of type int64 is not JSON serializable"
    recursive_fix_for_json_export(tmp)
    save_json(tmp, join(folder_predictions, 'postprocessing.json'))

    shutil.rmtree(join(output_folder, 'temp'))

    if not keep_postprocessed_files:
        shutil.rmtree(output_folder)
    return pp_fns, pp_fn_kwargs


def apply_postprocessing_to_folder(input_folder: str,
                                   output_folder: str,
                                   pp_fns: List[Callable],
                                   pp_fn_kwargs: List[dict],
                                   plans_file_or_dict: Union[str, dict] = None,
                                   dataset_json_file_or_dict: Union[str, dict] = None,
                                   num_processes=8) -> None:
    """
    If plans_file_or_dict or dataset_json_file_or_dict are None, we will look for them in input_folder
    """
    if plans_file_or_dict is None:
        expected_plans_file = join(input_folder, 'plans.json')
        if not isfile(expected_plans_file):
            raise RuntimeError(f"Expected plans file missing: {expected_plans_file}. The plans file should have been "
                               f"created while running nnUNetv2_predict. Sadge. If the folder you want to apply "
                               f"postprocessing to was create from an ensemble then just specify one of the "
                               f"plans files of the ensemble members in plans_file_or_dict")
        plans_file_or_dict = load_json(expected_plans_file)
    plans_manager = PlansManager(plans_file_or_dict)

    if dataset_json_file_or_dict is None:
        expected_dataset_json_file = join(input_folder, 'dataset.json')
        if not isfile(expected_dataset_json_file):
            raise RuntimeError(
                f"Expected plans file missing: {expected_dataset_json_file}. The dataset.json should have been "
                f"copied while running nnUNetv2_predict/nnUNetv2_ensemble. Sadge.")
        dataset_json_file_or_dict = load_json(expected_dataset_json_file)

    if not isinstance(dataset_json_file_or_dict, dict):
        dataset_json = load_json(dataset_json_file_or_dict)
    else:
        dataset_json = dataset_json_file_or_dict

    rw = plans_manager.image_reader_writer_class()

    maybe_mkdir_p(output_folder)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        files = subfiles(input_folder, suffix=dataset_json['file_ending'], join=False)

        _ = p.starmap(load_postprocess_save,
                      zip(
                          [join(input_folder, i) for i in files],
                          [join(output_folder, i) for i in files],
                          [rw] * len(files),
                          [pp_fns] * len(files),
                          [pp_fn_kwargs] * len(files)
                      )
                      )


def entry_point_determine_postprocessing_folder():
    parser = argparse.ArgumentParser('Writes postprocessing.pkl and postprocessing.json in input_folder.')
    parser.add_argument('-i', type=str, required=True, help='Input folder')
    parser.add_argument('-ref', type=str, required=True, help='Folder with gt labels')
    parser.add_argument('-plans_json', type=str, required=False, default=None,
                        help="plans file to use. If not specified we will look for the plans.json file in the "
                             "input folder (input_folder/plans.json)")
    parser.add_argument('-dataset_json', type=str, required=False, default=None,
                        help="dataset.json file to use. If not specified we will look for the dataset.json file in the "
                             "input folder (input_folder/dataset.json)")
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f"number of processes to use. Default: {default_num_processes}")
    parser.add_argument('--remove_postprocessed', action='store_true', required=False,
                        help='set this is you don\'t want to keep the postprocessed files')

    args = parser.parse_args()
    determine_postprocessing(args.i, args.ref, args.plans_json, args.dataset_json, args.np,
                             not args.remove_postprocessed)


def entry_point_apply_postprocessing():
    parser = argparse.ArgumentParser('Apples postprocessing specified in pp_pkl_file to input folder.')
    parser.add_argument('-i', type=str, required=True, help='Input folder')
    parser.add_argument('-o', type=str, required=True, help='Output folder')
    parser.add_argument('-pp_pkl_file', type=str, required=True, help='postprocessing.pkl file')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f"number of processes to use. Default: {default_num_processes}")
    parser.add_argument('-plans_json', type=str, required=False, default=None,
                        help="plans file to use. If not specified we will look for the plans.json file in the "
                             "input folder (input_folder/plans.json)")
    parser.add_argument('-dataset_json', type=str, required=False, default=None,
                        help="dataset.json file to use. If not specified we will look for the dataset.json file in the "
                             "input folder (input_folder/dataset.json)")
    args = parser.parse_args()
    pp_fns, pp_fn_kwargs = load_pickle(args.pp_pkl_file)
    apply_postprocessing_to_folder(args.i, args.o, pp_fns, pp_fn_kwargs, args.plans_json, args.dataset_json, args.np)


if __name__ == '__main__':
    trained_model_folder = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetTrainer__nnUNetPlans__3d_fullres'
    labelstr = join(nnUNet_raw, 'Dataset004_Hippocampus', 'labelsTr')
    plans_manager = PlansManager(join(trained_model_folder, 'plans.json'))
    dataset_json = load_json(join(trained_model_folder, 'dataset.json'))
    folds = (0, 1, 2, 3, 4)
    label_manager = plans_manager.get_label_manager(dataset_json)

    merged_output_folder = join(trained_model_folder, f'crossval_results_folds_{folds_tuple_to_string(folds)}')
    accumulate_cv_results(trained_model_folder, merged_output_folder, folds, 8, False)

    fns, kwargs = determine_postprocessing(merged_output_folder, labelstr, plans_manager.plans,
                                           dataset_json, 8, keep_postprocessed_files=True)
    save_pickle((fns, kwargs), join(trained_model_folder, 'postprocessing.pkl'))
    fns, kwargs = load_pickle(join(trained_model_folder, 'postprocessing.pkl'))

    apply_postprocessing_to_folder(merged_output_folder, merged_output_folder + '_pp', fns, kwargs,
                                   plans_manager.plans, dataset_json,
                                   8)
    compute_metrics_on_folder(labelstr,
                              merged_output_folder + '_pp',
                              join(merged_output_folder + '_pp', 'summary.json'),
                              plans_manager.image_reader_writer_class(),
                              dataset_json['file_ending'],
                              label_manager.foreground_regions if label_manager.has_regions else label_manager.foreground_labels,
                              label_manager.ignore_label,
                              8)
