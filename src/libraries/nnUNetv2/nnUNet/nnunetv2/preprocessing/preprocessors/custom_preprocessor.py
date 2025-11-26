#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import multiprocessing
import shutil
from time import sleep
from typing import Tuple, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.preprocessing.cropping.cropping import crop_to_brain
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero


class CroppedBrainPreprocessor(DefaultPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose=verbose)

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        # let's not mess up the inputs!
        data = data.astype(np.float32)  # this creates a copy
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_brain(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
            # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
            # LabelManager is pretty light computation-wise.
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append(label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                                   verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        return data, seg
    

class AuxiliaryClassificationPreprocessor(DefaultPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose=verbose)

    def run_case(self, image_files, seg_file, plans_manager, configuration_manager, dataset_json):
        data, seg, data_properties = super().run_case(image_files, seg_file, plans_manager, configuration_manager, dataset_json)

        if seg_file is not None:
            # If possible, load json file where auxiliary classification label is stored
            # From Tiff3DIO
            # see if aux file can be found
            expected_aux_file = seg_file[:-len(dataset_json['file_ending'])] + '.json'
            if isfile(expected_aux_file):
                data_properties['auxiliary_classifier_label'] = load_json(expected_aux_file)['auxiliary_classifier_label']
            else:
                data_properties['auxiliary_classifier_label'] = None   
        return data, seg, data_properties
    
    
class AuxiliaryRegressionPreprocessor(DefaultPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose=verbose)

    def run_case(self, image_files, seg_file, plans_manager, configuration_manager, dataset_json):
        data, seg, data_properties = super().run_case(image_files, seg_file, plans_manager, configuration_manager, dataset_json)

        if seg_file is not None:
            # If possible, load json file where auxiliary regression label is stored
            # From Tiff3DIO
            # see if aux file can be found
            expected_aux_file = seg_file[:-len(dataset_json['file_ending'])] + '.json'
            if isfile(expected_aux_file):
                data_properties['auxiliary_regressor_label'] = load_json(expected_aux_file)['auxiliary_regressor_label']
            else:
                data_properties['auxiliary_regressor_label'] = None   
        return data, seg, data_properties
    
    
class AuxiliaryRegressionThresholdPreprocessor(DefaultPreprocessor):
    """
    Class for preprocessing data in the context of an auxialiary task for predicting the segmentation threshold
    on our PET image. We assume here that we have only one threshold to predict and that this threshold is linked
    to the image in the first channel, i.e. PET in our case.
    The aim of this class is to normalize the threshold value as the image
    """
    def __init__(self, verbose: bool = True):
        super().__init__(verbose=verbose)
        
    def run_case(self, image_files, seg_file, plans_manager, configuration_manager, dataset_json):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()

        # load image(s)
        data, data_properties = rw.read_images(image_files)

        # if possible, load seg
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)
        else:
            seg = None
            
        print(seg is not None)
            
        # Compare to AuxiliaryRegressionPreprocessor and AuxiliaryClassificationPreprocessor we have to replace the function
        # as we need the updated data_properties in run_case_npy
        if seg_file is not None:
            # If possible, load json file where auxiliary regression label is stored
            # From Tiff3DIO
            # see if aux file can be found
            expected_aux_file = seg_file[:-len(dataset_json['file_ending'])] + '.json'
            print(expected_aux_file, isfile(expected_aux_file))
            if isfile(expected_aux_file):
                print(load_json(expected_aux_file))
                data_properties['auxiliary_regressor_label'] = load_json(expected_aux_file)['auxiliary_regressor_label']
            else:
                data_properties['auxiliary_regressor_label'] = None

        data, seg = self.run_case_npy(data, seg, data_properties, plans_manager, configuration_manager,
                                      dataset_json)
        return data, seg, data_properties
    
    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        # We also have to copy paste this function because we need to get the normalization value before resampling
        
        # let's not mess up the inputs!
        data = data.astype(np.float32)  # this creates a copy
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        old_data = data.copy()
        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)
        
        if has_seg:
            # We also need to normalize the threshold using the same method as the images
            # If we don't have seg, properties are not loaded (filename based on seg filename)
            # This happen only during inference (normally) so check if seg is given is sufficient
            normalization_ratio = self.get_normalization_ratio(old_data, data, seg)
            properties['auxiliary_regressor_label'] = [properties['auxiliary_regressor_label'][0] * normalization_ratio]

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
            # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
            # LabelManager is pretty light computation-wise.
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append(label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                                   verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        return data, seg
     
    @staticmethod    
    def get_normalization_ratio(original_image, normalized_image, seg=None):
        """
        Extract the ratio between original and normalized image for threshold normalization
        As nnunet default normalization schemes are applied to the whole image, thus if we compute the 
        ratio in the most restrictive case it should be ok.
        This most restrictive case is : 
        seg >= 0 (in ZScoreNormalization if use_mask_for_norm) and original_image !=0 (0 values may not change after norm)
        
        Args:
            normalizer: Instance of ImageNormalization class
            original_image: Original image before normalization
            seg: Segmentation mask (if needed)
        
        Returns:
            ratio: Single value to multiply your threshold by
        """
        # Most restrictive mask: both conditions
        if seg is not None:
            mask = (seg >= 0) & (original_image != 0)
        else:
            mask = original_image != 0
        
        if mask.any():
            original_mean = original_image[mask].mean()
            normalized_mean = normalized_image[mask].mean()
            ratio = normalized_mean / original_mean if original_mean != 0 else 1.0
        else:
            ratio = 1.0
        
        return ratio
        