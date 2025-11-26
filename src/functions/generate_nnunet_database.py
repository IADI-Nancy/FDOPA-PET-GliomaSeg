import argparse
import os
import re
import warnings
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
from ..utils import data_utils
from joblib import Parallel, delayed
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import write_json


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate nn-Unet database')
    parser.add_argument("--train_images_root", help='Root directory of train image dataset', 
                        type=str, required=True)
    parser.add_argument("--train_data_info_file", help='file with information on train population to exclude patients',
                        type=str, required=True)
    parser.add_argument("--test_images_root", help='Root directory of test image dataset', 
                        type=str, default=None)
    parser.add_argument("--test_data_info_file", help='file with information on test population to exclude patients',
                        type=str, default=None)
    parser.add_argument("--dataset", help='Dataset name', 
                        required=True, type=str)
    parser.add_argument("--skull_stripping", help='Choose whether to use skull stripped image',
                        action='store_true', default=False)
    parser.add_argument("--brain_centering", help='Choose whether to use skull stripped image',
                        action='store_true', default=False)
    parser.add_argument("--ref_spacing", 
                        help='Float value indicating the reference isotropic spacing the images will be resampled to',
                        type=float, default=1)
    parser.add_argument("--ref_size", 
                        help='Tuple with 3 values indicating the reference size the images will be resampled to',
                        nargs=3, type=int, default=(256, 256, 164))
    parser.add_argument('--labels', help='Labels to train the network',
                        default='tumor_brain', choices=['only_tumor', 'tumor_brain'])
    parser.add_argument("--threads", help='Number of parallel threads',
                        default=-1, type=int)
    parser.add_argument("--preprocessor", help='Name of the preprocessor',
                        type=str, default='DefaultPreprocessor')
    parser.add_argument("--architecture", help='Choice of architecture. Will set up the approriate Planner and preprocessor if needed',
                        type=str, default='default', choices=['default', 'AuxiliaryClassification', 'AuxiliaryRegression',
                                                              'ContrastiveSupervision', 'scSE', 'PE'])
    args = parser.parse_args()

    nnUNet_root = os.environ.get('nnUNet_raw')
    if not re.match(re.compile('Dataset[0-9]*'), args.dataset):
        start_n_dataset = 500
        dataset_number_list = sorted([int(re.split('Dataset|_', _)[1]) for _ in os.listdir(os.path.abspath(nnUNet_root))])
        contiguous_numbers = np.diff(dataset_number_list) == 1
        if np.all(contiguous_numbers):
            n_dataset = 500 + len(os.listdir(os.path.abspath(nnUNet_root))) + 1
        else:
            idx_last_contiguous = np.argmin(contiguous_numbers)
            n_dataset = dataset_number_list[idx_last_contiguous] + 1
        args.dataset = 'Dataset%s_%s' % (str(n_dataset).zfill(3), args.dataset)
    else:
        warnings.warn('WARNING %s already exists and will be overwritten.' % args.dataset)

    if (args.test_images_root is not None and args.test_data_info_file is None) or (args.test_images_root is None and args.test_data_info_file is not None):
        raise ValueError('If one of test_images_root or test_data_info_file is given, the other must also be given.')

    dataset_path = os.path.join(os.path.abspath(nnUNet_root), args.dataset)
    image_train_dir = os.path.join(dataset_path, 'imagesTr')
    label_train_dir = os.path.join(dataset_path, 'labelsTr')
    dir_list = [image_train_dir, label_train_dir]
    if args.test_images_root is not None:
        image_test_dir = os.path.join(dataset_path, 'imagesTs')
        label_test_dir = os.path.join(dataset_path, 'labelsTs')
        dir_list += [image_test_dir, label_test_dir]
    for i in dir_list:
        os.makedirs(i, exist_ok=True)
        
    # save command for retrieve parameters later (ex load template images)
    with open(os.path.join(dataset_path, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.labels == 'tumor_brain':
        n_labels = 2
    elif args.labels == 'only_tumor':
        n_labels = 1
    else:
        raise ValueError
    
    if '.xlsx' in args.train_data_info_file:
        train_data_info_file = pd.read_excel(os.path.abspath(args.train_data_info_file), index_col=0, dtype={'ID': str})
    elif '.csv' in args.train_data_info_file:
        train_data_info_file = pd.read_csv(os.path.abspath(args.train_data_info_file), index_col=0, dtype={'ID': str})
    else:
        raise ValueError
    if args.test_data_info_file is not None:
        if '.xlsx' in args.test_data_info_file:
            test_data_info_file = pd.read_excel(os.path.abspath(args.test_data_info_file), index_col=0, dtype={'ID': str})
        elif '.csv' in args.test_data_info_file:
            test_data_info_file = pd.read_csv(os.path.abspath(args.test_data_info_file), index_col=0, dtype={'ID': str})
        else:
            raise ValueError

    train_patients_list, train_excluded_patients = data_utils.get_images_path(os.path.abspath(args.train_images_root), 
                                                                              data_info=train_data_info_file)
    if args.test_images_root is not None:
        test_patients_list, test_excluded_patients = data_utils.get_images_path(os.path.abspath(args.test_images_root),
                                                                                data_info=test_data_info_file)

    def process_patient_data(patient, patients_list, image_dir, label_dir):
        print('%d/%d' % (patients_list.index(patient) + 1, len(patients_list)))
        patient_ID = str(patients_list.index(patient)).zfill(len(str(len(patients_list))))
        # Load image with the options specified
        image, mask = data_utils.get_TEP_data_mask(patient, plane='Axial', n_labels=n_labels,
                                                   ref_spacing=args.ref_spacing, ref_size=args.ref_size,
                                                   skull_stripping=args.skull_stripping, brain_centering=args.brain_centering)
        # Convert mask from one hot encoded to required format
        final_mask = data_utils.convert_mask_nnUNet_format(mask, n_labels)
        # Save images
        sitk.WriteImage(image, os.path.join(image_dir, 'PETseg_%s_0000.nii.gz' % patient_ID))
        sitk.WriteImage(final_mask, os.path.join(label_dir, 'PETseg_%s.nii.gz' % patient_ID))

    Parallel(n_jobs=args.threads)(delayed(process_patient_data)(patient, train_patients_list, image_train_dir, label_train_dir)
                                  for patient in train_patients_list)
    if args.test_images_root is not None:
        Parallel(n_jobs=args.threads)(delayed(process_patient_data)(patient, test_patients_list, image_test_dir, label_test_dir)
                                      for patient in test_patients_list)

    # Generate link between nnUnet identifier and original identifier
    train_patient_link = {'Original ID': [], 'nnUNet ID': []}
    for _ in range(len(train_patients_list)):
        train_patient_link['Original ID'].append(os.path.basename(train_patients_list[_]))
        train_patient_link['nnUNet ID'].append(str(_).zfill(len(str(len(train_patients_list)))))
    train_patient_link_df = pd.DataFrame(train_patient_link)
    train_patient_link_df.to_csv(os.path.join(dataset_path, 'Train_Patient_link.csv'))
    if args.test_images_root is not None:
        test_patient_link = {'Original ID': [], 'nnUNet ID': []}
        for _ in range(len(test_patients_list)):
            test_patient_link['Original ID'].append(os.path.basename(test_patients_list[_]))
            test_patient_link['nnUNet ID'].append(str(_).zfill(len(str(len(test_patients_list)))))
        test_patient_link_df = pd.DataFrame(test_patient_link)
        test_patient_link_df.to_csv(os.path.join(dataset_path, 'Test_Patient_link.csv'))

    # Generate dataset json    
    if args.architecture == 'AuxiliaryClassification':
        # Encode the labels for the auxiliary classifier and store it in dataset.json. 
        # We don't use LabelEncoder to control the order, especially the 0 label that needs to be our default label (the one in case of no seg)
        auxiliary_classifier_labels = {'no_measurable': 0, 'non_measurable': 1, 'measurable': 2}
        train_aux_classif_labels = pd.Series(data=train_data_info_file.loc[train_patient_link_df['Original ID'], 'Seg_detection']).set_axis(train_patient_link_df['nnUNet ID'])
        train_aux_classif_labels.replace(auxiliary_classifier_labels, inplace=True)
        dataset_kwargs = {'auxiliary_classifier_labels': auxiliary_classifier_labels}
        # Information on label of each sample will be store in a json file (SampleName.json) that will be read later
        # It is a bit similar as when using TIFF files where a json goes with TIFF file to specify spacing
        for id, label in train_aux_classif_labels.items():
            write_json({'auxiliary_classifier_label': label}, os.path.join(label_train_dir, 'PETseg_%s.json' % id))
        if args.test_images_root is not None:
            test_aux_classif_labels = pd.Series(data=test_data_info_file.loc[test_patient_link_df['Original ID'], 'Seg_detection']).set_axis(test_patient_link_df['nnUNet ID'])
            test_aux_classif_labels.replace(auxiliary_classifier_labels, inplace=True)
            # Information on label of each sample will be store in a json file (SampleName.json) in the label folder that will be read later
            # It is a bit similar as when using TIFF files where a json goes with TIFF file to specify spacing
            for id, label in test_aux_classif_labels.items():
                write_json({'auxiliary_classifier_label': label}, os.path.join(label_test_dir, 'PETseg_%s.json' % id))
        if args.preprocessor == "DefaultPreprocessor":
            warnings.warn('AuxiliaryClassification with DefaultPreprocessor is useless as it does not handle the feature. Changing for the appropriate preprocessor.')
            args.preprocessor = 'AuxiliaryClassificationPreprocessor'
    elif args.architecture == 'AuxiliaryRegression':
        # Store the number of regression outputs in dataset.json. 
        auxiliary_regressor_num_outputs = 1
        train_aux_reg_labels = pd.Series(data=train_data_info_file.loc[train_patient_link_df['Original ID'], 'Threshold_seg']).set_axis(train_patient_link_df['nnUNet ID'])
        dataset_kwargs = {'auxiliary_regressor_num_outputs': auxiliary_regressor_num_outputs}
        # Information on label of each sample will be store in a json file (SampleName.json) that will be read later
        # It is a bit similar as when using TIFF files where a json goes with TIFF file to specify spacing
        for id, label in train_aux_reg_labels.items():
            write_json({'auxiliary_regressor_label': [label]}, os.path.join(label_train_dir, 'PETseg_%s.json' % id))
        if args.test_images_root is not None:
            test_aux_reg_labels = pd.Series(data=test_data_info_file.loc[test_patient_link_df['Original ID'], 'Threshold_seg']).set_axis(test_patient_link_df['nnUNet ID'])
            # Information on label of each sample will be store in a json file (SampleName.json) in the label folder that will be read later
            # It is a bit similar as when using TIFF files where a json goes with TIFF file to specify spacing
            for id, label in test_aux_reg_labels.items():
                write_json({'auxiliary_regressor_label': [label]}, os.path.join(label_test_dir, 'PETseg_%s.json' % id))
        if args.preprocessor == "DefaultPreprocessor":
            warnings.warn('AuxiliaryRegression with DefaultPreprocessor is useless as it does not handle the feature. Changing for the appropriate preprocessor.')
            args.preprocessor = 'AuxiliaryRegressionPreprocessor'
    else:
        dataset_kwargs = {}
            
    if n_labels == 2:
        labels_name = {'background': 0, 'brain': 1, 'tumor': 2}
    else:
        labels_name = {'background': 0, 'tumor': 1}
    
    generate_dataset_json(dataset_path, channel_names={0: 'PT'},
                          labels= labels_name,
                          num_training_cases=len(train_patients_list), file_ending='.nii.gz', dataset_name=args.dataset, **dataset_kwargs)

    dataset_number = re.sub('[^0-9]', '', args.dataset)

    configs = '2d 3d_fullres 3d_lowres'

    if args.architecture == 'default':
        planner_suffix = ''
    else:
        planner_suffix = args.architecture
    
    os.system("nnUNetv2_plan_and_preprocess -d %s --verify_dataset_integrity -c %s -preprocessor_name %s -pl ExperimentPlanner%s > %s" % (dataset_number, configs,
                                                                                                                        args.preprocessor, planner_suffix,
                                                                                                                        os.path.join(dataset_path, "preprocessed_log.txt")))

    if 'Device' in train_data_info_file and 'Recurrence' in train_data_info_file:
        # Stratify based on Device and Initial Diagnosis/Recurrence
        data_utils.stratify(link_patients=train_patient_link_df, data_info=train_data_info_file,  
                            save_path = os.path.join(os.environ.get('nnUNet_preprocessed'), args.dataset, "splits_final.json"))
    else:
        print('Fold stratification based on Device and Initial Diagnosis/Recurrence could not be performed. Default nnUNet folds are kept.')
    
    for resnet_type in ['M', 'L', 'XL']:
        os.system("nnUNetv2_plan_experiment -d %s -pl nnUNetPlannerResEnc%s%s -preprocessor_name %s" % (dataset_number, resnet_type, 
                                                                                                        planner_suffix, args.preprocessor))    