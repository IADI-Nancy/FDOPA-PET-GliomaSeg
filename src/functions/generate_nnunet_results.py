import argparse
import os
import shutil
import json
import time
import datetime
import shutil
import pandas as pd
import numpy as np
from ..utils.data_utils import get_dataset_fullname
from ..utils.evaluation_utils import extract_global_results, extract_fold_results
from ..utils.post_hoc_tests import model_post_hoc_tests, rank_and_compare_models
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle, save_json
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2, label_or_region_to_key
from nnunetv2.postprocessing.remove_connected_components import remove_homolateral_healthy_brain, morphological_opening_by_reconstruction


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate nn-Unet results')
    parser.add_argument("--recap_results_file", help='file in which recapitulative results are saved', required=True, type=str)
    parser.add_argument("--dataset", help='Dataset name or dataset number', required=True, type=str)
    parser.add_argument("--train_images_root", help='Root directory of train image dataset', type=str,
                        default='/home/pyuser/data/nnUnet_GliomaSeg')
    parser.add_argument("--train_data_info_file", help='file with information on train population to exclude patients',
                        type=str, default='/home/pyuser/data/nnUnet_GliomaSeg/data_info.xlsx')
    parser.add_argument("--test_images_root", help='Root directory of test image dataset', type=str,
                        default=None)
    parser.add_argument("--test_data_info_file", help='file with information on test population to exclude patients',
                        type=str, default=None)
    parser.add_argument("--nnUNet_trainer", help='nnUNet_trainer used for training', type=str,
                        default='nnUNetTrainer')
    parser.add_argument("--nnUNet_plans", help='nnUNet plans used for training', type=str, default='nnUNetPlans')
    parser.add_argument("--configuration", help='Model configuration used for training', type=str,
                        default='3d_fullres')
    parser.add_argument("--force_postprocessing", 
                        help="On the sidelines of nnunet best post-processing, apply a force post-processing with "
                        "morphological_opening_by_reconstruction on tumor and remove_homolateral_healthy_brain on brain if present", 
                        action='store_true', default=False)
    parser.add_argument("--rank_models", 
                        help="Perform pairwise model ranking according to BRATS method. WARNING this could be very long if you have more than 15/20 models.", 
                        action='store_true', default=False)
    args = parser.parse_args()
    
    print(args)

    start_time = time.time()
    
    if args.dataset.isdigit():
        full_dataset_name = get_dataset_fullname(dataset_number=args.dataset)
    else:
        full_dataset_name = get_dataset_fullname(dataset_name=args.dataset)

    if (args.test_images_root is not None and args.test_data_info_file is None) or (args.test_images_root is None and args.test_data_info_file is not None):
        raise ValueError('If one of test_images_root or test_data_info_file is given, the other must also be given.')
    
    # Load database creation arguments
    nnunet_raw_root = os.environ.get('nnUNet_raw')
    with open(os.path.join(nnunet_raw_root, full_dataset_name, 'commandline_args.txt'), 'r') as f:
        database_args = json.load(f)

    # Load existing files
    if os.path.exists(args.recap_results_file):
        if '.xlsx' in args.recap_results_file:
            recap_results_file_dic = pd.read_excel(os.path.abspath(args.recap_results_file), index_col=0, sheet_name=None)
        else:
            raise ValueError
    else:
        os.makedirs(os.path.dirname(args.recap_results_file), exist_ok=True)
        recap_results_file_dic = {}
    
    # Load and prepare data information
    if '.xlsx' in args.train_data_info_file:
        train_data_info_file = pd.read_excel(os.path.abspath(args.train_data_info_file), index_col=0, dtype={'ID': str})
    elif '.csv' in args.train_data_info_file:
        train_data_info_file = pd.read_csv(os.path.abspath(args.train_data_info_file), index_col=0, dtype={'ID': str})
    else:
        raise ValueError
    
    train_patient_link_df = pd.read_csv(os.path.join(nnunet_raw_root, full_dataset_name, 'Train_Patient_link.csv'),
                                        index_col=1, dtype={'Original ID': str, 'nnUNet ID': str})  # Index = Original ID
        # Select only samples that were used in current task
    train_data_info_file = train_data_info_file[train_data_info_file.index.isin(train_patient_link_df.index)]
        # Change indexes with nnUNet IDs
    train_data_info_file['nnUNet ID'] = train_patient_link_df['nnUNet ID']
    train_data_info_file.set_index('nnUNet ID', inplace=True)
        # Add column with mixed factors
    train_data_info_file['Device_Carbidopa'] = train_data_info_file['Device'].astype(str) + '_C' + train_data_info_file['Carbidopa'].astype(str)
    train_data_info_file['Log10_volume'] = np.log10(train_data_info_file['Seg_volume_mL'] + 1)
    
    # Retrieve fold information
    fold_comp = load_json(os.path.join(os.environ.get('nnUNet_preprocessed'), full_dataset_name, "splits_final.json"))
    fold_val_idx = {'Fold': [], 'nnUNet ID': []}
    for i, fold in enumerate(fold_comp):
        fold_val_idx['Fold'].extend([i] * len(fold['val']))
        fold_val_idx['nnUNet ID'].extend([_.split('PETseg_')[1] for _ in fold['val']])
    fold_val_idx = pd.DataFrame(fold_val_idx).set_index('nnUNet ID')
    
    results_save_dir = os.path.join(os.environ.get('nnUNet_results_analysis'), '%s_results' % full_dataset_name, '%s__%s__%s' % (args.nnUNet_trainer, args.nnUNet_plans, args.configuration))
    os.makedirs(results_save_dir, exist_ok=True)
    
    dataset_model_dir = os.path.join(os.environ.get('nnUNet_results'), full_dataset_name, 
                                                '%s__%s__%s' % (args.nnUNet_trainer, args.nnUNet_plans, args.configuration))
    crossval_results_dir = os.path.join(dataset_model_dir, 'crossval_results_folds_0_1_2_3_4')
    if not os.path.exists(crossval_results_dir):
        os.system('nnUNetv2_find_best_configuration %s -p %s -c %s -tr %s' % (full_dataset_name, args.nnUNet_plans, args.configuration, args.nnUNet_trainer))
    # Copy file containing command line arguments for nnUNet dataset generation because it is needed for inference
    shutil.copy(os.path.join(nnunet_raw_root, full_dataset_name, 'commandline_args.txt'), 
                os.path.join(dataset_model_dir,'dataset_commandline_args.txt'))
    if args.force_postprocessing and not os.path.exists(os.path.join(crossval_results_dir, 'forced_postprocessed')):
        # Create a postprocessing.pkl with functions we want to force postprocessing according to our will
        label_link = load_json(os.path.join(os.environ.get('nnUNet_raw'), full_dataset_name, 'dataset.json'))['labels']
        pp_fns = [morphological_opening_by_reconstruction]
        pp_fn_kwargs = [{'need_props': True, 'kernel_size': 1, 'use_image_spacing': False, 'labels_or_regions': label_link['tumor']}]
        if 'brain' in label_link:
            pp_fns.append(remove_homolateral_healthy_brain)
            pp_fn_kwargs.append({'labels_or_regions': [label_link['brain'], label_link['tumor']], 'need_props': True, 'label_dict': label_link})
        pp_pkl_file_path = os.path.join(crossval_results_dir, 'forced_postprocessing.pkl')
        save_pickle((pp_fns, pp_fn_kwargs), pp_pkl_file_path)
        # Call nnUNetv2_apply_postprocessing
        os.system('nnUNetv2_apply_postprocessing -i %s -o %s -pp_pkl_file %s' % (crossval_results_dir, 
                                                                                 os.path.join(crossval_results_dir, 'forced_postprocessed'),
                                                                                 pp_pkl_file_path))
        # Get nnUNet results for forced pp (needed for json)
        compute_metrics_on_folder2(os.path.join(nnunet_raw_root, full_dataset_name, 'labelsTr'),
                                   os.path.join(crossval_results_dir, 'forced_postprocessed'), 
                                   os.path.join(crossval_results_dir, 'dataset.json'),
                                   os.path.join(crossval_results_dir, 'plans.json'), 
                                   os.path.join(crossval_results_dir, 'forced_postprocessed', 'summary.json'))
        # Create associated json (done after because we need performance results to mimic original json)
        # Code below is from the end of nnunetv2.postprocessing.remove_connected_components.determine_postprocessing
        baseline_results = load_json(os.path.join(crossval_results_dir, 'summary.json'))
        final_results = load_json(os.path.join(crossval_results_dir, 'forced_postprocessed', 'summary.json'))
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
        save_json(tmp, os.path.join(crossval_results_dir, 'forced_postprocessing.json'))
    # Get results on training set
    with pd.ExcelWriter(os.path.join(results_save_dir, 'Global_results_train.xlsx')) as writer:
        # Get results by samples
        train_global_results_dic = extract_global_results(full_dataset_name, 
                                                          train_patient_link_df.set_index('nnUNet ID', append=True).reset_index(level=0), 
                                                          os.path.join(dataset_model_dir, 'crossval_results_folds_0_1_2_3_4'), 
                                                          os.path.join(nnunet_raw_root, full_dataset_name, 'imagesTr'), 
                                                          force_postprocessing=args.force_postprocessing)
        # Compute overall and fold results
        fold_results_dic = {}
        for output in train_global_results_dic:
            fold_results_dic[output] = extract_fold_results(train_global_results_dic[output].drop('Original ID', axis=1), fold_val_idx)
            # Add overall results to recap_results_file
            overall_columns = [_ for _ in fold_results_dic[output].columns if 'overall' in _]
            col_name = '%s__%s__%s__%s' % (full_dataset_name, args.nnUNet_trainer, args.nnUNet_plans, args.configuration)
            train_recap_results_df = pd.DataFrame(pd.concat([fold_results_dic[output][col].add_prefix('%s ' % col) for col in overall_columns]), 
                                                    columns=[col_name])
            if not 'train_%s' % output in recap_results_file_dic:
                recap_results_file_dic['train_%s' % output] = train_recap_results_df.transpose()
            else:
                if col_name in recap_results_file_dic['train_%s' % output].index:
                    if set(train_recap_results_df[col_name].index).issubset(recap_results_file_dic['train_%s' % output].columns):
                        recap_results_file_dic['train_%s' % output].loc[col_name] = train_recap_results_df[col_name]
                    else:
                        recap_results_file_dic['train_%s' % output].drop(col_name, axis=0, inplace=True)
                        recap_results_file_dic['train_%s' % output] = pd.concat([recap_results_file_dic['train_%s' % output], 
                                                                                 train_recap_results_df[col_name].to_frame().transpose()], axis=0)
                else:
                    recap_results_file_dic['train_%s' % output] = pd.concat([recap_results_file_dic['train_%s' % output], 
                                                                            train_recap_results_df[col_name].to_frame().transpose()], axis=0)
            # Write results by samples and fold
            train_global_results_dic[output].to_excel(writer, sheet_name='%s_Sample' % output)
            fold_results_dic[output].transpose().to_excel(writer, sheet_name='%s_Fold' % output)
        # Post-hoc tests
        model_post_hoc_tests(args.configuration, train_global_results_dic, train_data_info_file, os.path.join(results_save_dir, 'Train_post_hoc_tests'))

    if args.test_images_root is not None:

        # Load and prepare data information
        if '.xlsx' in args.test_data_info_file:
            test_data_info_file = pd.read_excel(os.path.abspath(args.test_data_info_file), index_col=0, dtype={'ID': str})
        elif '.csv' in args.test_data_info_file:
            test_data_info_file = pd.read_csv(os.path.abspath(args.test_data_info_file), index_col=0, dtype={'ID': str})
        else:
            raise ValueError
        
        test_patient_link_df = pd.read_csv(os.path.join(nnunet_raw_root, full_dataset_name, 'Test_Patient_link.csv'),
                                           index_col=1, dtype={'Original ID': str, 'nnUNet ID': str})  # Index = Original ID
            # Select only samples that were used in current task
        test_data_info_file = test_data_info_file[test_data_info_file.index.isin(test_patient_link_df.index)]
            # Change indexes with nnUNet IDs
        test_data_info_file['nnUNet ID'] = test_patient_link_df['nnUNet ID']
        test_data_info_file.set_index('nnUNet ID', inplace=True)
            # Add column with mixed factors
        test_data_info_file['Device_Carbidopa'] = test_data_info_file['Device'].astype(str) + '_C' + test_data_info_file['Carbidopa'].astype(str)
        test_data_info_file['Log10_volume'] = np.log10(test_data_info_file['Seg_volume_mL'] + 1)

        if os.path.exists(os.path.join(nnunet_raw_root, full_dataset_name, 'imagesTs')) and \
            os.listdir(os.path.join(nnunet_raw_root, full_dataset_name, 'imagesTs')):
            test_results_dir = os.path.join(dataset_model_dir, 'test_results_folds_0_1_2_3_4')
        if not os.path.exists(test_results_dir):
            # Run prediction
            os.system('nnUNetv2_predict -d %s -i %s -o %s -f 0 1 2 3 4 -p %s -c %s -tr %s -npp 10 -nps 10 -device cuda' % (full_dataset_name, 
                                                                                                                            os.path.join(nnunet_raw_root, full_dataset_name, 'imagesTs'),
                                                                                                                            test_results_dir,
                                                                                                                            args.nnUNet_plans, args.configuration, args.nnUNet_trainer))
            # Run post processing if needed
            pp_json = load_json(os.path.join(crossval_results_dir, 'postprocessing.json'))
            if pp_json["postprocessing_fns"]:
                os.system('nnUNetv2_apply_postprocessing -i %s -o %s -pp_pkl_file %s -np 8 -plans_json %s' % (test_results_dir,
                                                                                                                os.path.join(test_results_dir, 'postprocessed'),
                                                                                                                os.path.join(crossval_results_dir, 'postprocessing.pkl'),
                                                                                                                os.path.join(crossval_results_dir, 'plans.json')))
                # TODO : is this really needed ? 
                # Copy needed files to match structure after find_best_configuration
                shutil.copy(os.path.join(test_results_dir, 'dataset.json'), os.path.join(test_results_dir, 'postprocessed', 'dataset.json'))
                shutil.copy(os.path.join(test_results_dir, 'plans.json'), os.path.join(test_results_dir, 'postprocessed', 'plans.json'))
            shutil.copy(os.path.join(crossval_results_dir, 'postprocessing.json'), os.path.join(test_results_dir, 'postprocessing.json'))
            shutil.copy(os.path.join(crossval_results_dir, 'postprocessing.pkl'), os.path.join(test_results_dir, 'postprocessing.pkl'))
        # Run forced post processing if needed:
        if args.force_postprocessing:
            os.system('nnUNetv2_apply_postprocessing -i %s -o %s -pp_pkl_file %s -np 8 -plans_json %s' % (test_results_dir, 
                                                                                                            os.path.join(test_results_dir, 'forced_postprocessed'), 
                                                                                                            os.path.join(crossval_results_dir, 'forced_postprocessing.pkl'), 
                                                                                                            os.path.join(crossval_results_dir, 'plans.json')))
            # TODO : is this really needed ? 
            # Copy needed files to match structure after find_best_configuration
            shutil.copy(os.path.join(test_results_dir, 'dataset.json'), os.path.join(test_results_dir, 'forced_postprocessed', 'dataset.json'))
            shutil.copy(os.path.join(test_results_dir, 'plans.json'), os.path.join(test_results_dir, 'forced_postprocessed', 'plans.json'))
            shutil.copy(os.path.join(crossval_results_dir, 'forced_postprocessing.json'), os.path.join(test_results_dir, 'forced_postprocessing.json'))
            shutil.copy(os.path.join(crossval_results_dir, 'forced_postprocessing.pkl'), os.path.join(test_results_dir, 'forced_postprocessing.pkl'))
        # Get results with nnunet evaluation in summary.json
        for output in train_global_results_dic:
            if output == 'raw':
                summary_dir = test_results_dir
            elif output == 'forced_pp':
                summary_dir = os.path.join(test_results_dir, 'forced_postprocessed')
            else:
                summary_dir = os.path.join(test_results_dir, 'postprocessed')
            compute_metrics_on_folder2(os.path.join(nnunet_raw_root, full_dataset_name, 'labelsTs'), 
                                        summary_dir, os.path.join(summary_dir, 'dataset.json'), 
                                        os.path.join(summary_dir, 'plans.json'), os.path.join(summary_dir, 'summary.json'))
        
        with pd.ExcelWriter(os.path.join(results_save_dir, 'Global_results_test.xlsx')) as writer:
            # Get results by samples
            test_global_results_dic = extract_global_results(full_dataset_name, 
                                                             test_patient_link_df.set_index('nnUNet ID', append=True).reset_index(level=0),
                                                             os.path.join(dataset_model_dir, 'test_results_folds_0_1_2_3_4'),
                                                             os.path.join(nnunet_raw_root, full_dataset_name, 'imagesTs'), 
                                                             force_postprocessing=args.force_postprocessing)
            # Compute overall and fold results
            for output in test_global_results_dic:
                # Get overall results and add to recap_results_file (bypass fold results by doing as if there is only one fold)
                test_overall_mean_std = extract_fold_results(test_global_results_dic[output].drop('Original ID', axis=1),
                                                                pd.DataFrame({'Fold': np.zeros_like(test_patient_link_df['nnUNet ID']), 
                                                                            'nnUNet ID': test_patient_link_df['nnUNet ID']}).set_index('nnUNet ID'))
                overall_columns = [_ for _ in test_overall_mean_std.columns if 'overall' in _]
                col_name = '%s__%s__%s__%s' % (full_dataset_name, args.nnUNet_trainer, args.nnUNet_plans, args.configuration)
                test_recap_results_df = pd.DataFrame(pd.concat([test_overall_mean_std[col].add_prefix('%s ' % col) for col in overall_columns]), 
                                                        columns=[col_name])
                if not 'test_%s' % output in recap_results_file_dic:
                    recap_results_file_dic['test_%s' % output] = test_recap_results_df.transpose()
                else:
                    if col_name in recap_results_file_dic['test_%s' % output].index:
                        if set(test_recap_results_df[col_name].index).issubset(recap_results_file_dic['test_%s' % output].columns):
                            recap_results_file_dic['test_%s' % output].loc[col_name] = test_recap_results_df[col_name]
                        else:
                            recap_results_file_dic['test_%s' % output].drop(col_name, axis=0, inplace=True)
                            recap_results_file_dic['test_%s' % output] = pd.concat([recap_results_file_dic['test_%s' % output], 
                                                                                    test_recap_results_df[col_name].to_frame().transpose()], axis=0)
                    else:
                        recap_results_file_dic['test_%s' % output] = pd.concat([recap_results_file_dic['test_%s' % output], 
                                                                                test_recap_results_df[col_name].to_frame().transpose()], axis=0)
                # Write results by samples and fold
                test_overall_mean_std.transpose().to_excel(writer, sheet_name='%s_Fold' % output)
                test_global_results_dic[output].to_excel(writer, sheet_name='%s_Sample' % output)
                # Post-hoc tests
                model_post_hoc_tests(args.configuration, test_global_results_dic, test_data_info_file, os.path.join(results_save_dir, 'Test_post_hoc_tests'))

    if not len(recap_results_file_dic[list(recap_results_file_dic.keys())[0]].index) == 1 and args.rank_models:
        # Rank models and compare them
        # WARNING: this could be very long if the number of models to compare is high as we perform pairwise comparison. Set to False if needed.
        rank_and_compare_models(recap_results_file_dic, os.path.abspath(args.recap_results_file))
    # Write recap results after since we add some columns
    with pd.ExcelWriter(os.path.abspath(args.recap_results_file)) as writer:
        for key in recap_results_file_dic:
            recap_results_file_dic[key].dropna(axis=1, how='all').to_excel(writer, sheet_name=key)
    # Copy global results in the same directory as recap file
    target_save_dir = os.path.join(os.path.dirname(os.path.abspath(args.recap_results_file)), 
                                   '%s_results' % full_dataset_name, '%s__%s__%s' % (args.nnUNet_trainer, args.nnUNet_plans, args.configuration))
    # os.makedirs(target_save_dir, exist_ok=True)
    if os.path.exists(target_save_dir):
        shutil.rmtree(target_save_dir)
    shutil.copytree(results_save_dir, target_save_dir)    
    print(str(datetime.timedelta(seconds=int(time.time() - start_time))))