# Database preparation
## Overview
`data_generation` performs the initial preprocessing required for downstream analysis by converting and organizing imaging data for subsequent analysis. Its main function is to convert PET scans from DICOM to NIfTI format with SUV normalization and generate tumor segmentation using a TBR 1.6 threshold, leveraging a tumor box and healthy brain VOI in NIfTI format. It ensures data integrity by handling missing slices and correcting spatial metadata. Optionally, it can create a skull-stripped version of the static PET image using [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) and refine the tumor VOI with a 1-mm morphological opening by reconstruction (as described in our article).

## Command-Line Arguments
The script accepts the following arguments:
`--input_dir DATASET_INPUT_DIR`: Root directory containing patient folders with DICOM images and ROI masks (required)
`--output_dir DATASET_OUTPUT_DIR`: Directory where processed images, ROIs, and results will be saved (required)
`--data_info_file DATA_INFO_FILE`: Path to a CSV or Excel file with population information (optional).
If given, this file must contain one line per sample and, for example, the following columns:
- `ID`: ID of the sample corresponding as its directory name (mandatory as first column)
- `Device`: name of the device on which the acquisition was performed (used for stratification)
- `Recurrence`: Whether the acquisition was performed at initial diagnosis (0) or not (1) (used for stratification)
- `Problem_segmentation`: Binary variable encoding samples with any issue that we would like to exclude
If the file doesn't exist, it will be created with columns containing information on the tumor segmentation (volume, threshold, measurable disease classification according to PET RANO criteria)
`--postprocessROI`: If set, applies additional post-processing to tumor VOIs generated during segmentation (optional flag)
`--skullstripping`: If set, performs skull-stripping on PET images using SynthStrip (optional flag)

Command-line example:
```bash
python -m src.functions.data_generation --input_dir DATASET_INPUT_DIR --output_dir DATASET_OUTPUT_DIR --data_info_file DATA_INFO_FILE --postprocessROI --skullstripping
```

## Expected Folder Structure
### Input Directory
```bash
input_dir/
  └── Patient_001/
        ├── Static_PET/                           # DICOM folder containing files for PET scan. Name must be Static_PET
        └── ROI/                                  # ROI masks (.nii files)
		     ├── Brain.nii.gz                     # Healthy brain VOI (binary NIfTI file)
             └── TumorBox.nii.gz                  # TumorBox VOI (binary NIfTI file)
  └── Patient_002/
        ├── Static_PET/
        └── ROI/
		     ├── Brain.nii.gz
             └── TumorBox.nii.gz
  ...
  └── data_info_file.csv or .xlsx                 # Population info
```

### Output Directory
```bash
output_dir/
  └── Patient_001/
        ├── Static_PET.nii.gz                     # NIfTI PET image converted to SUV
		├── Static_PET_skullstripped.nii.gz       # Skull-stripped PET image (if --skullstripping is used)
		├── Static_PET_skullstripped_mask.nii.gz  # Skull-stripped mask (binary NIfTI file) (if --skullstripping is used)
        └── ROI/                                  # ROI masks (.nii files)
              ├── Brain.nii.gz                    # Healthy brain mask (binary NIfTI file)
              ├── TumorBox.nii.gz                 # TumorBox mask (binary NIfTI file)
			  └── Tumor_TBR1,6_seg.nii.gz         # Segmented Tumor mask generated using TBR 1.6 threshold (binary NIfTI file)
  └── Patient_002/
        ├── Static_PET.nii.gz
        └── ROI/
              └── ...
  ...
  └── data_info_file.csv or .xlsx                 # Updated population info
```

# Generate nnU-Net database
## Overview
`generate_nnunet_database` organizes and prepares medical imaging data for training and evaluating nnU-Net models. It collects PET images and segmentation masks, apply the preprocessing as detail in our article (with some other options), structures them according to nnU-Net’s requirements, and generates all necessary metadata and configuration files. The script run the `nnUNetv2_plan_and_preprocess` command internally. 

## Command-Line Arguments
`--train_images_root TRAIN_IMAGES_ROOT`: Root directory containing training images and masks (required)
`--train_data_info_file TRAIN_DATA_INFO_FILE`: CSV or Excel file with information on the training population (required).
This file must contain one line per sample and, for example, the following columns:
- `ID`: ID of the sample corresponding as its directory name (mandatory as first column)
- `Device`: name of the device on which the acquisition was performed (used for stratification)
- `Recurrence`: Whether the acquisition was performed at initial diagnosis (0) or not (1) (used for stratification)
- `Problem_segmentation`: Binary variable encoding samples with any issue that we would like to exclude
`--test_images_root TEST_IMAGES_ROOT`: Root directory for test images and masks (optional)
`--test_data_info_file TEST_DATA_INFO_FILE`: CSV or Excel file with information on the test population (optional; required if --test_images_root is set). 
See `--train_data_info_file TRAIN_DATA_INFO_FILE`
`--dataset DATASET_NAME`: Name for the nnU-Net dataset (required)
`--skull_stripping`: Use skull-stripped images (optional flag)
`--brain_centering`: Center brain in the image as preprocessing step (optional flag; requires skull-stripped masks)
`--ref_spacing`: Reference isotropic spacing for resampling (default: 1.0 as used in the article)
`--ref_size`: Reference image size for resampling (default: 256 256 164 as used in the article)
`--labels`: Labels to use for training (only_tumor or tumor_brain, default: tumor_brain)
`--threads`: Number of parallel threads (default: -1, i.e., all available)
`--preprocessor`: Name of the preprocessor (default: DefaultPreprocessor)
`--architecture`: Network architecture/planner variant (default: default)

Command-line example:
```bash
python -m src.functions.generate_nnunet_database --train_images_root TRAIN_IMAGES_ROOT --train_data_info_file TRAIN_DATA_INFO_FILE --test_images_root TEST_IMAGES_ROOT --test_data_info_file TEST_DATA_INFO_FILE --dataset DATASET_NAME --ref_spacing 1 --ref_size 256 256 164 --labels tumor_brain --threads 20
```

## Expected Folder Structure
### Input Directory
`TRAIN_IMAGES_ROOT` and `TEST_IMAGES_ROOT`, if provided, must be organized as the output of `data_generation`.

### Output Directory
```bash
nnUNet_raw/
  └── DatasetXXX_NAME/
        └── imagesTr/
              ├── PETseg_000_0000.nii.gz
              ├── PETseg_001_0000.nii.gz
              └── ...
        └── labelsTr/
              ├── PETseg_000.nii.gz
              ├── PETseg_001.nii.gz
              └── ...
        └── imagesTs/                # (if TEST_IMAGES_ROOT provided)
              ├── PETseg_100_0000.nii.gz
              └── ...
        └── labelsTs/                # (if TEST_IMAGES_ROOT provided)
              ├── PETseg_100.nii.gz
              └── ...
        ├── dataset.json             # nnU-Net dataset metadata
        ├── commandline_args.txt     # Arguments used for the script
        ├── Train_Patient_link.csv   # Mapping between original and nnUNet IDs
        ├── Test_Patient_link.csv    # (if TEST_IMAGES_ROOT provided)
        └── splits_final.json        # Stratification info (either nnU-Net one or by Device + Recurrence if provided in train_data_info_file)
```
This script also automatically runs `nnUNetv2_plan_and_preprocess`, creating the appropriate files, such as preprocessed data and plan files, in the `nnUNet_preprocessed` folder.

# Train all folds
## Overview
`train_all_folds` automates the training of nnU-Net models across all cross-validation folds. It sequentially launches the nnU-Net training process for each fold, allowing to chain the training on the different folds.

## Command-Line Arguments
The script accepts the following arguments:

`--nnUNet_trainer TRAINER`: Name of the nnU-Net trainer to use (default: nnUNetTrainer)
`--nnUNet_plans PLANS`: Name of the nnU-Net plans to use (default: nnUNetPlans)
`--configuration CONFIGURATION`: Training configuration (default: 3d_fullres)
`--dataset DATASET_NUMBER`: Dataset name or number (required)
`--start_fold START_FOLD`: Index of the starting fold (default: 0)

Command-line example:
```bash
python ~/src/functions/train_all_folds.py --dataset DATASET_NUMBER --configuration CONFIGURATION --nnUNet_plans PLANS --nnUNet_trainer TRAINER
```

# Evaluate model
## Overview
`generate_nnunet_results` automates the post-processing and evaluation of nnU-Net segmentation results. It collects predictions from all folds, applies post-processing if requested, computes performance metrics, and summarizes results for both training and test sets (if given). The script can also perform comparison with other models (model ranking) and exports all relevant metrics and summaries to Excel files for further analysis.

## Command-Line Arguments
`--recap_results_file RECAP_RESULTS_FILE`: Path to the Excel file where summary results will be saved (required)
`--dataset DATASET_NAME_OR_NUMBER`: Name or number of the nnU-Net dataset (required)
`--nnUNet_trainer TRAINER`: Trainer used for nnU-Net training (default: nnUNetTrainer)
`--nnUNet_plans PLANS`: Plans used for nnU-Net training (default: nnUNetPlans)
`--configuration CONFIGURATION`: Model configuration used for nnU-Net training (default: 3d_fullres)
`--force_postprocessing`: If set, also applies a predefined post-processing as in our article (morphological opening on tumor, removal of homolateral healthy brain) (optional flag)
`--rank_models`: If set, performs pairwise model ranking and comparison (optional flag; may be slow for many models)

Command-line example:
```bash
python -m src.functions.generate_nnunet_results --recap_results_file RECAP_RESULTS_FILE --dataset DATASET_NAME_OR_NUMBER --nnUNet_trainer TRAINER --nnUNet_plans PLANS --configuration CONFIGURATION --force_postprocessing
```

# Inference
## Overview
`predict_evaluation_new_data` automates the inference and evaluation workflow for new medical imaging data using a trained nnU-Net model. It processes input images (DICOM or NIfTI), applies required preprocessing, generates segmentation predictions, and computes quantitative metrics or segmentation evaluation metrics if ground truth is available (binary NifTI masks). The script exports all results and metrics in a structured output directory for further analysis.

## Command-Line Arguments
`--input_dir PREDICT_INPUT_DIR`: Root directory containing new patient data (required)
`--output_dir PREDICT_OUTPUT_DIR`: Directory where all outputs (preprocessed data, predictions, results) will be saved (required)
`--dataset DATASET_NAME_OR_NUMBER`: Name or number of the nnU-Net dataset to use for inference (required)
`--nnUNet_trainer TRAINER`: Trainer name used for nnU-Net training (required)
`--nnUNet_plans PLANS`: Plan name used for nnU-Net training (required)
`--configuration CONFIGURATION`: Model configuration used for nnU-Net training (e.g., 3d_fullres). (required)
`--device DEVICE`: Device on which inference is done (see nnUNetv2_predict arguments, default `cuda` for inference on GPU)
`--force_postprocessing`: If set, also applies additional post-processing as in our article (morphological opening on tumor, removal of homolateral healthy brain) (optional flag)

Command-line example:
```bash
python -m src.functions.predict_evaluation_new_data --input_dir PREDICT_INPUT_DIR --output_dir PREDICT_OUTPUT_DIR --dataset DATASET_NAME_OR_NUMBER --nnUNet_trainer TRAINER --nnUNet_plans PLANS --configuration CONFIGURATION --device DEVICE --force_postprocessing
```

## Expected Folder Structure
### Input Directory
```bash
input_dir/
  └── Patient_001/
        ├── DICOM/                  # DICOM folder containing files for PET scan. No rule regarding the name
        └── ROI/                    # Optional ground truth masks (binary .nii files)
             ├── Brain.nii.gz
             └── TumorBox.nii.gz
  └── Patient_002/
        ├── DICOM/
        └── ROI/
             ├── Brain.nii.gz
             └── TumorBox.nii.gz
  ...
```

### Output Directory
```bash
output_dir/
  └── Generated_data/
        └── Patient_001/
              ├── Static_PET.nii.gz
              ├── Static_PET_skullstripped.nii.gz         # If skull-stripping is used
              ├── Static_PET_skullstripped_mask.nii.gz    # If skull-stripping is used
              └── ROI/                                    # Optional ground truth masks (binary .nii files)
                    ├── Brain.nii.gz
                    └── Tumor_TBR1,6_seg.nii.gz
  └── nnUNet_data/
        └── imagesTs/
              ├── PETseg_000_0000.nii.gz
              ├── PETseg_001_0000.nii.gz
              └── ...
        └── labelsTs/                                     # Optional ground truth masks (binary .nii files)
              ├── PETseg_000.nii.gz
              ├── PETseg_001.nii.gz
              └── ...
        └── Patient_link.csv                              # Mapping between original and nnUNet IDs
  └── nnUNet_results/
        ├── PETseg_000.nii.gz                             # Predicted segmentation masks
        ├── PETseg_001.nii.gz
        ├── ...  
        ├── dataset.json                                  # nnU-Net dataset metadata 
        ├── plans.json                                    # nnU-Net plans metadata
        ├── predict_from_raw_data_args.json               # Arguments used for the script
        ├── summary.json                                  # Evaluation file generated using nnU-Net
        └── postprocessed/                                # Post-processed predictions (if applicable)
              ├── PETseg_000.nii.gz                       
              ├── PETseg_001.nii.gz
              ├── ...
              ├── dataset.json                            # nnU-Net dataset metadata 
              ├── plans.json                              # nnU-Net plans metadata
              └── summary.json                            # Evaluation file generated using nnU-Net
        ├── postprocessing.json                           # Information on post-processing used
        ├── postprocessing.pkl
        └── forced_postprocessed/                         # Forced post-processing results (if applicable)
              ├── PETseg_000.nii.gz                       
              ├── PETseg_001.nii.gz
              ├── ...
              ├── dataset.json                            # nnU-Net dataset metadata 
              ├── plans.json                              # nnU-Net plans metadata
              └── summary.json                            # Evaluation file generated using nnU-Net
        ├── forced_postprocessing.json                    # Information on forced post-processing used
        ├── forced_postprocessing.pkl
        └── Global_results.xlsx                           # Quantitative metrics and optionally segmentation metrics summary
```