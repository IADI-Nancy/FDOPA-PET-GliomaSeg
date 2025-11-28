# FDOPA-PET-GliomaSeg
FDOPA-PET-GliomaSeg is a deep learning model for automated glioma segmentation on [¹⁸F]FDOPA PET. Developed from a modified nnU-Netv2 framework and trained on multicenter clinical data, FDOPA-PET-GliomaSeg provides robust and generalizable tumor delineation across scanners and institutions. This repository contains the source code for the article "External Validation of an Automated Model for the Extraction of Brain Amino Acid PET Parameters: Application to [18F]F-FDOPA PET" (submitted to Journal of Nuclear Medicine).

## Model
Model checkpoints have been made publicly available for [download on HuggingFace](https://huggingface.co/IADI-Nancy/FDOPA-PET-GliomaSeg).

## License

This project uses a multi-license structure:

- **Code**: Licensed under the Apache License 2.0 - see [code LICENSE](LICENSE)
- **Models**: Licensed under CC BY-NC 4.0 - see [model LICENSE](https://huggingface.co/IADI-Nancy/FDOPA-PET-GliomaSeg/blob/main/LICENSE.txt)
- **SynthStrip script**: Licensed under the FreeSurfer Software License Agreement - see [FreeSurfer LICENSE](src/libraries/mri_synthstrip/LICENSE_FREESURFER.txt)

## Coming soon
- Detailed ReadMe
- Dockerfile
