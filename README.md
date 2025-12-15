# FDOPA-PET-GliomaSeg
FDOPA-PET-GliomaSeg is a deep learning model for automated glioma segmentation on [18F]FDOPA PET. Developed from a modified nnU-Netv2 framework and trained on multicenter clinical data, FDOPA-PET-GliomaSeg provides robust and generalizable tumor delineation across scanners and institutions. This repository contains the source code for the article "Automatic Extraction of PET RANO Criteria with an Externally Validated Deep Learning Model: Application to [18F]F-FDOPA PET Imaging" (submitted to Neuro-Oncology).

## Model
Model checkpoints have been made publicly available for [download on HuggingFace](https://huggingface.co/IADI-Nancy/FDOPA-PET-GliomaSeg). Checkpoints are provided in PyTorch format and compatible with nnU-Netv2.

## Installation

We provide a Dockerfile to build a Docker image with the same environment as the one used to develop the model. We recommend running the model in this environment.
The first step is to clone the repository at the desired place
```bash
git clone https://github.com/IADI-Nancy/FDOPA-PET-GliomaSeg.git
cd FDOPA-PET-GliomaSeg
```

### Requirements
Our Docker is built on a [Pytorch image with version 2.3.0, CUDA 12.1 and CUDNN 8](https://hub.docker.com/layers/pytorch/pytorch/2.3.0-cuda12.1-cudnn8-devel/images/sha256-1bb28822b361bdb2d8cde5a58f08337490bf6e73fc96b0aa1035268c295f3d00). 

In order to run this Docker, if you use a GPU, your GPU is expected to have a Compute Capability of 5.0 or higher (to find the Compute Capability of your GPU, have a look to [nvidia](https://developer.nvidia.com/cuda-gpus) or [Wikipedia](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) )
This image also works on CPU but with reduced performance.
### Setup
1. Build Docker image from Docker file

Mandatory arguments:
- `IMAGE_NAME`: name of the image (ex. `fdopa-pet-gliomaseg`)
- `IMAGE_VERSION`: version of the image (ex. `pytorch2.3.0-cuda12.1-cudnn8`)

Run in your terminal: 
```bash
docker build -t IMAGE_NAME:IMAGE_VERSION -f ./docker/Dockerfile --rm=true .
```
2. Run a container

Mandatory arguments:
- `CONTAINER_NAME`: name of the container (ex. `inference-fdopa-pet-gliomaseg`)
- `IMAGE_NAME`: previously defined name of the image (ex. `fdopa-pet-gliomaseg`)
- `IMAGE_VERSION`: previously defined version of the image (ex. `pytorch2.3.0-cuda12.1-cudnn8`)
- `PATH_TO_DATA`: path to the directory where your data are stored

Optional_arguments
- `GPU_NUMBER`: GPU devices to add to the container, see [documentation](https://docs.docker.com/reference/cli/docker/container/run/#gpus) (ex. `all` or `device=0`)
- `USER_ID`: user ID on the computer (ex. on Linux obtained by running `id -u` in the terminal)
- `GROUP_ID`: user ID on the computer (ex. on Linux obtained by running `id -g` in the terminal)

2.1. For a single command and remove afterward

Supplementary mandatory arguments:
- `COMMAND`: command to run in the container, see [How to use scripts](how_to_use_scripts.md)

Run in your terminal:
```bash
docker run --rm -v PATH_TO_DATA:/root/data --gpus GPU_NUMBER --shm-size=8g --name CONTAINER_NAME -u USER_ID:GROUP_ID IMAGE_NAME:IMAGE_VERSION -c "COMMAND"
```

2.2 In detached interactive mode

Run in your terminal:
```bash
docker run -itd -v PATH_TO_DATA:/root/data --gpus GPU_NUMBER --shm-size=8g --name CONTAINER_NAME IMAGE_NAME:IMAGE_VERSION
docker exec -it -u USER_ID:GROUP_ID CONTAINER_NAME /bin/bash
```

Starting from this point, the user can run any script interactively (python scripts from [How to use scripts](how_to_use_scripts.md) for example).

## Inference
### Run

To segment new [18F]F-FDOPA PET images using our trained model, after building the Docker image, you can:

1. Run with Docker (recommended)
```bash
docker run --rm -v PATH_TO_DATA:/root/data --gpus GPU_NUMBER --shm-size=8g --name CONTAINER_NAME -e UID=USER_ID -e GID=GROUP_ID IMAGE_NAME:IMAGE_VERSION -c "python -m src.functions.predict_evaluation_new_data --input_dir PREDICT_INPUT_DIR --output_dir PREDICT_OUTPUT_DIR --dataset GliomaSeg_prepro_resample_only --nnUNet_trainer nnUNetTrainer --nnUNet_plans nnUNetPlans --configuration 3d_fullres --force_postprocessing"
```

2. Run directly with Python

This command should be run inside the Docker container (interactive mode) or in a local environment where all dependencies (Python, CUDA, PyTorch, nnU-Net, etc.) are properly installed and configured.
```bash
python -m src.functions.predict_evaluation_new_data --input_dir PREDICT_INPUT_DIR --output_dir PREDICT_OUTPUT_DIR --dataset GliomaSeg_prepro_resample_only --nnUNet_trainer nnUNetTrainer --nnUNet_plans nnUNetPlans --configuration 3d_fullres --force_postprocessing
```

with `PREDICT_INPUT_DIR` and `PREDICT_OUTPUT_DIR` being the root directory containing new patient data and where the outputs will be saved, respectively.
To get more details on the other scripts, advanced options, database preparation, training or evaluation, please consult [How to use scripts](how_to_use_scripts.md).

### Inference times

Mean inference time using the inference script, including preprocessing, inference, and postprocessing:

| Device          | Device type    | Time / patient    |
|-----------------|----------------|-------------------|
| NVIDIA A100     | GPU            | ~40s              |
| Quadro RTX 8000 | GPU            | ~1min05s          |
| Quadro RTX 5000 | GPU            | ~1min20s          |


## License

This project uses a multi-license structure:

- **Code**: Licensed under the Apache License 2.0 - see [code LICENSE](LICENSE)
- **Models**: Licensed under CC BY-NC 4.0 - see [model LICENSE](https://huggingface.co/IADI-Nancy/FDOPA-PET-GliomaSeg/blob/main/LICENSE.txt)
- **[SynthStrip script](src/libraries/mri_synthstrip/mri_synthstrip)**: Licensed under the FreeSurfer Software License Agreement - see [FreeSurfer LICENSE](src/libraries/mri_synthstrip/LICENSE_FREESURFER.txt)

## Citation

Coming soon
