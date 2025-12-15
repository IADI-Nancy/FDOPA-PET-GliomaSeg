ARG PYTORCH_VER=2.3.0-cuda12.1-cudnn8-devel
FROM pytorch/pytorch:${PYTORCH_VER}

# Set nnUNetv2 variables environnement
ENV nnUNet_raw="/root/nnUNetv2_data/nnUNet_raw"
ENV nnUNet_preprocessed="/root/nnUNetv2_data/nnUNet_preprocessed"
ENV nnUNet_results="/root/nnUNetv2_data/nnUNet_results"
ENV nnUNet_results_analysis="/root/nnUNetv2_data/nnUNet_results_analysis"
# Set mri_synthstrip in path
ENV PATH="$PATH:/root/src/libraries/mri_synthstrip"

# Copy required files from local to image
COPY ./docker/requirements.txt requirements.txt
COPY ./src /root/src
COPY ./docker/scripts scripts

RUN export DEBIAN_FRONTEND=noninteractive \
	# install needed utilities (wget, gzip, git, etc.)
	&& apt update \
	&& apt install -y --no-install-recommends \
		wget \
		unzip \
		git \
		screen \
		openssh-client \
		curl \
	# init conda (required for some reasons)
	&& /opt/conda/bin/conda init \
	# install dependencies (xet, git-lfs)
	&& bash scripts/xet_install.sh \
	&& git xet install \
	&& bash scripts/script.deb.sh \
	&& apt install git-lfs \
	&& rm -rf /var/lib/apt/lists/* \
	# # install other python dependencies and nnUNetv2
	&& pip install --force-reinstall --no-cache-dir networkx==3.1 typing_extensions==4.11.0 opencv-python==4.11.0.86 \
	&& pip install -e /root/src/libraries/nnUNetv2/nnUNet \
	&& pip install --no-cache-dir -r requirements.txt \
	# create nnunet directories
	&& mkdir -p $nnUNet_raw \
	&& mkdir -p $nnUNet_preprocessed \
	&& mkdir -p $nnUNet_results \
	&& mkdir -p $nnUNet_results_analysis \
	# set permission to allow execution of mri_synthstrip
	&& chmod 744 /root/src/libraries/mri_synthstrip/mri_synthstrip \
	# download model weights
	&& cd $nnUNet_results \
	&& git clone https://huggingface.co/IADI-Nancy/FDOPA-PET-GliomaSeg \
	&& mv $nnUNet_results/FDOPA-PET-GliomaSeg/Dataset509_GliomaSeg_prepro_resample_only $nnUNet_results/Dataset509_GliomaSeg_prepro_resample_only \
	&& rm -r $nnUNet_results/FDOPA-PET-GliomaSeg \
	# create empty dir for dataset in nnUNet_raw and nnUNet_preprocessed (could be needed in code)
	&& mkdir -p $nnUNet_raw/Dataset509_GliomaSeg_prepro_resample_only \
	&& mkdir -p $nnUNet_preprocessed/Dataset509_GliomaSeg_prepro_resample_only \
	# populate it with plans.json, dataset.json, etc.
	&& cp $nnUNet_results/Dataset509_GliomaSeg_prepro_resample_only/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json $nnUNet_results/Dataset509_GliomaSeg_prepro_resample_only/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset_commandline_args.txt $nnUNet_raw/Dataset509_GliomaSeg_prepro_resample_only/ \
	&& mv $nnUNet_raw/Dataset509_GliomaSeg_prepro_resample_only/dataset_commandline_args.txt $nnUNet_raw/Dataset509_GliomaSeg_prepro_resample_only/commandline_args.txt \
	&& cp $nnUNet_results/Dataset509_GliomaSeg_prepro_resample_only/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json $nnUNet_results/Dataset509_GliomaSeg_prepro_resample_only/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json $nnUNet_preprocessed/Dataset509_GliomaSeg_prepro_resample_only/ \
	&& mv $nnUNet_preprocessed/Dataset509_GliomaSeg_prepro_resample_only/plans.json $nnUNet_preprocessed/Dataset509_GliomaSeg_prepro_resample_only/nnUNetPlans.json
	
WORKDIR /root
ENTRYPOINT ["/bin/bash"]