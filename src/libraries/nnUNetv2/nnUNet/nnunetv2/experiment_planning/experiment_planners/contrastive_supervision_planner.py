from typing import List, Union, Tuple
from nnunetv2.network_architecture.contrastive_deep_supervision import PlainConvUNetContrastiveSupervision, ResidualEncoderUNetContrastiveSupervision
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import nnUNetPlannerResEncM, nnUNetPlannerResEncL, nnUNetPlannerResEncXL


class ExperimentPlannerContrastiveSupervision(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetPlansContSup',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name, 
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = PlainConvUNetContrastiveSupervision
    
    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        if configuration_name == '2d' or configuration_name == '3d_fullres':
            # we do not deviate from ExperimentPlanner so we can reuse its data
            return 'nnUNetPlans' + '_' + configuration_name
        else:
            return self.plans_identifier + '_' + configuration_name


class nnUNetPlannerResEncMContrastiveSupervision(nnUNetPlannerResEncM):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetMPlansContSup',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNetContrastiveSupervision
    
    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        if configuration_name == '2d' or configuration_name == '3d_fullres':
            # we do not deviate from ExperimentPlanner so we can reuse its data
            return 'nnUNetPlans' + '_' + configuration_name
        else:
            return self.plans_identifier + '_' + configuration_name


class nnUNetPlannerResEncLContrastiveSupervision(nnUNetPlannerResEncL):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetLPlansContSup',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNetContrastiveSupervision
    
    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        if configuration_name == '2d' or configuration_name == '3d_fullres':
            # we do not deviate from ExperimentPlanner so we can reuse its data
            return 'nnUNetPlans' + '_' + configuration_name
        else:
            return self.plans_identifier + '_' + configuration_name


class nnUNetPlannerResEncXLContrastiveSupervision(nnUNetPlannerResEncXL):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 40,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetXLPlansContSup',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNetContrastiveSupervision
    
    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        if configuration_name == '2d' or configuration_name == '3d_fullres':
            # we do not deviate from ExperimentPlanner so we can reuse its data
            return 'nnUNetPlans' + '_' + configuration_name
        else:
            return self.plans_identifier + '_' + configuration_name