from typing import List, Union, Tuple
from nnunetv2.network_architecture.attention_network import PlainConvUNetSE, PlainConvUNetscSE, PlainConvUNetPE, ResidualEncoderUNetSE, ResidualEncoderUNetscSE, ResidualEncoderUNetPE 
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import nnUNetPlannerResEncM, nnUNetPlannerResEncL, nnUNetPlannerResEncXL

# TODO : maybe do differently if we want that the attention part is part of VRAM estimate for planning
# Need to modify inside get_plans_for_configuration instead of calling parent class but ATM it is faster like this

class ExperimentPlannerUNetSE(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetPlansSE',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = PlainConvUNetSE
        
    def get_plans_for_configuration(self, spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache):
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache)
        plan['architecture']['arch_kwargs'].update({"squeeze_excitation": True, "squeeze_excitation_reduction_ratio": 1./16,
                                                    'attention_placement_strategy': 'P5'})
        return plan
    
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


class ExperimentPlannerUNetscSE(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetPlansscSE',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = PlainConvUNetscSE
        
    def get_plans_for_configuration(self, spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache):
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache)
        plan['architecture']['arch_kwargs'].update({"squeeze_excitation": True, "squeeze_excitation_reduction_ratio": 2,
                                                    'attention_placement_strategy': 'P4'})
        return plan
    
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


class ExperimentPlannerUNetPE(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetPlansPE',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = PlainConvUNetPE
        
    def get_plans_for_configuration(self, spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache):
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache)
        plan['architecture']['arch_kwargs'].update({"squeeze_excitation": True, "squeeze_excitation_reduction_ratio": 2,
                                                    'attention_placement_strategy': 'P6'})
        return plan
    
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


class nnUNetPlannerResEncMSE(nnUNetPlannerResEncM):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetMPlansSE',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNetSE

    def get_plans_for_configuration(self, spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache):
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache)
        plan['architecture']['arch_kwargs'].update({"squeeze_excitation": True, "squeeze_excitation_reduction_ratio": 1./16,
                                                    'attention_placement_strategy': 'P5'})
        return plan
    
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


class nnUNetPlannerResEncLSE(nnUNetPlannerResEncL):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetLPlansSE',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNetSE

    def get_plans_for_configuration(self, spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache):
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache)
        plan['architecture']['arch_kwargs'].update({"squeeze_excitation": True, "squeeze_excitation_reduction_ratio": 1./16,
                                                    'attention_placement_strategy': 'P5'})
        return plan
    
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


class nnUNetPlannerResEncXLSE(nnUNetPlannerResEncXL):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 40,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetXLPlansSE',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNetSE

    def get_plans_for_configuration(self, spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache):
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache)
        plan['architecture']['arch_kwargs'].update({"squeeze_excitation": True, "squeeze_excitation_reduction_ratio": 1./16,
                                                    'attention_placement_strategy': 'P5'})
        return plan
    
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


class nnUNetPlannerResEncMscSE(nnUNetPlannerResEncM):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetMPlansscSE',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNetscSE

    def get_plans_for_configuration(self, spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache):
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache)
        plan['architecture']['arch_kwargs'].update({"squeeze_excitation": True, "squeeze_excitation_reduction_ratio": 2,
                                                    'attention_placement_strategy': 'P4'})
        return plan
    
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


class nnUNetPlannerResEncLscSE(nnUNetPlannerResEncL):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetLPlansscSE',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNetscSE

    def get_plans_for_configuration(self, spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache):
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache)
        plan['architecture']['arch_kwargs'].update({"squeeze_excitation": True, "squeeze_excitation_reduction_ratio": 2,
                                                    'attention_placement_strategy': 'P4'})
        return plan
    
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


class nnUNetPlannerResEncXLscSE(nnUNetPlannerResEncXL):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 40,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetXLPlansscSE',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNetscSE

    def get_plans_for_configuration(self, spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache):
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache)
        plan['architecture']['arch_kwargs'].update({"squeeze_excitation": True, "squeeze_excitation_reduction_ratio": 2,
                                                    'attention_placement_strategy': 'P4'})
        return plan
    
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


class nnUNetPlannerResEncMPE(nnUNetPlannerResEncM):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetMPlansPE',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNetPE

    def get_plans_for_configuration(self, spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache):
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache)
        plan['architecture']['arch_kwargs'].update({"squeeze_excitation": True, "squeeze_excitation_reduction_ratio": 2,
                                                    'attention_placement_strategy': 'P6'})
        return plan
    
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


class nnUNetPlannerResEncLPE(nnUNetPlannerResEncL):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetLPlansPE',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNetPE

    def get_plans_for_configuration(self, spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache):
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache)
        plan['architecture']['arch_kwargs'].update({"squeeze_excitation": True, "squeeze_excitation_reduction_ratio": 2,
                                                    'attention_placement_strategy': 'P6'})
        return plan
    
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


class nnUNetPlannerResEncXLPE(nnUNetPlannerResEncXL):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 40,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetXLPlansPE',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNetPE

    def get_plans_for_configuration(self, spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache):
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier, approximate_n_voxels_dataset, _cache)
        plan['architecture']['arch_kwargs'].update({"squeeze_excitation": True, "squeeze_excitation_reduction_ratio": 2,
                                                    'attention_placement_strategy': 'P6'})
        return plan
    
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