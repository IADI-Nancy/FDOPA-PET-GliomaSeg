from typing import Union, Type, List, Tuple

import torch.nn as nn
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD

class AuxiliaryRegressor(nn.Module):
    # Similar to SENet https://github.com/Project-MONAI/MONAI/blob/46a5272196a6c2590ca2589029eed8e4d56ff008/monai/networks/nets/senet.py#L271
    # or VGG with dropout
    def __init__(self, 
                 input_feature_channels: int,
                 num_outputs: int, 
                 global_pooling: str='average',
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None):
        super().__init__()
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs

        if global_pooling == 'average' or global_pooling == 'avg':
            self.global_pooling = nn.AdaptiveAvgPool3d(1)
        elif global_pooling == 'maximum' or global_pooling == 'max':
            self.global_pooling = nn.AdaptiveMaxPool3d(1)
        else:
            raise ValueError('Wrong value for global_pooling: %s. Must be one of "average", "avg", "maximum", "max".')
        
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None

        self.linear = nn.Linear(input_feature_channels, num_outputs)
        
    def forward(self, x):
        x = self.global_pooling(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)        

        x = self.linear(x)

        return x
        
class PlainConvUNetAuxiliaryRegressor(PlainConvUNet):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 num_outputs_aux_reg: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 dropout_op_aux_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 global_pooling: str='average'
                 ):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                         n_conv_per_stage, num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs,
                         dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, nonlin_first)
        self.auxiliary_regressor = AuxiliaryRegressor(self.encoder.output_channels[-1],  
                                                      num_outputs_aux_reg, global_pooling, dropout_op, dropout_op_aux_kwargs)
    
    def forward(self, x):
        skips = self.encoder(x)
        # Only return the output of auxiliary_regressor at train train time
        # This is equivalent as the nnunet decoder deep supervision so we can use this to ensure it is compatible everywhere
        if not self.decoder.deep_supervision:
            return self.decoder(skips)
        else:
            return self.decoder(skips), self.auxiliary_regressor(skips[-1])
    

class ResidualEncoderUNetAuxiliaryRegressor(ResidualEncoderUNet):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 num_outputs_aux_reg: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 dropout_op_aux_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 global_pooling: str='average'
                 ):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                         n_blocks_per_stage, num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs,
                         dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, block, bottleneck_channels, 
                         stem_channels)
        self.auxiliary_regressor = AuxiliaryRegressor(self.encoder.output_channels[-1], 
                                                        num_outputs_aux_reg, global_pooling, dropout_op, dropout_op_aux_kwargs)
        
    def forward(self, x):
        skips = self.encoder(x)
        # Only return the output of auxiliary_regressor at train train time
        # This is equivalent as the nnunet decoder deep supervision so we can use this to ensure it is compatible everywhere
        if not self.decoder.deep_supervision:
            return self.decoder(skips)
        else:
            return self.decoder(skips), self.auxiliary_regressor(skips[-1])