from typing import Union, Type, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder

class SepConv(nn.Module):
    # https://github.com/ArchipLab-LinfengZhang/contrastive-deep-supervision/blob/main/CIFAR/resnet.py#L30
    def __init__(self,
               conv_op: Type[_ConvNd],
               input_channels: int,
               output_channels: int,
               kernel_size: Union[int, List[int], Tuple[int, ...]],
               stride: Union[int, List[int], Tuple[int, ...]],
               conv_bias: bool = False,
               norm_op: Union[None, Type[nn.Module]] = None,
               norm_op_kwargs: dict = None,
               nonlin: Union[None, Type[torch.nn.Module]] = None,
               nonlin_kwargs: dict = None
               ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        self.conv1 = conv_op(input_channels, input_channels, kernel_size, stride, padding=[(i - 1) // 2 for i in kernel_size],
                             dilation=1, bias=conv_bias, groups=input_channels)
        ops.append(self.conv1)
        self.conv2 = conv_op(input_channels, input_channels, 1, 1, padding=0, dilation=1, bias=conv_bias)
        ops.append(self.conv2)
        if norm_op is not None:
            self.norm1 = norm_op(input_channels, **norm_op_kwargs)
            ops.append(self.norm1)
        if nonlin is not None:
            self.nonlin1 = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin1)
        self.conv3 = conv_op(input_channels, input_channels, kernel_size, 1, padding=[(i - 1) // 2 for i in kernel_size],
                             dilation=1, bias=conv_bias, groups=input_channels)
        ops.append(self.conv3)
        self.conv4 = conv_op(input_channels, output_channels, 1, 1, padding=0, dilation=1, bias=conv_bias)
        ops.append(self.conv4)
        if norm_op is not None:
            self.norm2 = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm2)
        if nonlin is not None:
            self.nonlin2 = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin2)

        self.all_modules = nn.Sequential(*ops)
    
    def forward(self, x):
        return self.all_modules(x)
    
    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)


class ProjectionHead(nn.Module):
    # Projection head design from http://arxiv.org/abs/2207.05306 
    # https://github.com/ArchipLab-LinfengZhang/contrastive-deep-supervision/blob/main/CIFAR/resnet.py#L136
    # Ojective : projection heads map the backbone features into a normalized embedding space, where the contrastive learning loss is applied
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 ):
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        if conv_op == nn.Conv2d:
            gap = nn.AdaptiveAvgPool2d(1)
        elif conv_op == nn.Conv3d:
            gap = nn.AdaptiveAvgPool3d(1)
        elif conv_op == nn.Conv1d:
            gap = nn.AdaptiveAvgPool1d(1)

        if num_convs == 0:
            self.proj = gap
            self.output_channels = input_channels
        else:
            self.proj = nn.Sequential(
                SepConv(
                    conv_op, input_channels, output_channels[0], kernel_size, stride, conv_bias, norm_op,
                    norm_op_kwargs, nonlin, nonlin_kwargs
                ),
                *[
                    SepConv(
                        conv_op, output_channels[i - 1], output_channels[i], kernel_size, stride, conv_bias, norm_op,
                        norm_op_kwargs, nonlin, nonlin_kwargs
                    )
                    for i in range(1, num_convs)
                ],
                gap
            )
            self.output_channels = output_channels[-1]
        self.stride = maybe_convert_scalar_to_list(conv_op, stride)

    def forward(self, x):
        return self.proj(x)
    
    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output
    

class PlainConvEncoderContrastiveSupervision(PlainConvEncoder):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 deep_supervision,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv',
                 ):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, 
                         strides, n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                         dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips, nonlin_first, pool)
        self.deep_supervision = deep_supervision
        contrastive_supervision_stages = []
        for s in range(n_stages):
            # contrastive_supervision_stages.append(ProjectionHead(n_stages - s - 1, conv_op, features_per_stage[s], features_per_stage[s + 1:],
            #                                                      kernel_sizes[s], 1, conv_bias, norm_op, norm_op_kwargs,
            #                                                      nonlin, nonlin_kwargs,))
            contrastive_supervision_stages.append(ProjectionHead(1, conv_op, features_per_stage[s], features_per_stage[-1],
                                                                 kernel_sizes[s], 1, conv_bias, norm_op, norm_op_kwargs,
                                                                 nonlin, nonlin_kwargs,))
        self.contrastive_supervision_stages = nn.Sequential(*contrastive_supervision_stages)

    def forward(self, x):
        ret = []
        contrastive_supervision_ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            ret.append(x)
            if self.deep_supervision:
                # Get projected features
                proj_features = self.contrastive_supervision_stages[s](x)
                # Flatten
                proj_features = proj_features.view(proj_features.size(0), -1)
                # Normalize
                proj_features = F.normalize(proj_features, dim=1)
                contrastive_supervision_ret.append(proj_features)

        if self.return_skips:
            if not self.deep_supervision:
                return ret
            else:
                return ret, contrastive_supervision_ret
        else:
            if not self.deep_supervision:
                return ret[-1]
            else:
                return ret[-1], contrastive_supervision_ret
    

class PlainConvUNetContrastiveSupervision(PlainConvUNet):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, 
                         kernel_sizes, strides, n_conv_per_stage, num_classes, 
                         n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs,
                         dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, nonlin_first)
        self.encoder = PlainConvEncoderContrastiveSupervision(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                                              n_conv_per_stage, deep_supervision, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                                              dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                                              nonlin_first=nonlin_first)
    
    def forward(self, x_i, x_j=None):
        # x_j is None at val time but not a problem since there is no deep supervision
        # If we want x_j anyway we need to do a custom nnUNetPredictor to handle this
        if self.encoder.deep_supervision:
            skips, z_i = self.encoder(x_i)
            _, z_j = self.encoder(x_j)
            # z_i and z_j are lists of tensor with the projector output at each stage
            # DeepSupervisionWrapper takes as input a list of tensor for each stage that will be given to the loss 
            # Constrative loss takes as input a tensor of shape (2N, D) i.e. the concatenatation of features of z_i and z_j for a stage
            # The output of this function for the contrastive part must be a list of stage tensors of shape (2N, D)
            z = [torch.cat([i, j], dim=0) for i, j in zip(*[z_i, z_j])]
        else:
            skips = self.encoder(x_i)
        # Only return the output of contrastive deep supervision at train train time
        # This is equivalent as the nnunet decoder deep supervision so we can use this to ensure it is compatible everywhere
        if not self.decoder.deep_supervision:
            return self.decoder(skips)
        else:
            return self.decoder(skips), z
        

class ResidualEncoderContrastiveSupervision(ResidualEncoder):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 deep_supervision,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage, 
                         conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, block,
                         bottleneck_channels, return_skips, disable_default_stem, stem_channels, pool_type, stochastic_depth_p,
                         squeeze_excitation, squeeze_excitation_reduction_ratio)
        self.deep_supervision = deep_supervision
        contrastive_supervision_stages = []
        for s in range(n_stages):
            # contrastive_supervision_stages.append(ProjectionHead(n_stages - s - 1, conv_op, features_per_stage[s], features_per_stage[s + 1:],
            #                                                      kernel_sizes[s], 1, conv_bias, norm_op, norm_op_kwargs,
            #                                                      nonlin, nonlin_kwargs,))
            contrastive_supervision_stages.append(ProjectionHead(1 if s != (n_stages - 1) else 0, conv_op, 
                                                                 features_per_stage[s], features_per_stage[-1],
                                                                 kernel_sizes[s], 1, conv_bias, norm_op, norm_op_kwargs,
                                                                 nonlin, nonlin_kwargs,))
        self.contrastive_supervision_stages = nn.Sequential(*contrastive_supervision_stages)

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        contrastive_supervision_ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            ret.append(x)
            if self.deep_supervision:
                # Get projected features
                proj_features = self.contrastive_supervision_stages[s](x)
                # Flatten
                proj_features = proj_features.view(proj_features.size(0), -1)
                # Normalize
                proj_features = F.normalize(proj_features, dim=1)
                contrastive_supervision_ret.append(proj_features)

        if self.return_skips:
            if not self.deep_supervision:
                return ret
            else:
                return ret, contrastive_supervision_ret
        else:
            if not self.deep_supervision:
                return ret[-1]
            else:
                return ret[-1], contrastive_supervision_ret
            

class ResidualEncoderUNetContrastiveSupervision(ResidualEncoderUNet):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None):
        super().__init__(input_channels, n_stages, features_per_stage, 
        conv_op, kernel_sizes, strides, n_blocks_per_stage, num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, 
        norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, block, bottleneck_channels, stem_channels)

        self.encoder = ResidualEncoderContrastiveSupervision(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                                             n_blocks_per_stage, deep_supervision, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                                             dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                                             return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        
    def forward(self, x_i, x_j=None):
        # x_j is None at val time but not a problem since there is no deep supervision
        # If we want x_j anyway we need to do a custom nnUNetPredictor to handle this
        if not self.encoder.deep_supervision:
            skips = self.encoder(x_i)
        else:
            skips, z_i = self.encoder(x_i)
            _, z_j = self.encoder(x_j)
            # z_i and z_j are lists of tensor with the projector output at each stage
            # DeepSupervisionWrapper takes as input a list of tensor for each stage that will be given to the loss 
            # Constrative loss takes as input a tensor of shape (2N, D) i.e. the concatenatation of features of z_i and z_j for a stage
            # The output of this function for the contrastive part must be a list of stage tensors of shape (2N, D)
            z = [torch.cat([i, j], dim=0) for i, j in zip(*[z_i, z_j])]
        # Only return the output of contrastive deep supervision at train train time
        # This is equivalent as the nnunet decoder deep supervision so we can use this to ensure it is compatible everywhere
        if not self.decoder.deep_supervision:
            return self.decoder(skips)
        else:
            return self.decoder(skips), z