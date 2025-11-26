# Base code from https://github.com/ai-med/squeeze_and_excitation/blob/master/squeeze_and_excitation/squeeze_and_excitation_3D.py
# Adapted to work for 2D + 3D and to match dynamic_network_architectures.building_blocks.regularization.SqueezeExcite
# We just don't reproduce the make_divisible as I can't really state if it's worth it. Difficult to understand compare to //
from typing import Union, Type, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from dynamic_network_architectures.building_blocks.regularization import SqueezeExcite
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD, StackedResidualBlocks
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op, get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0


class ResidualEncoderUNetSE(ResidualEncoderUNet):
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
                 stem_channels: int = None,
                 squeeze_excitation: bool = True,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 attention_placement_strategy: str = 'P5'
                 ):
        """
        attention_placement_strategy (str, optional): where should be placed the attention blocks
        P1 = Encoder only
        P2 = Decoder only
        P3 = Bottleck only
        P4 = Encoder + Decoder
        P5 = Encoder + Bottleneck
        P6 = Encodeur + Bottleneck + Decoder
        default = P5 to match original nnUNet implementation (not activated but present in Encoder + Bottleneck)
        """
        if attention_placement_strategy not in ['P%d' % _ for _ in range(1,7)]:
            raise ValueError('Wrong attention_placement_strategy')
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage,
                         num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                         dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, block, bottleneck_channels, stem_channels)
        if attention_placement_strategy != 'P2':
            self.encoder = ResidualEncoderSE(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, 
                                            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, 
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels, 
                                            return_skips=True, disable_default_stem=False, stem_channels=stem_channels, 
                                            squeeze_excitation=squeeze_excitation,  
                                            squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio,
                                            attention_placement_strategy=attention_placement_strategy)
        if attention_placement_strategy in ['P2', 'P4', 'P6']:
            self.decoder = UNetDecoderSE(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                         squeeze_excitation=squeeze_excitation, 
                                         squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio)
        
        
class ResidualEncoderSE(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
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
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 attention_placement_strategy='P4'
                 ):
        """

        :param input_channels:
        :param n_stages:
        :param features_per_stage: Note: If the block is BottleneckD, then this number is supposed to be the number of
        features AFTER the expansion (which is not coded implicitly in this repository)! See todo!
        :param conv_op:
        :param kernel_sizes:
        :param strides:
        :param n_blocks_per_stage:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param block:
        :param bottleneck_channels: only needed if block is BottleneckD
        :param return_skips: set this to True if used as encoder in a U-Net like network
        :param disable_default_stem: If True then no stem will be created. You need to build your own and ensure it is executed first, see todo.
        The stem in this implementation does not so stride/pooling so building your own stem is a necessity if you need this.
        :param stem_channels: if None, features_per_stage[0] will be used for the default stem. Not recommended for BottleneckD
        :param pool_type: if conv, strided conv will be used. avg = average pooling, max = max pooling
        """
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if bottleneck_channels is None or isinstance(bottleneck_channels, int):
            bottleneck_channels = [bottleneck_channels] * n_stages
        assert len(
            bottleneck_channels) == n_stages, "bottleneck_channels must be None or have as many entries as we have resolution stages (n_stages)"
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        # build a stem, Todo maybe we need more flexibility for this in the future. For now, if you need a custom
        #  stem you can just disable the stem and build your own.
        #  THE STEM DOES NOT DO STRIDE/POOLING IN THIS IMPLEMENTATION
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            self.stem = StackedConvBlocks(1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, conv_bias,
                                          norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
            input_channels = stem_channels
        else:
            self.stem = None

        # now build the network
        stages = []
        for s in range(n_stages):
            stride_for_conv = strides[s] if pool_op is None else 1
           
            if attention_placement_strategy in ['P1', 'P4']:
                stage = StackedResidualBlocks(
                    n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], stride_for_conv,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                    block=block, bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                    squeeze_excitation=squeeze_excitation if s != (n_stages - 1) else False,
                    squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
                )
            elif attention_placement_strategy in ['P5', 'P6']:
                stage = StackedResidualBlocks(
                    n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], stride_for_conv,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                    block=block, bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                    squeeze_excitation=squeeze_excitation,
                    squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
                )
            elif attention_placement_strategy == 'P3':
                stage = StackedResidualBlocks(
                    n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], stride_for_conv,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                    block=block, bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                    squeeze_excitation=squeeze_excitation if s == (n_stages - 1) else False,
                    squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
                )
            else:
                raise ValueError('Wrong attention_placement_strategy')   

            if pool_op is not None:
                stage = nn.Sequential(pool_op(strides[s]), stage)

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output
        

class StackedConvBlocksSE(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16
                 ):
        """

        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param squeeze_excitation:
        :param squeeze_excitation_reduction_ratio:
        """
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
                for i in range(1, num_convs)
            ]
        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)
        
        # Squeeze Excitation
        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = SqueezeExcite(self.output_channels, conv_op,
                                                    rd_ratio=squeeze_excitation_reduction_ratio, rd_divisor=8)

    def forward(self, x):
        out = self.convs(x)
        if self.apply_se:
            out = self.squeeze_excitation(out)
        return out

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        print(size_after_stride)
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output
    
    
class PlainConvEncoderSE(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
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
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 attention_placement_strategy='P4'
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            if attention_placement_strategy in ['P1', 'P4']:
                stage_modules.append(StackedConvBlocksSE(
                    n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first,
                    squeeze_excitation if s != (n_stages - 1) else False, squeeze_excitation_reduction_ratio
                ))
            elif attention_placement_strategy in ['P5', 'P6']:
                stage_modules.append(StackedConvBlocksSE(
                    n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first,
                    squeeze_excitation, squeeze_excitation_reduction_ratio
                ))
            elif attention_placement_strategy == 'P3':
                stage_modules.append(StackedConvBlocksSE(
                        n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                        conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first,
                        squeeze_excitation if s != (n_stages - 1) else False, squeeze_excitation_reduction_ratio
                    ))
            else:
                raise ValueError('Wrong attention_placement_strategy')  
            
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output
    
    
class UNetDecoderSE(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoderSE, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16
                 ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs


        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocksSE(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first,
                squeeze_excitation,
                squeeze_excitation_reduction_ratio
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output


class PlainConvUNetSE(PlainConvUNet):
    def __init__(
        self,
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
        nonlin_first: bool = False,
        squeeze_excitation: bool = True,
        squeeze_excitation_reduction_ratio: float = 1. / 16,
        attention_placement_strategy: str = 'P5'
    ):
        """
        attention_placement_strategy (str, optional): where should be placed the attention blocks
        P1 = Encoder only
        P2 = Decoder only
        P3 = Bottleck only
        P4 = Encoder + Decoder
        P5 = Encoder + Bottleneck
        P6 = Encodeur + Bottleneck + Decoder
        default = P5 to match original nnUNet implementation (not activated but present in Encoder + Bottleneck)
        """
        if attention_placement_strategy not in ['P%d' % _ for _ in range(1,7)]:
            raise ValueError('Wrong attention_placement_strategy')
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, 
                         strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder, conv_bias, 
                         norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, 
                         deep_supervision, nonlin_first)
        if attention_placement_strategy != 'P2':
            self.encoder = PlainConvEncoderSE(
                input_channels,
                n_stages,
                features_per_stage,
                conv_op,
                kernel_sizes,
                strides,
                n_conv_per_stage,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                return_skips=True,
                nonlin_first=nonlin_first,
                squeeze_excitation=squeeze_excitation,
                squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio,
                attention_placement_strategy=attention_placement_strategy
            )  
        if attention_placement_strategy in ['P2', 'P4', 'P6']:
            self.decoder = UNetDecoderSE(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                         nonlin_first=nonlin_first,
                                         squeeze_excitation=squeeze_excitation, 
                                         squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio)


class ChannelSELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, channels, conv_op, reduction_ratio=2,
                 act_layer=nn.ReLU, norm_layer=None, gate_layer=nn.Sigmoid):
        """
        :param channels: No of input channels
        :param reduction_ratio: By how much should the channels should be reduced
        """
        super(ChannelSELayer, self).__init__()

        if conv_op == nn.Conv2d:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif conv_op == nn.Conv3d:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        elif conv_op == nn.Conv1d:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        rd_channels = channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(channels, rd_channels, bias=True)
        self.fc2 = nn.Linear(rd_channels, channels, bias=True)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = act_layer(inplace=True)
        self.gate = gate_layer()


    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, channels, D, H, W) for 3D and (batch_size, channels, H, W) for 2D
        :return: output tensor
        """
        batch_size, channels = input_tensor.size()[:2]
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.act(self.bn(self.fc1(squeeze_tensor.view(batch_size, channels))))
        fc_out_2 = self.gate(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, channels, *[1]*(input_tensor.ndim - 2)))

        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, channels, conv_op, gate_layer=nn.Sigmoid):
        """
        :param channels: No of input channels

        """
        super(SpatialSELayer, self).__init__()
        self.conv = conv_op(channels, 1, 1)
        self.gate = gate_layer()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, channels, D, H, W) for 3D and (batch_size, channels, H, W) for 2D
        :return: output_tensor
        """
        # channel squeeze
        global_shape = input_tensor.size()
        batch_size, channels = global_shape[:2]
        image_shape = global_shape[2:]

        if weights:
            if len(image_shape) == 2:
                # In 2D case there is this line, don't understand why but I copy it for matching original code
                weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channels, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.gate(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, *image_shape))

        return output_tensor


class ChannelSpatialSELayer(nn.Module):
    """
       Extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, channels, conv_op, reduction_ratio=2, act_layer=nn.ReLU, norm_layer=None, gate_layer=nn.Sigmoid):
        """
        :param channels: No of input channels
        :param reduction_ratio: By how much should the channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(channels, conv_op, reduction_ratio, act_layer, norm_layer, gate_layer)
        self.sSE = SpatialSELayer(channels, conv_op, gate_layer)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class ProjectExciteLayer(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, channels, conv_op, reduction_ratio=2,
                 act_layer=nn.ReLU, norm_layer=None, gate_layer=nn.Sigmoid):
        """
        :param channels: No of input channels
        :param reduction_ratio: By how much should the channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        if not (conv_op == nn.Conv2d or conv_op == nn.Conv3d):
            raise ValueError('ProjectExciteLayer only implemented for 2D and 3D data')
        rd_channels = channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.act = act_layer()
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.conv_c = conv_op(in_channels=channels, out_channels=rd_channels, kernel_size=1, stride=1)
        self.conv_cT = conv_op(in_channels=rd_channels, out_channels=channels, kernel_size=1, stride=1)
        self.gate = gate_layer()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, channels, D, H, W) for 3D and (batch_size, channels, H, W) for 2D
        :return: output tensor
        """
        global_shape = input_tensor.size()
        batch_size, channels = global_shape[:2]
        image_shape = global_shape[2:]

        # Project:
        if len(image_shape) == 3:
            D, H, W = image_shape
            # Average along channels and different axes
            squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

            squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

            squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

            # tile tensors to original size and add:
            final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, channels, 1, 1, W),
                                        squeeze_tensor_h.view(batch_size, channels, 1, H, 1),
                                        squeeze_tensor_d.view(batch_size, channels, D, 1, 1)])
        else:
            H, W = image_shape
            # Average along channels and different axes
            squeeze_tensor_h = F.adaptive_avg_pool2d(input_tensor, (1, W))

            squeeze_tensor_d = F.adaptive_avg_pool2d(input_tensor, (H, 1))

            # tile tensors to original size and add:
            final_squeeze_tensor = sum([squeeze_tensor_h.view(batch_size, channels, 1, W),
                                        squeeze_tensor_d.view(batch_size, channels, H, 1)])

        # Excitation:
        final_squeeze_tensor = self.gate(self.conv_cT(self.act(self.bn(self.conv_c(final_squeeze_tensor)))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor
        

class BasicBlockDscSE(BasicBlockD):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 2 # WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
                 # todo wideresnet?
                 ):
        super().__init__(conv_op, input_channels, output_channels, kernel_size, stride, conv_bias, norm_op,
                         norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                         False, squeeze_excitation_reduction_ratio)
        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = ChannelSpatialSELayer(self.output_channels, conv_op, reduction_ratio=squeeze_excitation_reduction_ratio)


class BasicBlockDPE(BasicBlockD):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 2 # WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
                 # todo wideresnet?
                 ):
        super().__init__(conv_op, input_channels, output_channels, kernel_size, stride, conv_bias, norm_op,
                         norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                         False, squeeze_excitation_reduction_ratio)
        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = ProjectExciteLayer(self.output_channels, conv_op, reduction_ratio=squeeze_excitation_reduction_ratio)


class BottleneckDscSE(BottleneckD):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 2 # WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
                 # todo wideresnet?
                 ):
        super().__init__(conv_op, input_channels, output_channels, kernel_size, stride, conv_bias, norm_op,
                         norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                         False, squeeze_excitation_reduction_ratio)
        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = ChannelSpatialSELayer(self.output_channels, conv_op, reduction_ratio=squeeze_excitation_reduction_ratio)


class BottleneckDPE(BottleneckD):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 2 # WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
                 # todo wideresnet?
                 ):
        super().__init__(conv_op, input_channels, output_channels, kernel_size, stride, conv_bias, norm_op,
                         norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                         False, squeeze_excitation_reduction_ratio)
        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = ProjectExciteLayer(self.output_channels, conv_op, reduction_ratio=squeeze_excitation_reduction_ratio)


class StackedResidualBlocksscSE(nn.Module):
    def __init__(self,
                 n_blocks: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 2 
                 ):
        """
        Stack multiple instances of block.

        :param n_blocks: number of residual blocks
        :param conv_op: nn.ConvNd class
        :param input_channels: only relevant for forst block in the sequence. This is the input number of features.
        After the first block, the number of features in the main path to which the residuals are added is output_channels
        :param output_channels: number of features in the main path to which the residuals are added (and also the
        number of features of the output)
        :param kernel_size: kernel size for all nxn (n!=1) convolutions. Default: 3x3
        :param initial_stride: only affects the first block. All subsequent blocks have stride 1
        :param conv_bias: usually False
        :param norm_op: nn.BatchNormNd, InstanceNormNd etc
        :param norm_op_kwargs: dictionary of kwargs. Leave empty ({}) for defaults
        :param dropout_op: nn.DropoutNd, can be None for no dropout
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param block: BasicBlockD or BottleneckD
        :param bottleneck_channels: if block is BottleneckD then we need to know the number of bottleneck features.
        Bottleneck will use first 1x1 conv to reduce input to bottleneck features, then run the nxn (see kernel_size)
        conv on that (bottleneck -> bottleneck). Finally the output will be projected back to output_channels
        (bottleneck -> output_channels) with the final 1x1 conv
        :param stochastic_depth_p: probability of applying stochastic depth in residual blocks
        :param squeeze_excitation: whether to apply squeeze and excitation or not. WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
        :param squeeze_excitation_reduction_ratio: ratio by how much squeeze and excitation should reduce channels
        respective to number of out channels of respective block
        """
        super().__init__()
        assert n_blocks > 0, 'n_blocks must be > 0'
        assert block in [BasicBlockDscSE, BottleneckDscSE], 'block must be BasicBlockDscSE or BottleneckDscSE'
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * n_blocks
        if not isinstance(bottleneck_channels, (tuple, list)):
            bottleneck_channels = [bottleneck_channels] * n_blocks
        # IN CONTRAST TO BASE NNUNET SQUEEZE AND EXCITATION IS ONLY APPLIED ON THE LAST BLOCK SIMILARLY TO ORIGINAL PUBLICATION
        if block == BasicBlockDscSE:
            blocks = nn.Sequential(
                block(conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias,
                      norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                      False, squeeze_excitation_reduction_ratio),
                *[block(conv_op, output_channels[n - 1], output_channels[n], kernel_size, 1, conv_bias, norm_op,
                        norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                        False, squeeze_excitation_reduction_ratio) for n in range(1, n_blocks - 1)],
                block(conv_op, output_channels[n_blocks - 2], output_channels[n_blocks - 1], kernel_size, 1, conv_bias, norm_op,
                        norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                        squeeze_excitation, squeeze_excitation_reduction_ratio)
            )
        else:
            blocks = nn.Sequential(
                block(conv_op, input_channels, bottleneck_channels[0], output_channels[0], kernel_size,
                      initial_stride, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                      nonlin, nonlin_kwargs, stochastic_depth_p, False, squeeze_excitation_reduction_ratio),
                *[block(conv_op, output_channels[n - 1], bottleneck_channels[n], output_channels[n], kernel_size,
                        1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                        nonlin, nonlin_kwargs, stochastic_depth_p, False,
                        squeeze_excitation_reduction_ratio) for n in range(1, n_blocks - 1)],
                block(conv_op, output_channels[n_blocks - 2], bottleneck_channels[n_blocks - 1], output_channels[n_blocks - 1], kernel_size,
                        1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                        nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation,
                        squeeze_excitation_reduction_ratio)
                
            )
        self.blocks = blocks
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)
        self.output_channels = output_channels[-1]

    def forward(self, x):
        return self.blocks(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output = self.blocks[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.blocks[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output
    

class StackedResidualBlocksPE(nn.Module):
    def __init__(self,
                 n_blocks: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 2
                 ):
        """
        Stack multiple instances of block.

        :param n_blocks: number of residual blocks
        :param conv_op: nn.ConvNd class
        :param input_channels: only relevant for forst block in the sequence. This is the input number of features.
        After the first block, the number of features in the main path to which the residuals are added is output_channels
        :param output_channels: number of features in the main path to which the residuals are added (and also the
        number of features of the output)
        :param kernel_size: kernel size for all nxn (n!=1) convolutions. Default: 3x3
        :param initial_stride: only affects the first block. All subsequent blocks have stride 1
        :param conv_bias: usually False
        :param norm_op: nn.BatchNormNd, InstanceNormNd etc
        :param norm_op_kwargs: dictionary of kwargs. Leave empty ({}) for defaults
        :param dropout_op: nn.DropoutNd, can be None for no dropout
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param block: BasicBlockD or BottleneckD
        :param bottleneck_channels: if block is BottleneckD then we need to know the number of bottleneck features.
        Bottleneck will use first 1x1 conv to reduce input to bottleneck features, then run the nxn (see kernel_size)
        conv on that (bottleneck -> bottleneck). Finally the output will be projected back to output_channels
        (bottleneck -> output_channels) with the final 1x1 conv
        :param stochastic_depth_p: probability of applying stochastic depth in residual blocks
        :param squeeze_excitation: whether to apply squeeze and excitation or not. WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
        :param squeeze_excitation_reduction_ratio: ratio by how much squeeze and excitation should reduce channels
        respective to number of out channels of respective block
        """
        super().__init__()
        assert n_blocks > 0, 'n_blocks must be > 0'
        assert block in [BasicBlockDPE, BottleneckDPE], 'block must be BasicBlockDPE or BottleneckDPE'
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * n_blocks
        if not isinstance(bottleneck_channels, (tuple, list)):
            bottleneck_channels = [bottleneck_channels] * n_blocks
        # IN CONTRAST TO BASE NNUNET SQUEEZE AND EXCITATION IS ONLY APPLIED ON THE LAST BLOCK SIMILARLY TO ORIGINAL PUBLICATION
        if block == BasicBlockDscSE:
            blocks = nn.Sequential(
                block(conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias,
                      norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                      False, squeeze_excitation_reduction_ratio),
                *[block(conv_op, output_channels[n - 1], output_channels[n], kernel_size, 1, conv_bias, norm_op,
                        norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                        False, squeeze_excitation_reduction_ratio) for n in range(1, n_blocks - 1)],
                block(conv_op, output_channels[n_blocks - 2], output_channels[n_blocks - 1], kernel_size, 1, conv_bias, norm_op,
                        norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                        squeeze_excitation, squeeze_excitation_reduction_ratio)
            )
        else:
            blocks = nn.Sequential(
                block(conv_op, input_channels, bottleneck_channels[0], output_channels[0], kernel_size,
                      initial_stride, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                      nonlin, nonlin_kwargs, stochastic_depth_p, False, squeeze_excitation_reduction_ratio),
                *[block(conv_op, output_channels[n - 1], bottleneck_channels[n], output_channels[n], kernel_size,
                        1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                        nonlin, nonlin_kwargs, stochastic_depth_p, False,
                        squeeze_excitation_reduction_ratio) for n in range(1, n_blocks - 1)],
                block(conv_op, output_channels[n_blocks - 2], bottleneck_channels[n_blocks - 1], output_channels[n_blocks - 1], kernel_size,
                        1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                        nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation,
                        squeeze_excitation_reduction_ratio)
                
            )
        self.blocks = blocks
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)
        self.output_channels = output_channels[-1]

    def forward(self, x):
        return self.blocks(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output = self.blocks[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.blocks[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


class ResidualEncoderscSE(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
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
                 squeeze_excitation_reduction_ratio: float = 2, # WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
                 attention_placement_strategy: str = 'P4'
                 ):
        """

        :param input_channels:
        :param n_stages:
        :param features_per_stage: Note: If the block is BottleneckD, then this number is supposed to be the number of
        features AFTER the expansion (which is not coded implicitly in this repository)! See todo!
        :param conv_op:
        :param kernel_sizes:
        :param strides:
        :param n_blocks_per_stage:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param block:
        :param bottleneck_channels: only needed if block is BottleneckD
        :param return_skips: set this to True if used as encoder in a U-Net like network
        :param disable_default_stem: If True then no stem will be created. You need to build your own and ensure it is executed first, see todo.
        The stem in this implementation does not so stride/pooling so building your own stem is a necessity if you need this.
        :param stem_channels: if None, features_per_stage[0] will be used for the default stem. Not recommended for BottleneckD
        :param pool_type: if conv, strided conv will be used. avg = average pooling, max = max pooling
        """
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if bottleneck_channels is None or isinstance(bottleneck_channels, int):
            bottleneck_channels = [bottleneck_channels] * n_stages
        assert len(
            bottleneck_channels) == n_stages, "bottleneck_channels must be None or have as many entries as we have resolution stages (n_stages)"
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        # build a stem, Todo maybe we need more flexibility for this in the future. For now, if you need a custom
        #  stem you can just disable the stem and build your own.
        #  THE STEM DOES NOT DO STRIDE/POOLING IN THIS IMPLEMENTATION
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            self.stem = StackedConvBlocks(1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, conv_bias,
                                          norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
            input_channels = stem_channels
        else:
            self.stem = None

        # now build the network
        stages = []
        for s in range(n_stages):
            stride_for_conv = strides[s] if pool_op is None else 1
        
            if attention_placement_strategy in ['P1', 'P4']:
                stage = StackedResidualBlocksscSE(
                    n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], stride_for_conv,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                    block=block, bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                    squeeze_excitation=squeeze_excitation if s != (n_stages - 1) else False,
                    squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
                )
            elif attention_placement_strategy in ['P5', 'P6']:
                stage = StackedResidualBlocksscSE(
                    n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], stride_for_conv,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                    block=block, bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                    squeeze_excitation=squeeze_excitation,
                    squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
                )
            elif attention_placement_strategy == 'P3':
                stage = StackedResidualBlocksscSE(
                    n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], stride_for_conv,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                    block=block, bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                    squeeze_excitation=squeeze_excitation if s == (n_stages - 1) else False,
                    squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
                )
            else:
                raise ValueError('Wrong attention_placement_strategy') 

            if pool_op is not None:
                stage = nn.Sequential(pool_op(strides[s]), stage)

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output
    

class ResidualEncoderPE(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
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
                 squeeze_excitation_reduction_ratio: float = 2, # WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
                 attention_placement_strategy: str = 'P6'
                 ):
        """

        :param input_channels:
        :param n_stages:
        :param features_per_stage: Note: If the block is BottleneckD, then this number is supposed to be the number of
        features AFTER the expansion (which is not coded implicitly in this repository)! See todo!
        :param conv_op:
        :param kernel_sizes:
        :param strides:
        :param n_blocks_per_stage:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param block:
        :param bottleneck_channels: only needed if block is BottleneckD
        :param return_skips: set this to True if used as encoder in a U-Net like network
        :param disable_default_stem: If True then no stem will be created. You need to build your own and ensure it is executed first, see todo.
        The stem in this implementation does not so stride/pooling so building your own stem is a necessity if you need this.
        :param stem_channels: if None, features_per_stage[0] will be used for the default stem. Not recommended for BottleneckD
        :param pool_type: if conv, strided conv will be used. avg = average pooling, max = max pooling
        """
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if bottleneck_channels is None or isinstance(bottleneck_channels, int):
            bottleneck_channels = [bottleneck_channels] * n_stages
        assert len(
            bottleneck_channels) == n_stages, "bottleneck_channels must be None or have as many entries as we have resolution stages (n_stages)"
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        # build a stem, Todo maybe we need more flexibility for this in the future. For now, if you need a custom
        #  stem you can just disable the stem and build your own.
        #  THE STEM DOES NOT DO STRIDE/POOLING IN THIS IMPLEMENTATION
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            self.stem = StackedConvBlocks(1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, conv_bias,
                                          norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
            input_channels = stem_channels
        else:
            self.stem = None

        # now build the network
        stages = []
        for s in range(n_stages):
            stride_for_conv = strides[s] if pool_op is None else 1

            if attention_placement_strategy in ['P1', 'P4']:
                stage = StackedResidualBlocksPE(
                    n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], stride_for_conv,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                    block=block, bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                    squeeze_excitation=squeeze_excitation if s != (n_stages - 1) else False,
                    squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
                )
            elif attention_placement_strategy in ['P5', 'P6']:
                stage = StackedResidualBlocksPE(
                    n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], stride_for_conv,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                    block=block, bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                    squeeze_excitation=squeeze_excitation,
                    squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
                )
            elif attention_placement_strategy == 'P3':
                stage = StackedResidualBlocksPE(
                    n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], stride_for_conv,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                    block=block, bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                    squeeze_excitation=squeeze_excitation if s == (n_stages - 1) else False,
                    squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
                )
            else:
                raise ValueError('Wrong attention_placement_strategy') 

            if pool_op is not None:
                stage = nn.Sequential(pool_op(strides[s]), stage)

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output


class StackedConvBlocksscSE(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 2
                 ):
        """

        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param squeeze_excitation: whether to apply squeeze and excitation or not. WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
        :param squeeze_excitation_reduction_ratio: ratio by how much squeeze and excitation should reduce channels
        respective to number of out channels of respective block
        """
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
                for i in range(1, num_convs)
            ]
        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = ChannelSpatialSELayer(self.output_channels, conv_op, reduction_ratio=squeeze_excitation_reduction_ratio)

    def forward(self, x):
        out = self.convs(x)
        if self.apply_se:
            out = self.squeeze_excitation(out)
        return out

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output
    

class StackedConvBlocksPE(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 2
                 ):
        """

        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param squeeze_excitation: whether to apply squeeze and excitation or not
        # WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
        :param squeeze_excitation_reduction_ratio: ratio by how much squeeze and excitation should reduce channels
        respective to number of out channels of respective block
        """
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
                for i in range(1, num_convs)
            ]
        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = ProjectExciteLayer(self.output_channels, conv_op, reduction_ratio=squeeze_excitation_reduction_ratio)

    def forward(self, x):
        out = self.convs(x)
        if self.apply_se:
            out = self.squeeze_excitation(out)
        return out

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


class PlainConvEncoderscSE(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
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
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 2, # WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
                 attention_placement_strategy='P5'
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            if attention_placement_strategy in ['P1', 'P4']:
                stage_modules.append(StackedConvBlocksscSE(
                    n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first,
                    squeeze_excitation if s != (n_stages - 1) else False, squeeze_excitation_reduction_ratio
                ))
            elif attention_placement_strategy in ['P5', 'P6']:
                stage_modules.append(StackedConvBlocksscSE(
                    n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first,
                    squeeze_excitation, squeeze_excitation_reduction_ratio
                ))
            elif attention_placement_strategy == 'P3':
                stage_modules.append(StackedConvBlocksscSE(
                        n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                        conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first,
                        squeeze_excitation if s != (n_stages - 1) else False, squeeze_excitation_reduction_ratio
                    ))
            else:
                raise ValueError('Wrong attention_placement_strategy')
             
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class PlainConvEncoderPE(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
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
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 2, # WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
                 attention_placement_strategy='P6'
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            if attention_placement_strategy in ['P1', 'P4']:
                stage_modules.append(StackedConvBlocksPE(
                    n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first,
                    squeeze_excitation if s != (n_stages - 1) else False, squeeze_excitation_reduction_ratio
                ))
            elif attention_placement_strategy in ['P5', 'P6']:
                stage_modules.append(StackedConvBlocksPE(
                    n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first,
                    squeeze_excitation, squeeze_excitation_reduction_ratio
                ))
            elif attention_placement_strategy == 'P3':
                stage_modules.append(StackedConvBlocksPE(
                        n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                        conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first,
                        squeeze_excitation if s != (n_stages - 1) else False, squeeze_excitation_reduction_ratio
                    ))
            else:
                raise ValueError('Wrong attention_placement_strategy')
            
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output
    

class UNetDecoderscSE(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoderscSE, ResidualEncoderscSE],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 2 # WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
                 ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs


        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocksscSE(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first,
                squeeze_excitation,
                squeeze_excitation_reduction_ratio
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output
    

class UNetDecoderPE(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoderPE, ResidualEncoderPE],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 2 # WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
                 ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs


        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocksPE(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first,
                squeeze_excitation,
                squeeze_excitation_reduction_ratio
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output
    

class ResidualEncoderUNetscSE(ResidualEncoderUNet):
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
                 block: Union[Type[BasicBlockDscSE], Type[BottleneckDscSE]] = BasicBlockDscSE,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 2, # WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
                 attention_placement_strategy: str = 'P4'
                 ):
        """
        attention_placement_strategy (str, optional): where should be placed the attention blocks
        P1 = Encoder only
        P2 = Decoder only
        P3 = Bottleck only
        P4 = Encoder + Decoder
        P5 = Encoder + Bottleneck
        P6 = Encodeur + Bottleneck + Decoder
        default = P4 as in original publication the best configuration was SE blocks only after encoder and decoder and not after bottleneck
        https://arxiv.org/pdf/1808.08127    
        """
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        if attention_placement_strategy != 'P2':
            self.encoder = ResidualEncoderscSE(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, 
                                               n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, 
                                               dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels, 
                                               return_skips=True, disable_default_stem=False, stem_channels=stem_channels, 
                                               squeeze_excitation=squeeze_excitation,  
                                               squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio)
        else:
            self.encoder = ResidualEncoderscSE(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, 
                                               n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, 
                                               dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels, 
                                               return_skips=True, disable_default_stem=False, stem_channels=stem_channels, 
                                               squeeze_excitation=False)
        if attention_placement_strategy in ['P2', 'P4', 'P6']:
            self.decoder = UNetDecoderscSE(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                        squeeze_excitation=squeeze_excitation, 
                                        squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio)
        else:
            self.decoder = UNetDecoderscSE(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, 
                                           squeeze_excitation=False)
    
    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)
        

class ResidualEncoderUNetPE(nn.Module):
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
                 block: Union[Type[BasicBlockDPE], Type[BottleneckDPE]] = BasicBlockDPE,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 2, # WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
                 attention_placement_strategy: str = 'P6'
                 ):
        """
        attention_placement_strategy (str, optional): where should be placed the attention blocks
        P1 = Encoder only
        P2 = Decoder only
        P3 = Bottleck only
        P4 = Encoder + Decoder
        P5 = Encoder + Bottleneck
        P6 = Encodeur + Bottleneck + Decoder
        default = P6 as in original publication, the best configuration was SE blocks after encoder, bottleneck and decoder
        https://arxiv.org/pdf/1906.04649
        """
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        if attention_placement_strategy != 'P2':
            self.encoder = ResidualEncoderPE(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                            dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                            return_skips=True, disable_default_stem=False, stem_channels=stem_channels, 
                                            squeeze_excitation=squeeze_excitation,
                                            squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio)
        else:
            self.encoder = ResidualEncoderPE(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, 
                                             n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, 
                                             dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels, 
                                             return_skips=True, disable_default_stem=False, stem_channels=stem_channels, 
                                             squeeze_excitation=False)
        if attention_placement_strategy in ['P2', 'P4', 'P6']:
            self.decoder = UNetDecoderPE(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                        squeeze_excitation=squeeze_excitation, 
                                        squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio)
        else:
            self.decoder = UNetDecoderPE(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                        squeeze_excitation=False)
        
    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)
        

class PlainConvUNetscSE(PlainConvUNet):
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
                 nonlin_first: bool = False,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 2, # WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
                 attention_placement_strategy: str = 'P4'
                 ):
        """
        attention_placement_strategy (str, optional): where should be placed the attention blocks
        P1 = Encoder only
        P2 = Decoder only
        P3 = Bottleck only
        P4 = Encoder + Decoder
        P5 = Encoder + Bottleneck
        P6 = Encodeur + Bottleneck + Decoder
        default = P4 as in original publication the best configuration was SE blocks only after encoder and decoder and not after bottleneck
        https://arxiv.org/pdf/1808.08127   
        """
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage,
                          num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, dropout_op, 
                          dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, nonlin_first)
        if attention_placement_strategy != 'P2':
            self.encoder = PlainConvEncoderscSE(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                                n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                                dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                                nonlin_first=nonlin_first, squeeze_excitation=squeeze_excitation,
                                                squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio)
        if attention_placement_strategy in ['P2', 'P4', 'P6']:
            self.decoder = UNetDecoderscSE(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                        nonlin_first=nonlin_first, squeeze_excitation=squeeze_excitation,
                                        squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio)
        

class PlainConvUNetPE(PlainConvUNet):
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
                 nonlin_first: bool = False,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 2, # WARNING IN CONTRAST TO NNUNET SE IMPLEMENTATION RATIO HERE IS > 1 BECAUSE WE DIVIDE INSTEAD OF MULTIPLYING
                 attention_placement_strategy: str = 'P6'
                 ):
        """
        attention_placement_strategy (str, optional): where should be placed the attention blocks
        P1 = Encoder only
        P2 = Decoder only
        P3 = Bottleck only
        P4 = Encoder + Decoder
        P5 = Encoder + Bottleneck
        P6 = Encodeur + Bottleneck + Decoder
        default = P6 as in original publication, the best configuration was SE blocks after encoder, bottleneck and decoder
        https://arxiv.org/pdf/1906.04649
        """
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage,
                          num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, dropout_op, 
                          dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, nonlin_first)
        if attention_placement_strategy != 'P2':
            self.encoder = PlainConvEncoderPE(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                                n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                                dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                                nonlin_first=nonlin_first, squeeze_excitation=squeeze_excitation,
                                                squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio)
        if attention_placement_strategy in ['P2', 'P4', 'P6']:
            self.decoder = UNetDecoderPE(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                        nonlin_first=nonlin_first, squeeze_excitation=squeeze_excitation,
                                        squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio)
        