import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, DiceWithComplementLoss, TverskyLoss
from nnunetv2.training.loss.focal_loss import FocalLoss, FocalTversky_loss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerDiceLoss(nnUNetTrainer):
    def _build_loss(self):
        loss = MemoryEfficientSoftDiceLoss(**{'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': self.label_manager.has_regions, 'smooth': 1e-5, 'ddp': self.is_ddp},
                            apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerDiceCELoss_noSmooth(nnUNetTrainer):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp}, weights={'dice':1, 'bce': 1},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weights={'dice':1, 'ce': 1},
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)
            
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    


class nnUNetTrainerDiceFocalCELoss(nnUNetTrainer):
    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({'apply_nonlin': torch.sigmoid, 'alpha': 0.25, 'gamma': 2},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp}, weights={'dice':1, 'bce': 1},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss, bce_class=FocalLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                    {'apply_nonlin': softmax_helper_dim1, 'alpha': 0.25, 'gamma': 2},
                                    weights={'dice':1, 'ce': 1},
                                    ignore_label=self.label_manager.ignore_label, 
                                    dice_class=MemoryEfficientSoftDiceLoss, ce_class=FocalLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
    

class nnUNetTrainerDiceFocalCELoss_noSmooth(nnUNetTrainer):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({'apply_nonlin': torch.sigmoid, 'alpha': 0.25, 'gamma': 2},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                    weights={'dice':1, 'bce': 1},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss, bce_class=FocalLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, 
                                   {'apply_nonlin': softmax_helper_dim1, 'alpha': 0.25, 'gamma': 2},
                                    weights={'dice':1, 'ce': 1},
                                    ignore_label=self.label_manager.ignore_label,
                                    dice_class=MemoryEfficientSoftDiceLoss, ce_class=FocalLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    

class nnUNetTrainer_TrainableComposite(nnUNetTrainer):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp}, weights="Kendall2018",
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weights="Kendall2018",
                                   ignore_label=self.label_manager.ignore_label,
                                   dice_class=MemoryEfficientSoftDiceLoss)
            
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainer_noSmooth_TrainableComposite(nnUNetTrainer):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp}, weights="Kendall2018",
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weights="Kendall2018",
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)
            
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    

class nnUNetTrainer_noSmooth_LossScaleBalancing(nnUNetTrainer):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp}, weights="Lee2021",
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weights="Lee2021",
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)
            
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerDice1ComplementCELoss_noSmooth(nnUNetTrainer):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp, 
                                    'loss_type': 1, 'complement': True},
                                    weights={'dice': 1, 'bce': 1},
                                    use_ignore_label=self.label_manager.ignore_label is not None,
                                    dice_class=DiceWithComplementLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp,
                                   'loss_type': 1, 'complement': True}, {},
                                    weights={'dice': 1, 'ce': 1},
                                    ignore_label=self.label_manager.ignore_label,
                                    dice_class=DiceWithComplementLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    

class nnUNetTrainerDice3ComplementCELoss_noSmooth(nnUNetTrainer):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp, 
                                    'loss_type': 3, 'complement': True},
                                    weights={'dice': 1, 'bce': 1},
                                    use_ignore_label=self.label_manager.ignore_label is not None,
                                    dice_class=DiceWithComplementLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp,
                                   'loss_type': 1, 'complement': True}, {},
                                    weights={'dice': 1, 'ce': 1},
                                    ignore_label=self.label_manager.ignore_label,
                                    dice_class=DiceWithComplementLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerFocalDiceCELoss_noSmooth(nnUNetTrainer):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp, 
                                    'loss_type': 1, 'complement': False, 'focal': 'homemade'},
                                    weights={'dice': 1, 'bce': 1},
                                    use_ignore_label=self.label_manager.ignore_label is not None,
                                    dice_class=DiceWithComplementLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp,
                                   'loss_type': 1, 'complement': False, 'focal': 'homemade'}, {},
                                    weights={'dice': 1, 'ce': 1},
                                    ignore_label=self.label_manager.ignore_label,
                                    dice_class=DiceWithComplementLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    
    
class nnUNetTrainerFocalDiceFocalCELoss_noSmooth(nnUNetTrainer):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({'apply_nonlin': torch.sigmoid, 'alpha': 0.25, 'gamma': 2},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp, 
                                    'loss_type': 1, 'complement': False, 'focal': 'homemade'},
                                    weights={'dice': 1, 'bce': 1},
                                    use_ignore_label=self.label_manager.ignore_label is not None,
                                    dice_class=DiceWithComplementLoss, bce_class=FocalLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp,
                                   'loss_type': 1, 'complement': False, 'focal': 'homemade'}, 
                                  {'apply_nonlin': softmax_helper_dim1, 'alpha': 0.25, 'gamma': 2},
                                    weights={'dice': 1, 'ce': 1},
                                    ignore_label=self.label_manager.ignore_label,
                                    dice_class=DiceWithComplementLoss, ce_class=FocalLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    
    
class nnUNetTrainerFocalDiceLitCELoss_noSmooth(nnUNetTrainer):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp, 
                                    'loss_type': 1, 'complement': False, 'focal': 'litterature'},
                                    weights={'dice': 1, 'bce': 1},
                                    use_ignore_label=self.label_manager.ignore_label is not None,
                                    dice_class=DiceWithComplementLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp,
                                   'loss_type': 1, 'complement': False, 'focal': 'litterature'}, {},
                                    weights={'dice': 1, 'ce': 1},
                                    ignore_label=self.label_manager.ignore_label,
                                    dice_class=DiceWithComplementLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    
    
class nnUNetTrainerFocalDiceLitFocalCELoss_noSmooth(nnUNetTrainer):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({'apply_nonlin': torch.sigmoid, 'alpha': 0.25, 'gamma': 2},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp, 
                                    'loss_type': 1, 'complement': False, 'focal': 'litterature'},
                                    weights={'dice': 1, 'bce': 1},
                                    use_ignore_label=self.label_manager.ignore_label is not None,
                                    dice_class=DiceWithComplementLoss, bce_class=FocalLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp,
                                   'loss_type': 1, 'complement': False, 'focal': 'litterature'}, 
                                  {'apply_nonlin': softmax_helper_dim1, 'alpha': 0.25, 'gamma': 2},
                                    weights={'dice': 1, 'ce': 1},
                                    ignore_label=self.label_manager.ignore_label,
                                    dice_class=DiceWithComplementLoss, ce_class=FocalLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    

class nnUNetTrainerTverskyCELoss_noSmooth(nnUNetTrainer):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp, 
                                    'alpha': 0.3, 'beta': 0.7},
                                    weights={'dice': 1, 'bce': 1},
                                    use_ignore_label=self.label_manager.ignore_label is not None,
                                    dice_class=TverskyLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp,
                                   'alpha': 0.3, 'beta': 0.7}, {},
                                    weights={'dice': 1, 'ce': 1},
                                    ignore_label=self.label_manager.ignore_label,
                                    dice_class=TverskyLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    
    
class nnUNetTrainerFocalTverskyCELoss_noSmooth(nnUNetTrainer):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp, 
                                    'alpha': 0.3, 'beta': 0.7, 'gamma': 0.75},
                                    weights={'dice': 1, 'bce': 1},
                                    use_ignore_label=self.label_manager.ignore_label is not None,
                                    dice_class=FocalTversky_loss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp,
                                   'alpha': 0.3, 'beta': 0.7, 'gamma': 0.75}, {},
                                    weights={'dice': 1, 'ce': 1},
                                    ignore_label=self.label_manager.ignore_label,
                                    dice_class=FocalTversky_loss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    
    
class nnUNetTrainerTverskyFocalCELoss_noSmooth(nnUNetTrainer):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({'apply_nonlin': torch.sigmoid, 'alpha': 0.25, 'gamma': 2},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp, 
                                    'alpha': 0.3, 'beta': 0.7},
                                    weights={'dice': 1, 'bce': 1},
                                    use_ignore_label=self.label_manager.ignore_label is not None,
                                    dice_class=TverskyLoss, bce_class=FocalLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp,
                                   'alpha': 0.3, 'beta': 0.7}, 
                                  {'apply_nonlin': softmax_helper_dim1, 'alpha': 0.25, 'gamma': 2},
                                    weights={'dice': 1, 'ce': 1},
                                    ignore_label=self.label_manager.ignore_label,
                                    dice_class=TverskyLoss, ce_class=FocalLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    
    
class nnUNetTrainerFocalTverskyFocalCELoss_noSmooth(nnUNetTrainer):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({'apply_nonlin': torch.sigmoid, 'alpha': 0.25, 'gamma': 2},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp, 
                                    'alpha': 0.3, 'beta': 0.7, 'gamma': 0.75},
                                    weights={'dice': 1, 'bce': 1},
                                    use_ignore_label=self.label_manager.ignore_label is not None,
                                    dice_class=FocalTversky_loss, bce_class=FocalLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp,
                                   'alpha': 0.3, 'beta': 0.7, 'gamma': 0.75}, 
                                  {'apply_nonlin': softmax_helper_dim1, 'alpha': 0.25, 'gamma': 2},
                                    weights={'dice': 1, 'ce': 1},
                                    ignore_label=self.label_manager.ignore_label,
                                    dice_class=FocalTversky_loss, ce_class=FocalLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss