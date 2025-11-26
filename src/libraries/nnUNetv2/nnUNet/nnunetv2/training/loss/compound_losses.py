import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss, DiceWithComplementLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn


class CompoundLoss(nn.Module):
    def __init__(self):
        super(CompoundLoss, self).__init__()
        # These will be set in child functions
        self.compound_loss = None
        self.compound_loss_link_name = None
        self.weights = None
        self.weights_method = None
        self.total_loss_list = None
        self.compound_loss_list = None

    def set_learned_weights(self):
        # TODO : maybe move each method in a separate class that inherits from CompoundLoss and have a separate Trainer
        if self.compound_loss is not None:
            if self.weights_method == 'Kendall2018':
                # We will learn uncertainty as loss weights :
                # https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
                # https://discuss.pytorch.org/t/how-to-learn-the-weights-between-two-losses/39681
                # Code Mously Diaw
                # WARNING : to use this loss must not take negative values or it will diverge
                # The learned parameters are the log variance for each loss term i.e. log(sigma**2)
                self.weights = torch.nn.ParameterDict({_: torch.nn.Parameter(torch.empty(1).uniform_(-2.0, 5.0)) for _ in self.compound_loss})
            elif self.weights_method == "Lee2021":
                # https://openaccess.thecvf.com/content/ICCV2021/papers/Lee_Learning_Multiple_Pixelwise_Tasks_Based_on_Loss_Scale_Balancing_ICCV_2021_paper.pdf
                # At start values for weights are set 1 / n_losses
                n_losses = len(self.compound_loss.keys())
                self.weights = {_: 1 / n_losses for _ in self.compound_loss}
                # For next steps we'll need to store the losses, this is a little overhead since nnUNet already store it but 
                # it's easier to have everything here to run the loss instead of modifying all the trainer
                self.total_loss_list = []
                self.compound_loss_list = {_: [] for _ in self.compound_loss}
            else:
                raise ValueError('weights method %s not implemented. Actually working: "Kendall2018", "Lee2021".' % self.weights_method)
        else:
            raise ValueError('losses need to be defined in order to create learned weights.')

    def check_keys(self):
        if self.compound_loss.keys() != self.weights.keys():
            raise ValueError('Keys of loss and weight dictionary are different.')

    def compute_total_loss(self):
        if self.weights_method == 'Kendall2018':
            # 0.5*(loss/sigma**2 + log(sigma**2))=loss/2sigma**2 + log(sigma) as we learn log(sigma**2)
            return sum([0.5 * (torch.exp(-self.weights[loss]) * self.compound_loss[loss] + self.weights[loss]) for loss in self.compound_loss])
        else:
            return sum([self.compound_loss[loss] * self.weights[loss] for loss in self.compound_loss if self.weights[loss] != 0])
    
    def update_weights(self, n_epoch=None, total_epoch=None):
        if self.weights_method == "Lee2021":
            # Weights updating depends on the period = n_epochs
            # TODO : determine iter differently as here we don't measure n_epochs
            if n_epoch is None:
                period = len(self.total_loss_list) + 1
            else:
                period = n_epoch + 1
            n_losses = len(self.compound_loss.keys())
            pi_k = {_: 1 / n_losses for _ in self.compound_loss}
            if period == 1:
                self.weights = {_: pi_k[_] for _ in self.compound_loss}
            elif period == 2:
                # In second period equalizing loss scale by adjusting each weight to be inversely proportional to the corresponding loss
                # However, we do not know the losses for the current period L^t_k yet
                # Assume similar ratios between components between past and current ier L^t/L^(t-1) = L^t_1/L^(t-1)_1 = L^t_1/L^(t-1)_1 = L^t_2/L^(t-1)_2 = ... = L^t_n/L^(t-1)_n
                # 
                self.weights = {_: pi_k[_] * self.total_loss_list[-1]/self.compound_loss_list[_][-1] for _ in self.compound_loss}
            elif period > 2:
                # Keep weights as in previous period
                w_k = {_: pi_k[_] * self.total_loss_list[-1]/self.compound_loss_list[_][-1] for _ in self.compound_loss}
                # Beta is increasing with the epoch. Paper uses values of 0.02 to 0.5 per epoch. We'll set it to 0.02 if we don't know total epoch else goal si to reach 10
                beta_step = 0.02 if total_epoch is None else 10/total_epoch
                beta = beta_step * (period - 2) # TODO : check that it is increasing correctly
                # Assign weights as a function of difficulty of the task
                d_k = {_: ((self.compound_loss_list[_][-1]/self.compound_loss_list[_][-2])/(self.total_loss_list[-1]/self.total_loss_list[-2]))**beta 
                       for _ in self.compound_loss}
                # α is a parameter to make the overall loss unchanged by the update of weights
                alpha_t = 1 / (sum([pi_k[_] * d_k[_] for _ in self.compound_loss]))
                # In contrast to to the paper we include alpha in the compound loss weight so it doesn't change the code too much
                self.weights = {_: alpha_t * w_k[_] * d_k[_]  for _ in self.compound_loss}

    def store_losses(self, average_epoch_total_loss=None, average_epoch_compound_loss=None):
        if self.compound_loss_list is not None and average_epoch_compound_loss is not None:
            for key in self.compound_loss:
                self.compound_loss_list[key].append(average_epoch_compound_loss[key])
        
        if self.total_loss_list is not None and average_epoch_total_loss is not None:
            self.total_loss_list.append(average_epoch_total_loss)
        

class DC_and_CE_loss(CompoundLoss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weights={'dice': 1, 'ce': 1}, ignore_label=None,
                 dice_class=SoftDiceLoss, ce_class=RobustCrossEntropyLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.ignore_label = ignore_label

        self.ce = ce_class(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.compound_loss_link_name = {'comp_loss_ce': type(self.ce).__name__,
                                         'comp_loss_dice': type(self.dc).__name__}

        self.compound_loss = {self.compound_loss_link_name['comp_loss_ce']: 0, 
                               self.compound_loss_link_name['comp_loss_dice']: 0}
        if isinstance(weights, str):
            self.weights_method = weights
            self.set_learned_weights()
        else:
            self.weights_method = 'fixed'
            self.weights = {self.compound_loss_link_name['comp_loss_ce']: weights['ce'],
                            self.compound_loss_link_name['comp_loss_dice']: weights['dice']}

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weights[self.compound_loss_link_name['comp_loss_dice']] != 0 else torch.zeros(1)
        self.compound_loss[self.compound_loss_link_name['comp_loss_dice']] = dc_loss
        if (self.weights_method == 'Kendall2018' or self.weights_method == 'Lee2021') and not isinstance(self.dc, DiceWithComplementLoss):
            # in nnunet implementation DiceLoss = -Dice but when weight are learned we must not have negative values
            # so DiceLoss = 1 - Dice. Computed here to avoid impact somewhere else in the code
            self.compound_loss[self.compound_loss_link_name['comp_loss_dice']] += 1
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weights[self.compound_loss_link_name['comp_loss_ce']] != 0 and (self.ignore_label is None or num_fg > 0) else torch.zeros(1)
        self.compound_loss[self.compound_loss_link_name['comp_loss_ce']] = ce_loss

        result = self.compute_total_loss()
        return result


class DC_and_BCE_loss(CompoundLoss):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weights={'dice': 1, 'bce': 1}, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss, bce_class=nn.BCEWithLogitsLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super().__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.use_ignore_label = use_ignore_label
        self.compound_loss_link_name = {}

        self.bce = bce_class(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)
        self.compound_loss_link_name = {'comp_loss_bce': type(self.bce).__name__,
                                         'comp_loss_dice': type(self.dc).__name__}

        self.compound_loss = {self.compound_loss_link_name['comp_loss_bce']: 0,
                                self.compound_loss_link_name['comp_loss_dice']: 0}
        if isinstance(weights, str):
            self.weights_method = self.weights
            self.set_learned_weights()
        else:
            self.weights_method = 'fixed'
            self.weights = {self.compound_loss_link_name['comp_loss_ce']: weights['ce'],
                            self.compound_loss_link_name['comp_loss_dice']: weights['dice']}   

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        self.compound_loss[self.compound_loss_link_name['comp_loss_dice']] = dc_loss \
            if self.weights[self.compound_loss_link_name['comp_loss_dice']] != 0 else torch.zeros(1)
        if (self.weights_method == 'Kendall2018' or self.weights_method == 'Lee2021') and not isinstance(self.dc, DiceWithComplementLoss):
            # in nnunet implementation DiceLoss = -Dice but when weight are learned we must not have negative values
            # so DiceLoss = 1 - Dice. Computed here to avoid impact somewhere else in the code
            self.compound_loss[self.compound_loss_link_name['comp_loss_dice']] += 1
        target_regions = target_regions.float()
        if mask is not None:
            bce_loss = (self.bce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8) \
                if self.weights[self.compound_loss_link_name['comp_loss_bce']] != 0 else torch.zeros(1)
        else:
            bce_loss = self.bce(net_output, target_regions) \
                if self.weights[self.compound_loss_link_name['comp_loss_bce']] != 0 else torch.zeros(1)
        self.compound_loss[self.compound_loss_link_name['comp_loss_bce']] = bce_loss

        result = self.compute_total_loss()
        return result


class DC_and_topk_loss(CompoundLoss):
    def __init__(self, soft_dice_kwargs, topk_kwargs, weights={'dice': 1, 'topk': 1}, ignore_label=None, 
                 dice_class=SoftDiceLoss, topk_class=TopKLoss):
        """
        Weights for TopK and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            topk_kwargs['ignore_index'] = ignore_label

        self.ignore_label = ignore_label
        self.compound_loss_link_name = {}

        self.topk = topk_class(**topk_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.compound_loss_link_name = {'comp_loss_topk': type(self.topk).__name__,
                                         'comp_loss_dice': type(self.dc).__name__}

        self.compound_loss = {self.compound_loss_link_name['comp_loss_topk']: 0,
                               self.compound_loss_link_name['comp_loss_dice']: 0}
        if isinstance(weights, str):
            self.weights_method = self.weights
            self.set_learned_weights()
        else:
            self.weights_method = 'fixed'
            self.weights = {self.compound_loss_link_name['comp_loss_ce']: weights['ce'],
                            self.compound_loss_link_name['comp_loss_dice']: weights['dice']}

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weights[self.compound_loss_link_name['comp_loss_dice']] != 0 else torch.zeros(1)
        self.compound_loss[self.compound_loss_link_name['comp_loss_dice']] = dc_loss
        if (self.weights_method == 'Kendall2018' or self.weights_method == 'Lee2021') and not isinstance(self.dc, DiceWithComplementLoss):
            # in nnunet implementation DiceLoss = -Dice but when weight are learned we must not have negative values
            # so DiceLoss = 1 - Dice. Computed here to avoid impact somewhere else in the code
            self.compound_loss[self.compound_loss_link_name['comp_loss_dice']] += 1
        topk_loss = self.topk(net_output, target) \
            if self.weights[self.compound_loss_link_name['comp_loss_topk']] != 0 and (self.ignore_label is None or num_fg > 0) else torch.zeros(1)
        self.compound_loss[self.compound_loss_link_name['comp_loss_topk']] = topk_loss

        result = self.compute_total_loss()
        return result


class Seg_and_Auxiliary_loss(CompoundLoss):
    """
    Unlike the other classes this class is implemented with losses already instatiated. 
    This is the simplest way since the seg part of the loss might be wrapped in another class for example for deep supervision or auxiliary classification
    """
    def __init__(self, seg_loss, aux_loss, weights={'seg': 1, 'aux': 1}):
        super().__init__()
        self.aux = aux_loss
        self.seg = seg_loss
        self.compound_loss_link_name = {'comp_loss_aux': type(self.aux.loss).__name__ if isinstance(self.aux, DeepSupervisionWrapper) else type(self.aux).__name__,
                                        'comp_loss_seg': type(self.seg.loss).__name__ if isinstance(self.aux, DeepSupervisionWrapper) else type(self.seg).__name__}

        self.compound_loss = {self.compound_loss_link_name['comp_loss_aux']: 0, 
                               self.compound_loss_link_name['comp_loss_seg']: 0}
        if isinstance(weights, str):
            self.weights_method = weights
            self.set_learned_weights()
        else:
            self.weights_method = 'fixed'
            self.weights = {self.compound_loss_link_name['comp_loss_aux']: weights['aux'],
                            self.compound_loss_link_name['comp_loss_seg']: weights['seg']}

    def forward(self, output_seg: torch.Tensor, target_seg: torch.Tensor, output_aux: torch.Tensor, target_aux: torch.Tensor=None):
        seg_loss = self.seg(output_seg, target_seg) if self.weights[self.compound_loss_link_name['comp_loss_seg']] != 0 else torch.zeros(1)
        self.compound_loss[self.compound_loss_link_name['comp_loss_seg']] = seg_loss
        if target_aux is not None:
            aux_loss = self.aux(output_aux, target_aux) if self.weights[self.compound_loss_link_name['comp_loss_aux']] != 0 else torch.zeros(1)
        else:
            aux_loss = self.aux(output_aux) if self.weights[self.compound_loss_link_name['comp_loss_aux']] != 0 else torch.zeros(1)
        self.compound_loss[self.compound_loss_link_name['comp_loss_aux']] = aux_loss

        result = self.compute_total_loss()
        return result