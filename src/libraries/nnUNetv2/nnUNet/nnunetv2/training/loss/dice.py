from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc
    

class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool=True, smooth: float = 1., ddp: bool = True,
                  alpha: float = 0.3, beta: float = 0.7, reduction: str = 'mean'):
        """
        paper: https://arxiv.org/pdf/1706.05721.pdf
        """
        super(TverskyLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        tversky = (tp + self.smooth) / torch.clip(tp + self.alpha*fp + self.beta*fn + self.smooth, 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]

        if self.reduction == 'mean':
            return -tversky.mean()
        elif self.reduction == 'sum':
            return -tversky.sum()
        elif self.reduction == 'none':
            return -tversky
        else:
            raise ValueError(f"Reduction method %s is not supported." % self.reduction)
    

class DiceWithComplementLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1., ddp: bool = True,
                 loss_type: int = 1, complement: bool = False, weight: str = None, focal: str = None, gamma: float = 2):
        """
        Implement losses as in https://doi.org/10.1016/j.isprsjprs.2020.01.013
        :param apply_nonlin: function to apply to logits
        :param batch_dice: compute dice over all voxels in batch or mean of dice in each image of the batch
        :param loss_type: equation of the loss as in article :
            1. 2*sum_i(p_i*l_i)/(sum_i(p_i) + sum_i(l_i))
            2. 2*sum_i(p_i*l_i)/(sum_i(p_i**2 + l_i**2))
            3. sum_i(p_i*l_i)/(sum_i(p_i**2 + l_i**2) - sum_i(p_i*l_i))
        :param complement: if True for each class the loss L(p_i, l_i) is equal to 
        (L(p_i, l_i) + L(1-p_i, 1-l_i))/2
        :param smooth: ?
        :param weights: (None or str) weights for each label can be volume or volume_squared
        :param focal: (None or str) type of focal loss can be hm (homade) 
        or litterature (Wang et al. Focal Dice Loss and Image Dilation for Brain Tumor Segmentation)
        :param gamma: gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
        focus on hard misclassified example
        """
        super(DiceWithComplementLoss, self).__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.loss_type = loss_type
        self.complement = complement
        self.weight = weight
        self.focal = focal
        if gamma > 1:
            self.gamma = 1 / gamma
        else:
            self.gamma = gamma

        # Check args
        if self.loss_type not in [1,2,3]:
            raise ValueError('loss_type must be in [1,2,3] got %s' % self.loss_type)

        if self.weight not in [None, 'volume', 'volume_square']:
            raise ValueError('weight must be in [None, volume, volume_square] got %s' % self.weight)
        
        if self.focal not in [None, 'hm', 'homemade', 'litterature']:
            raise ValueError('focal must be in [None, hm, homemade, litterature] got %s' % self.focal)
        
    def forward(self, x, y, loss_mask=None):
        # This forward function is inspired by MemoryEfficientSoftDiceLoss forward function
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if self.batch_dice:
            axes = [0] + list(range(2, x.ndim))
        else:
            axes = list(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)
        
        if self.weight is None:
            weight = torch.ones(x.shape[1], dtype=torch.float, device=x.device)
        else:
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)
            # Weighting as in GDL weight computation, we use 1/V
            weight = torch.clip(1 / (sum_gt + 1e-6), 1e-6)# add some eps to prevent div by zero
            if self.weight == 'volume_square':
                weight = weight ** 2

        # TODO add ddp somewhere when computing loss
        if self.loss_type == 1:
            numerator = 2 * x * y_onehot.int()
            denominator = x + y_onehot.int()
        elif self.loss_type == 2:
            numerator = 2 * x * y_onehot.int()
            denominator = x ** 2 + y_onehot.int() ** 2
        elif self.loss_type == 3:
            numerator = x * y_onehot
            denominator = x ** 2 + y_onehot.int() ** 2 - x * y_onehot.int()
        else:
            raise ValueError('loss_type must be in [1,2,3] got %s' % self.loss_type)

        if loss_mask is not None:
            numerator = numerator * loss_mask
            denominator = denominator * loss_mask

        numerator = numerator.sum(axes)
        denominator = denominator.sum(axes)

        if self.ddp and self.batch_dice:
            numerator = AllGatherGrad.apply(numerator).sum(0)
            denominator = AllGatherGrad.apply(denominator).sum(0)
        
        dc = (weight * (numerator + self.smooth)) / (weight * torch.clip(denominator + self.smooth, 1e-8))
        
        if self.complement:
            if self.loss_type == 1:
                complement_numerator = 2 * (1-x) * (1-y_onehot.int())
                complement_denominator = (1-x) + (1-y_onehot.int())
            elif self.loss_type == 2:
                complement_numerator = 2 * (1-x) * (1-y_onehot.int())
                complement_denominator = (1-x) ** 2 + (1-y_onehot.int()) ** 2
            elif self.loss_type == 3:
                complement_numerator = (1-x) * (1-y_onehot.int())
                complement_denominator = (1-x) ** 2 + (1-y_onehot.int()) ** 2 - (1-x) * (1-y_onehot.int())
            else:
                raise ValueError('loss_type must be in [1,2,3] got %s' % self.loss_type)
            
            if loss_mask is not None:
                complement_numerator = complement_numerator * loss_mask
                complement_denominator = complement_denominator * loss_mask

            complement_numerator = complement_numerator.sum(axes)
            complement_denominator = complement_denominator.sum(axes)

            if self.ddp and self.batch_dice:
                complement_numerator = AllGatherGrad.apply(complement_numerator).sum(0)
                complement_denominator = AllGatherGrad.apply(complement_denominator).sum(0)
            
            complement_dc = (weight * (complement_numerator + self.smooth)) / (weight * torch.clip(complement_denominator + self.smooth, 1e-8))
            dc = (dc + complement_dc) / 2
    
        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        if self.focal is not None:
            # TODO : reflexion on effect depending on batch dice etc.
            # In original article the aim is to counter imbalance between classes and force focus on class harder to predict
            # This will be the behave here when multiclass + batch_dice because we have one dice per class as presented in article
            # If no batch dice : when binary problem focus on samples that are hard to predict but in multiclass it is not clear.
            if self.focal == 'litterature':
                # add min bound to avoid 0 which will results in nan
                dc = torch.pow(torch.clip(dc, min=1e-6, max=1), self.gamma)
            else:
                focal_weights = torch.pow(1 - dc, self.gamma)
                # TODO : maybe better way than return here ? Atm, I prefer to be sure that the return as the correct values
                return ((1 - dc) * focal_weights).mean()
        dc = dc.mean()
        return 1 - dc


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device, dtype=torch.bool)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (~y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (~y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here
        # benchmark whether tiling the mask would be faster (torch.tile). It probably is for large batch sizes
        # OK it barely makes a difference but the implementation above is a tiny bit faster + uses less vram
        # (using nnUNetv2_train 998 3d_fullres 0)
        # tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        # fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        # fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        # tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn


if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    dl_old = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    dl_new = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    res_old = dl_old(pred, ref)
    res_new = dl_new(pred, ref)
    print(res_old, res_new)
