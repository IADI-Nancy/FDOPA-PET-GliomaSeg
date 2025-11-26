from time import time
from typing import Tuple, Union, List

import numpy as np
import sklearn.metrics as skm
import torch
from torch import autocast
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2DAuxiliaryRegressor
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3DAuxiliaryRegressor
from nnunetv2.training.loss.compound_losses import Seg_and_Auxiliary_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from torch import distributed as dist
from nnunetv2.utilities.collate_outputs import collate_outputs


class nnUNetTrainerAuxiliaryRegressor(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.logger.my_fantastic_logging['auxiliary_mse'] = list()

    def _build_loss(self):
        # Manage Segmentation loss as usual
        seg_loss = super()._build_loss()
        aux_classif_loss = torch.nn.MSELoss()
        total_loss = Seg_and_Auxiliary_loss(seg_loss, aux_classif_loss, weights={'seg': 1, 'aux': 0.1})
        return total_loss
    
    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?

        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2DAuxiliaryRegressor(dataset_tr, self.batch_size,
                                                          initial_patch_size,
                                                          self.configuration_manager.patch_size,
                                                          self.label_manager,
                                                          oversample_foreground_percent=self.oversample_foreground_percent,
                                                          sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
            dl_val = nnUNetDataLoader2DAuxiliaryRegressor(dataset_val, self.batch_size,
                                                           self.configuration_manager.patch_size,
                                                           self.configuration_manager.patch_size,
                                                           self.label_manager,
                                                           oversample_foreground_percent=self.oversample_foreground_percent,
                                                           sampling_probabilities=None, pad_sides=None, transforms=val_transforms)
        else:
            dl_tr = nnUNetDataLoader3DAuxiliaryRegressor(dataset_tr, self.batch_size,
                                                          initial_patch_size,
                                                          self.configuration_manager.patch_size,
                                                          self.label_manager,
                                                          oversample_foreground_percent=self.oversample_foreground_percent,
                                                          sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
            dl_val = nnUNetDataLoader3DAuxiliaryRegressor(dataset_val, self.batch_size,
                                                           self.configuration_manager.patch_size,
                                                           self.configuration_manager.patch_size,
                                                           self.label_manager,
                                                           oversample_foreground_percent=self.oversample_foreground_percent,
                                                           sampling_probabilities=None, pad_sides=None, transforms=val_transforms)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val
    
    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        if isinstance(self.loss, DeepSupervisionWrapper):
            if hasattr(self.loss.loss, 'update_weights') and callable(self.loss.loss.update_weights):
                self.loss.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)
        else:
            if hasattr(self.loss, 'update_weights') and callable(self.loss.update_weights):
                self.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target_seg = batch['target']
        target_aux = batch['target_aux']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target_seg, list):
            target_seg = [i.to(self.device, non_blocking=True) for i in target_seg]
        else:
            target_seg = target_seg.to(self.device, non_blocking=True)
        target_aux = target_aux.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output_seg, output_aux = self.network(data)
            # del data
            l = self.loss(output_seg, target_seg, output_aux, target_aux)
            if hasattr(self.loss, 'compound_loss'):
                comp_l = self.loss.compound_loss
            else: 
                comp_l = {}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        out = {'loss': l.detach().cpu().numpy()}
        out.update({'comp_loss_%s' % l_i: comp_l[l_i].detach().cpu().numpy() for l_i in comp_l})
        return out
    
    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
            comp_loss_here = {}
            for key in outputs:
                if 'comp_loss' in key:
                    comp_loss_tr = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(comp_loss_tr, outputs[key])
                    comp_loss_here[key] = np.vstack(comp_loss_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])
            comp_loss_here = {key.split('comp_loss_')[1]: np.mean(outputs[key]) for key in outputs if 'comp_loss' in key}

        self.logger.log('train_losses', loss_here, self.current_epoch)
        self.logger.log('train_compound_losses', comp_loss_here, self.current_epoch)

        if isinstance(self.loss, DeepSupervisionWrapper):
            if hasattr(self.loss.loss, 'store_losses') and callable(self.loss.loss.store_losses):
                self.loss.loss.store_losses(average_epoch_total_loss=loss_here, average_epoch_compound_loss=comp_loss_here)
        else:
            if hasattr(self.loss, 'store_losses') and callable(self.loss.store_losses):
                self.loss.store_losses(average_epoch_total_loss=loss_here, average_epoch_compound_loss=comp_loss_here)
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target_seg = batch['target']
        target_aux = batch['target_aux']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target_seg, list):
            target_seg = [i.to(self.device, non_blocking=True) for i in target_seg]
        else:
            target_seg = target_seg.to(self.device, non_blocking=True)
        target_aux = target_aux.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output_seg, output_aux = self.network(data)
            del data
            l = self.loss(output_seg, target_seg, output_aux, target_aux)
            if hasattr(self.loss, 'compound_loss'):
                comp_l = self.loss.compound_loss
            else: 
                comp_l = {}

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output_seg = output_seg[0]
            target_seg = target_seg[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output_seg.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output_seg) > 0.5).long()
        else:
            # no need for softmax
            current_output_seg = output_seg.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output_seg.shape, device=output_seg.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, current_output_seg, 1)
            del current_output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask_seg = (target_seg != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target_seg[target_seg == self.label_manager.ignore_label] = 0
            else:
                if target_seg.dtype == torch.bool:
                    mask_seg = ~target_seg[:, -1:]
                else:
                    mask_seg = 1 - target_seg[:, -1:]
                # CAREFUL that you don't rely on target_seg after this line!
                target_seg = target_seg[:, :-1]
        else:
            mask_seg = None

        tp_seg, fp_seg, fn_seg, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_seg, axes=axes, mask=mask_seg)

        tp_hard = tp_seg.detach().cpu().numpy()
        fp_hard = fp_seg.detach().cpu().numpy()
        fn_hard = fn_seg.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
        
        mse = skm.mean_squared_error(target_aux.detach().cpu().numpy(), output_aux.detach().cpu().numpy())

        out = {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard,
               'mse_aux': mse}
        out.update({'comp_loss_%s' % l_i: comp_l[l_i].detach().cpu().numpy() for l_i in comp_l})
        return out
    
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp_seg = np.sum(outputs_collated['tp_hard'], 0)
        fp_seg = np.sum(outputs_collated['fp_hard'], 0)
        fn_seg = np.sum(outputs_collated['fn_hard'], 0)
        mse_aux = np.mean(outputs_collated['mse_aux'])

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps_seg = [None for _ in range(world_size)]
            dist.all_gather_object(tps_seg, tp_seg)
            tp_seg = np.vstack([i[None] for i in tps_seg]).sum(0)

            fps_seg = [None for _ in range(world_size)]
            dist.all_gather_object(fps_seg, fp_seg)
            fp_seg = np.vstack([i[None] for i in fps_seg]).sum(0)

            fns_seg = [None for _ in range(world_size)]
            dist.all_gather_object(fns_seg, fn_seg)
            fn_seg = np.vstack([i[None] for i in fns_seg]).sum(0)

            mses_aux = [None for _ in range(world_size)]
            dist.all_gather_object(mses_aux, mse_aux)
            mse_aux = np.vstack([i[None] for i in mses_aux]).mean()

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
            comp_loss_here = {}
            for key in outputs_collated:
                if 'comp_loss' in key:
                    comp_loss_val = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(comp_loss_val, outputs_collated[key])
                    comp_loss_here[key] = np.vstack(comp_loss_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])
            comp_loss_here = {key.split('comp_loss_')[1]: np.mean(outputs_collated[key]) for key in outputs_collated if 'comp_loss' in key}

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp_seg, fp_seg, fn_seg)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('auxiliary_mse', mse_aux, self.current_epoch)  
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('val_compound_losses', comp_loss_here, self.current_epoch)
    
    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file("compound_train_loss : %s" % ', '.join('%s=%.4f' % (key, value) for key, value in 
                                                                      self.logger.my_fantastic_logging['train_compound_losses'][-1].items()))
        if isinstance(self.loss, DeepSupervisionWrapper):
            if hasattr(self.loss.loss, 'weights'):
                self.print_to_log_file("compound_loss_weights : %s" % ', '.join('%s=%.4f' % (key, value) for key, value in 
                                                                            self.loss.loss.weights.items()))
        else:
            if hasattr(self.loss, 'weights'):
                self.print_to_log_file("compound_loss_weights : %s" % ', '.join('%s=%.4f' % (key, value) for key, value in 
                                                                            self.loss.weights.items()))
        if list(self.loss.parameters()):
            self.print_to_log_file("composite loss trained parameters : %s" % {name: param.data for name, param in self.loss.named_parameters()})
        self.print_to_log_file("compound_val_loss : %s" % ', '.join('%s=%.4f' % (key, value) for key, value in 
                                                                      self.logger.my_fantastic_logging['val_compound_losses'][-1].items()))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file('Auxiliary regressor MSE', np.round(self.logger.my_fantastic_logging['auxiliary_mse'][-1], decimals=4))
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    # TODO : at the moment we deactivtate the prediction of the auxiliary branch at inference time
    # If we want these predictions anyway we need to modify : 
    # 1. In nnUNetTrainerAuxiliaryRegressor : modify perform_actual_validation
    # 2. Create a new class inherating from nnUNetPredictor where we modify predict_sliding_window_return_logits (maybe others ? for inference done outside the trainer for example when calling nnunetv2_predict)
    # 3. modify nnunetv2.evaluation.evaluate_predictions.compute_metrics so that it compute mse etc for the auxiliary regressor
    # Something else? 
    
class nnUNetTrainerAuxiliaryRegressor_higher_weight(nnUNetTrainerAuxiliaryRegressor):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
    
    def _build_loss(self):
        # Manage Segmentation loss as usual
        seg_loss = super(nnUNetTrainerAuxiliaryRegressor, self)._build_loss()
        aux_classif_loss = torch.nn.MSELoss()
        total_loss = Seg_and_Auxiliary_loss(seg_loss, aux_classif_loss, weights={'seg': 1, 'aux': 0.5})
        return total_loss