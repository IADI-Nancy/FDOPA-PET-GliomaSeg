#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerAuxiliaryClassifier import nnUNetTrainerAuxiliaryClassifier, nnUNetTrainerAuxiliaryClassifier_higher_weight
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerAuxiliaryRegressor import nnUNetTrainerAuxiliaryRegressor, nnUNetTrainerAuxiliaryRegressor_higher_weight
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerContrastiveSupervision import nnUNetTrainerContrastiveSupervision, nnUNetTrainerContrastiveSupervision_higher_weight
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import *
from nnunetv2.training.lr_scheduler.ranger22_scheduler import Ranger22Scheduler
from nnunetv2.training.optimizer.ranger22_optimizer import Ranger22
from torch import autocast
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerCELoss import nnUNetTrainerrCEL1Loss

class nnUNetTrainerRanger(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3

    def configure_optimizers(self):
        optimizer = Ranger22(list(self.network.parameters())  + list(self.loss.parameters()), self.initial_lr, weight_decay=self.weight_decay, amsgrad=True)
        #TODO : test amsgrad T/F, weight_decay nnUNet defaut/Ranger default
        # 1st: 500 epochs, asmgrad=True, weight_decay=nnUNet default, initial_lr = 5e-3
        # 2nd: 1000 epochs
        # 3rd: 500 epochs, asmgrad=False, weight_decay=Ranger default, initial_lr = 1e-3
        # Scheduler with decrease starting at 50% epoch
        lr_scheduler = Ranger22Scheduler(optimizer, self.num_iterations_per_epoch, self.num_epochs)
        return optimizer, lr_scheduler
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)
            if hasattr(self.loss, 'compound_loss'):
                comp_l = self.loss.compound_loss
            else: 
                comp_l = {}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.lr_scheduler.step()
        else:
            l.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
        out = {'loss': l.detach().cpu().numpy()}
        out.update({'comp_loss_%s' % l_i: comp_l[l_i].detach().cpu().numpy() for l_i in comp_l})
        return out
    
    def on_train_epoch_start(self):
        self.network.train()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=7)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        if isinstance(self.loss, DeepSupervisionWrapper):
            if hasattr(self.loss.loss, 'update_weights') and callable(self.loss.loss.update_weights):
                self.loss.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)
        else:
            if hasattr(self.loss, 'update_weights') and callable(self.loss.update_weights):
                self.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)


class nnUNetTrainerRanger_250epochs(nnUNetTrainerRanger):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerRanger_noSmooth(nnUNetTrainerRanger, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)


class nnUNetTrainerRanger_noSmooth_250epochs(nnUNetTrainerRanger, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerRanger_noSmooth2(nnUNetTrainerRanger, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 5e-3
        self.weight_decay = 1e-4

    def configure_optimizers(self):
        optimizer = Ranger22(list(self.network.parameters())  + list(self.loss.parameters()), self.initial_lr, weight_decay=self.weight_decay, grad_norm=False)
        lr_scheduler = Ranger22Scheduler(optimizer, self.num_iterations_per_epoch, self.num_epochs)
        return optimizer, lr_scheduler
    

class nnUNetTrainerRanger_noSmooth3(nnUNetTrainerRanger_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
    def configure_optimizers(self):
        optimizer = Ranger22(list(self.network.parameters())  + list(self.loss.parameters()), self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = Ranger22Scheduler(optimizer, self.num_iterations_per_epoch, self.num_epochs, warmdown_start_size=0.5)
        return optimizer, lr_scheduler
    

class nnUNetTrainerRanger_TrainableComposite_250epochs(nnUNetTrainerRanger, nnUNetTrainer_TrainableComposite):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerRanger_noSmooth_TrainableComposite_250epochs(nnUNetTrainerRanger, nnUNetTrainer_noSmooth_TrainableComposite):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerRanger_noSmooth_LossScaleBalancing_250epochs(nnUNetTrainerRanger, nnUNetTrainer_noSmooth_LossScaleBalancing):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerRanger_noSmooth_DiceFocalCE_250epochs(nnUNetTrainerRanger, nnUNetTrainerDiceFocalCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerRanger_noSmooth_Dice1ComplementCE_250epochs(nnUNetTrainerRanger, nnUNetTrainerDice1ComplementCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerRanger_noSmooth_Dice3ComplementCE_250epochs(nnUNetTrainerRanger, nnUNetTrainerDice3ComplementCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerRanger_rCEL1_250epochs(nnUNetTrainerRanger, nnUNetTrainerrCEL1Loss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerRanger_noSmooth_FocalDiceCE_250epochs(nnUNetTrainerRanger, nnUNetTrainerFocalDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
        

class nnUNetTrainerRanger_noSmooth_FocalDiceFocalCE_250epochs(nnUNetTrainerRanger, nnUNetTrainerFocalDiceFocalCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerRanger_noSmooth_FocalDiceLitCE_250epochs(nnUNetTrainerRanger, nnUNetTrainerFocalDiceLitCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
        

class nnUNetTrainerRanger_noSmooth_FocalDiceLitFocalCE_250epochs(nnUNetTrainerRanger, nnUNetTrainerFocalDiceLitFocalCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
        
    
class nnUNetTrainerRanger_noSmooth_TverskyCE_250epochs(nnUNetTrainerRanger, nnUNetTrainerTverskyCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
        

class nnUNetTrainerRanger_noSmooth_FocalTverskyFocalCE_250epochs(nnUNetTrainerRanger, nnUNetTrainerFocalTverskyFocalCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
        

class nnUNetTrainerRanger_noSmooth_TverskyFocalCE_250epochs(nnUNetTrainerRanger, nnUNetTrainerTverskyFocalCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
        

class nnUNetTrainerRanger_noSmooth_FocalTverskyCE_250epochs(nnUNetTrainerRanger, nnUNetTrainerFocalTverskyCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250
        

class nnUNetTrainerRangerAuxiliaryClassifier(nnUNetTrainerAuxiliaryClassifier):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3

    def configure_optimizers(self):
        optimizer = Ranger22(list(self.network.parameters())  + list(self.loss.parameters()), self.initial_lr, weight_decay=self.weight_decay, amsgrad=True)
        #TODO : test amsgrad T/F, weight_decay nnUNet defaut/Ranger default
        # 1st: 500 epochs, asmgrad=True, weight_decay=nnUNet default, initial_lr = 5e-3
        # 2nd: 1000 epochs
        # 3rd: 500 epochs, asmgrad=False, weight_decay=Ranger default, initial_lr = 1e-3
        # Scheduler with decrease starting at 50% epoch
        lr_scheduler = Ranger22Scheduler(optimizer, self.num_iterations_per_epoch, self.num_epochs)
        return optimizer, lr_scheduler
    
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
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.lr_scheduler.step()
        else:
            l.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
        out = {'loss': l.detach().cpu().numpy()}
        out.update({'comp_loss_%s' % l_i: comp_l[l_i].detach().cpu().numpy() for l_i in comp_l})
        return out
    
    def on_train_epoch_start(self):
        self.network.train()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=7)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        if isinstance(self.loss, DeepSupervisionWrapper):
            if hasattr(self.loss.loss, 'update_weights') and callable(self.loss.loss.update_weights):
                self.loss.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)
        else:
            if hasattr(self.loss, 'update_weights') and callable(self.loss.update_weights):
                self.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)


class nnUNetTrainerRangerAuxiliaryClassifier_250epochs(nnUNetTrainerRangerAuxiliaryClassifier):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerRangerAuxiliaryClassifier_noSmooth(nnUNetTrainerRangerAuxiliaryClassifier, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)


class nnUNetTrainerRangerAuxiliaryClassifier_noSmooth_250epochs(nnUNetTrainerRangerAuxiliaryClassifier_250epochs, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        
class nnUNetTrainerRangerAuxiliaryClassifier_higher_weight(nnUNetTrainerAuxiliaryClassifier_higher_weight):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3

    def configure_optimizers(self):
        optimizer = Ranger22(list(self.network.parameters())  + list(self.loss.parameters()), self.initial_lr, weight_decay=self.weight_decay, amsgrad=True)
        #TODO : test amsgrad T/F, weight_decay nnUNet defaut/Ranger default
        # 1st: 500 epochs, asmgrad=True, weight_decay=nnUNet default, initial_lr = 5e-3
        # 2nd: 1000 epochs
        # 3rd: 500 epochs, asmgrad=False, weight_decay=Ranger default, initial_lr = 1e-3
        # Scheduler with decrease starting at 50% epoch
        lr_scheduler = Ranger22Scheduler(optimizer, self.num_iterations_per_epoch, self.num_epochs)
        return optimizer, lr_scheduler
    
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
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.lr_scheduler.step()
        else:
            l.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
        out = {'loss': l.detach().cpu().numpy()}
        out.update({'comp_loss_%s' % l_i: comp_l[l_i].detach().cpu().numpy() for l_i in comp_l})
        return out
    
    def on_train_epoch_start(self):
        self.network.train()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=7)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        if isinstance(self.loss, DeepSupervisionWrapper):
            if hasattr(self.loss.loss, 'update_weights') and callable(self.loss.loss.update_weights):
                self.loss.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)
        else:
            if hasattr(self.loss, 'update_weights') and callable(self.loss.update_weights):
                self.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)


class nnUNetTrainerRangerAuxiliaryClassifier_higher_weight_250epochs(nnUNetTrainerRangerAuxiliaryClassifier_higher_weight):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerRangerAuxiliaryClassifier_higher_weight_noSmooth(nnUNetTrainerRangerAuxiliaryClassifier_higher_weight, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)


class nnUNetTrainerRangerAuxiliaryClassifier_higher_weight_noSmooth_250epochs(nnUNetTrainerRangerAuxiliaryClassifier_higher_weight_250epochs, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)


class nnUNetTrainerRangerAuxiliaryRegressor(nnUNetTrainerAuxiliaryRegressor):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3

    def configure_optimizers(self):
        optimizer = Ranger22(list(self.network.parameters())  + list(self.loss.parameters()), self.initial_lr, weight_decay=self.weight_decay, amsgrad=True)
        #TODO : test amsgrad T/F, weight_decay nnUNet defaut/Ranger default
        # 1st: 500 epochs, asmgrad=True, weight_decay=nnUNet default, initial_lr = 5e-3
        # 2nd: 1000 epochs
        # 3rd: 500 epochs, asmgrad=False, weight_decay=Ranger default, initial_lr = 1e-3
        # Scheduler with decrease starting at 50% epoch
        lr_scheduler = Ranger22Scheduler(optimizer, self.num_iterations_per_epoch, self.num_epochs)
        return optimizer, lr_scheduler
    
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
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.lr_scheduler.step()
        else:
            l.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
        out = {'loss': l.detach().cpu().numpy()}
        out.update({'comp_loss_%s' % l_i: comp_l[l_i].detach().cpu().numpy() for l_i in comp_l})
        return out
    
    def on_train_epoch_start(self):
        self.network.train()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=7)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        if isinstance(self.loss, DeepSupervisionWrapper):
            if hasattr(self.loss.loss, 'update_weights') and callable(self.loss.loss.update_weights):
                self.loss.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)
        else:
            if hasattr(self.loss, 'update_weights') and callable(self.loss.update_weights):
                self.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)


class nnUNetTrainerRangerAuxiliaryRegressor_250epochs(nnUNetTrainerRangerAuxiliaryRegressor):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerRangerAuxiliaryRegressor_noSmooth(nnUNetTrainerRangerAuxiliaryRegressor, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)


class nnUNetTrainerRangerAuxiliaryRegressor_noSmooth_250epochs(nnUNetTrainerRangerAuxiliaryRegressor_250epochs, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        
class nnUNetTrainerRangerAuxiliaryRegressor_higher_weight(nnUNetTrainerAuxiliaryRegressor_higher_weight):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3

    def configure_optimizers(self):
        optimizer = Ranger22(list(self.network.parameters())  + list(self.loss.parameters()), self.initial_lr, weight_decay=self.weight_decay, amsgrad=True)
        #TODO : test amsgrad T/F, weight_decay nnUNet defaut/Ranger default
        # 1st: 500 epochs, asmgrad=True, weight_decay=nnUNet default, initial_lr = 5e-3
        # 2nd: 1000 epochs
        # 3rd: 500 epochs, asmgrad=False, weight_decay=Ranger default, initial_lr = 1e-3
        # Scheduler with decrease starting at 50% epoch
        lr_scheduler = Ranger22Scheduler(optimizer, self.num_iterations_per_epoch, self.num_epochs)
        return optimizer, lr_scheduler
    
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
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.lr_scheduler.step()
        else:
            l.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
        out = {'loss': l.detach().cpu().numpy()}
        out.update({'comp_loss_%s' % l_i: comp_l[l_i].detach().cpu().numpy() for l_i in comp_l})
        return out
    
    def on_train_epoch_start(self):
        self.network.train()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=7)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        if isinstance(self.loss, DeepSupervisionWrapper):
            if hasattr(self.loss.loss, 'update_weights') and callable(self.loss.loss.update_weights):
                self.loss.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)
        else:
            if hasattr(self.loss, 'update_weights') and callable(self.loss.update_weights):
                self.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)


class nnUNetTrainerRangerAuxiliaryRegressor_higher_weight_250epochs(nnUNetTrainerRangerAuxiliaryRegressor_higher_weight):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerRangerAuxiliaryRegressor_higher_weight_noSmooth(nnUNetTrainerRangerAuxiliaryRegressor_higher_weight, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)


class nnUNetTrainerRangerAuxiliaryRegressor_higher_weight_noSmooth_250epochs(nnUNetTrainerRangerAuxiliaryRegressor_higher_weight_250epochs, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)


class nnUNetTrainerRangerContrastiveSupervision(nnUNetTrainerContrastiveSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3

    def configure_optimizers(self):
        optimizer = Ranger22(list(self.network.parameters())  + list(self.loss.parameters()), self.initial_lr, weight_decay=self.weight_decay, amsgrad=True)
        #TODO : test amsgrad T/F, weight_decay nnUNet defaut/Ranger default
        # 1st: 500 epochs, asmgrad=True, weight_decay=nnUNet default, initial_lr = 5e-3
        # 2nd: 1000 epochs
        # 3rd: 500 epochs, asmgrad=False, weight_decay=Ranger default, initial_lr = 1e-3
        # Scheduler with decrease starting at 50% epoch
        lr_scheduler = Ranger22Scheduler(optimizer, self.num_iterations_per_epoch, self.num_epochs)
        return optimizer, lr_scheduler
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        data2 = batch['data_contrastive']
        target_seg = batch['target']

        data = data.to(self.device, non_blocking=True)
        data2 = data2.to(self.device, non_blocking=True)
        if isinstance(target_seg, list):
            target_seg = [i.to(self.device, non_blocking=True) for i in target_seg]
        else:
            target_seg = target_seg.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output_seg, output_aux = self.network(data, data2)
            # del data
            l = self.loss(output_seg, target_seg, output_aux)
            if hasattr(self.loss, 'compound_loss'):
                comp_l = self.loss.compound_loss
            else: 
                comp_l = {}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.lr_scheduler.step()
        else:
            l.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
        out = {'loss': l.detach().cpu().numpy()}
        out.update({'comp_loss_%s' % l_i: comp_l[l_i].detach().cpu().numpy() for l_i in comp_l})
        return out
    
    def on_train_epoch_start(self):
        self.network.train()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=7)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        if isinstance(self.loss, DeepSupervisionWrapper):
            if hasattr(self.loss.loss, 'update_weights') and callable(self.loss.loss.update_weights):
                self.loss.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)
        else:
            if hasattr(self.loss, 'update_weights') and callable(self.loss.update_weights):
                self.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)


class nnUNetTrainerRangerContrastiveSupervision_250epochs(nnUNetTrainerRangerContrastiveSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerRangerContrastiveSupervision_noSmooth(nnUNetTrainerRangerContrastiveSupervision, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)


class nnUNetTrainerRangerContrastiveSupervision_noSmooth_250epochs(nnUNetTrainerRangerContrastiveSupervision_250epochs, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        
class nnUNetTrainerRangerContrastiveSupervision_higher_weight(nnUNetTrainerContrastiveSupervision_higher_weight):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3

    def configure_optimizers(self):
        optimizer = Ranger22(list(self.network.parameters())  + list(self.loss.parameters()), self.initial_lr, weight_decay=self.weight_decay, amsgrad=True)
        #TODO : test amsgrad T/F, weight_decay nnUNet defaut/Ranger default
        # 1st: 500 epochs, asmgrad=True, weight_decay=nnUNet default, initial_lr = 5e-3
        # 2nd: 1000 epochs
        # 3rd: 500 epochs, asmgrad=False, weight_decay=Ranger default, initial_lr = 1e-3
        # Scheduler with decrease starting at 50% epoch
        lr_scheduler = Ranger22Scheduler(optimizer, self.num_iterations_per_epoch, self.num_epochs)
        return optimizer, lr_scheduler
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        data2 = batch['data_contrastive']
        target_seg = batch['target']

        data = data.to(self.device, non_blocking=True)
        data2 = data2.to(self.device, non_blocking=True)
        if isinstance(target_seg, list):
            target_seg = [i.to(self.device, non_blocking=True) for i in target_seg]
        else:
            target_seg = target_seg.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output_seg, output_aux = self.network(data, data2)
            # del data
            l = self.loss(output_seg, target_seg, output_aux)
            if hasattr(self.loss, 'compound_loss'):
                comp_l = self.loss.compound_loss
            else: 
                comp_l = {}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.lr_scheduler.step()
        else:
            l.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
        out = {'loss': l.detach().cpu().numpy()}
        out.update({'comp_loss_%s' % l_i: comp_l[l_i].detach().cpu().numpy() for l_i in comp_l})
        return out
    
    def on_train_epoch_start(self):
        self.network.train()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=7)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        if isinstance(self.loss, DeepSupervisionWrapper):
            if hasattr(self.loss.loss, 'update_weights') and callable(self.loss.loss.update_weights):
                self.loss.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)
        else:
            if hasattr(self.loss, 'update_weights') and callable(self.loss.update_weights):
                self.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)


class nnUNetTrainerRangerContrastiveSupervision_higher_weight_250epochs(nnUNetTrainerRangerContrastiveSupervision_higher_weight):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerRangerContrastiveSupervision_higher_weight_noSmooth(nnUNetTrainerRangerContrastiveSupervision_higher_weight, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)


class nnUNetTrainerRangerContrastiveSupervision_higher_weight_noSmooth_250epochs(nnUNetTrainerRangerContrastiveSupervision_higher_weight_250epochs, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)