import os
import os.path as osp

import pandas as pd

from torch_geometric.data import DataLoader

import torch

from torch import optim as optim_module
from torch.optim import lr_scheduler as lr_sched_module
from torch.nn import functional as F

import pytorch_lightning as pl

from mot_neural_solver.data.mot_graph_dataset import MOTGraphDataset
from mot_neural_solver.models.mpn import MOTMPNet
from mot_neural_solver.models.resnet import resnet50_fc256, load_pretrained_weights
from mot_neural_solver.path_cfg import OUTPUT_PATH
from mot_neural_solver.utils.evaluation import compute_perform_metrics
from mot_neural_solver.tracker.mpn_tracker import MPNTracker
import time


class MOTNeuralSolver(pl.LightningModule):
    """
    Pytorch Lightning wrapper around the MPN defined in model/mpn.py.
    (see https://pytorch-lightning.readthedocs.io/en/latest/lightning-module.html)

    It includes all data loading and train / val logic., and it is used for both training and testing models.
    """
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.model = self.load_model()
    
    def forward(self, x):
        self.model(x)

    def load_model(self):
        model = MOTMPNet(self.hparams['graph_model_params']).cuda()
        return model

    def _get_data(self, mode, return_data_loader = True):
        assert mode in ('train', 'val', 'test')

        dataset = MOTGraphDataset(dataset_params=self.hparams['dataset_params'],
                                  mode=mode,
                                  splits= self.hparams['data_splits'][mode],
                                  logger=None)

        if return_data_loader and len(dataset) > 0:
            train_dataloader = DataLoader(dataset,
                                          batch_size = self.hparams['train_params']['batch_size'],
                                          shuffle = True if mode == 'train' else False,
                                          num_workers=self.hparams['train_params']['num_workers'])
            return train_dataloader
        
        elif return_data_loader and len(dataset) == 0:
            return []
        
        else:
            return dataset

    def train_dataloader(self):
        return self._get_data(mode = 'train')

    def val_dataloader(self):
        return self._get_data('val')

    def test_dataset(self, return_data_loader=False):
        return self._get_data('test', return_data_loader = return_data_loader)

    def configure_optimizers(self):
        optim_class = getattr(optim_module, self.hparams['train_params']['optimizer']['type'])
        optimizer = optim_class(self.model.parameters(), **self.hparams['train_params']['optimizer']['args'])

        if self.hparams['train_params']['lr_scheduler']['type'] is not None:
            lr_sched_class = getattr(lr_sched_module, self.hparams['train_params']['lr_scheduler']['type'])
            lr_scheduler = lr_sched_class(optimizer, **self.hparams['train_params']['lr_scheduler']['args'])

            return [optimizer], [lr_scheduler]

        else:
            return optimizer

    def _compute_loss(self, outputs, batch):
        # Define Balancing weight
        positive_vals = batch.edge_labels.sum()

        if positive_vals:
            pos_weight = (batch.edge_labels.shape[0] - positive_vals) / positive_vals

        else:  # If there are no positives labels, avoid dividing by zero
            pos_weight = torch.zeros(1, device=positive_vals.device)

        # Compute Weighted BCE:
        loss = 0
        num_steps = len(outputs['classified_edges'])
        for step in range(num_steps):
            # Classification loss
            cls_loss = F.binary_cross_entropy_with_logits(outputs['classified_edges'][step].view(-1),
                                                            batch.edge_labels.view(-1),
                                                            pos_weight= pos_weight)
            loss += self.hparams['train_params']['loss_weights']['tracking'] * cls_loss

            # Mask loss
            gt_masks = batch.mask_labels
            ixs = batch.mask_gt_ixs
            pred_masks = outputs['mask_predictions'][step]
            # Use only valid ixs
            gt_masks = gt_masks[ixs]
            pred_masks = pred_masks[ixs]
            # Contribute only if there is a matched det
            if gt_masks.numel():
                mask_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
                loss += self.hparams['train_params']['loss_weights']['segmentation']*mask_loss

        return loss

    def _train_val_step(self, batch, batch_idx, train_val):
        device = (next(self.model.parameters())).device
        batch.to(device)

        outputs = self.model(batch)
        loss = self._compute_loss(outputs, batch)
        logs = {**compute_perform_metrics(outputs, batch), **{'loss': loss}}
        log = {key + f'/{train_val}': val for key, val in logs.items()}

        if train_val == 'train':
            return {'loss': loss, 'log': log}

        else:
            return log

    def training_step(self, batch, batch_idx):
        return self._train_val_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._train_val_step(batch, batch_idx, 'val')

    def validation_epoch_end(self, outputs):
        metrics = pd.DataFrame(outputs).mean(axis=0).to_dict()
        metrics = {metric_name: torch.as_tensor(metric) for metric_name, metric in metrics.items()}
        return {'val_loss': metrics['loss/val'], 'log': metrics}

    def track_all_seqs(self, output_files_dir, dataset, use_gt = False, verbose = False, pred_oracles=False):
        tracker = MPNTracker(dataset=dataset,
                             graph_model=self.model,
                             use_gt=use_gt,
                             eval_params=self.hparams['eval_params'],
                             dataset_params=self.hparams['dataset_params'])

        constr_satisf_rate = pd.Series(dtype=float)
        for seq_name in dataset.seq_names:
            if verbose:
                print("Tracking sequence ", seq_name)
            tracker.track(seq_name)
            constr_satisf_rate[seq_name] = tracker.full_graph.constr_satisf_rate
            os.makedirs(output_files_dir, exist_ok=True)
            tracker.save_results_to_file(osp.join(output_files_dir, seq_name + '.txt'))

            # Prediction oracles
            if pred_oracles:
                print("Prediction oracles are being calculated")
                gt_edge_dir = osp.join(output_files_dir, 'pred_oracles', 'gt_edge')
                gt_mask_dir = osp.join(output_files_dir, 'pred_oracles', 'gt_mask')
                os.makedirs(gt_edge_dir, exist_ok=True)
                os.makedirs(gt_mask_dir, exist_ok=True)

                tracker.track(seq_name, pred_oracle_mode='gt_edge')
                tracker.save_results_to_file(osp.join(gt_edge_dir, seq_name + '.txt'))
                tracker.track(seq_name, pred_oracle_mode='gt_mask')
                tracker.save_results_to_file(osp.join(gt_mask_dir, seq_name + '.txt'))

            if verbose:
                print("Done! \n")
        constr_satisf_rate['OVERALL'] = constr_satisf_rate.mean()

        return constr_satisf_rate
