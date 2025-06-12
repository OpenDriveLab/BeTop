'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Pipeline developed upon planTF: 
https://arxiv.org/pdf/2309.10443
'''
import logging
import sys
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection

from src.metrics import MR, minADE, minFDE
from src.optim.warmup_cos_lr import WarmupCosLR

from src.features.topo_features import generate_behavior_braids, generate_map_briads

from src.models.betop.modules.contigency_utils import contigency_loss
import math

logger = logging.getLogger(__name__)

def wrap_to_pi(theta):
    return (theta+math.pi) % (2*math.pi) - math.pi

def focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
    ) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.joint_pred = model.joint_pred

    def on_fit_start(self) -> None:
        metrics_collection = MetricCollection(
            {
                "minADE1": minADE(k=1).to(self.device),
                "minADE6": minADE(k=6).to(self.device),
                "minFDE1": minFDE(k=1).to(self.device),
                "minFDE6": minFDE(k=6).to(self.device),
                "MR": MR().to(self.device),
            }
        )
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
        }
        self.train_cnt = 0
        self.val_cnt = 0
        self.ep_cnt = 0
        print('start training')
    
    def on_train_epoch_start(self) -> None:
        self.train_cnt = 0
        self.val_cnt = 0
        self.ep_cnt += 1

    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        features, _, _ = batch
        res = self.forward(features["feature"].data)

        losses = self._compute_objectives(res, features["feature"].data)
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)

        return losses["loss"]
    
    def _build_topo_targets(self, data):

        #build actor occupancy preds: [b, A, A]
        targets = data["agent"]["position"][:, :, -80:]
        target_valid = data["agent"]["valid_mask"][:, :, -80 :]
        dist_valid = target_valid[..., :-1] * target_valid[..., 1:]
        actor_pos_curr= data["agent"]["position"][:, :, 20]
        actor_head_curr = wrap_to_pi(data["agent"]["heading"][:, :, 20])

        # mask the source occupancy for not-moving actors
        dyn = torch.linalg.norm(torch.diff(targets, dim=-2) * dist_valid.float()[..., None], dim=-1).sum(-1) 
        dyn = (dyn > 3).float() #[b, a]
        agent_valid = torch.any(target_valid.float(), dim=-1)
        comb_valid = (agent_valid* dyn)[:, :, None] * agent_valid[:, None, :]
        actor_occ = generate_behavior_braids(targets, actor_pos_curr, actor_head_curr)
        actor_occ = actor_occ * comb_valid.float()

        #build actor-map occupancy preds: [b, A, M]
        map_center = data["map"]["point_position"][:, :, 0]
        map_mask = data["map"]["valid_mask"]
        actor_map_mask = (agent_valid)[:, :, None] * torch.any(map_mask, dim=-1)[:, None, :].float()
        map_occ = generate_map_briads(targets, map_center, target_valid, map_mask)

        return actor_occ, map_occ, comb_valid, actor_map_mask
    
    def _topo_loss(
        self, prediction, targets, valid_mask,
        top_k=False, top_k_ratio=1.):
        """
        build the top-k CE loss for occupancy predictions
        preds, targets, valid_mask: [b, src, tgt]
        """
        b, s, t = prediction.shape
        targets = targets.float()

        loss = focal_loss(
            prediction,
            targets,
            reduction='none',
        )

        loss = loss * valid_mask
        loss = loss.view(b, s*t)
        if top_k:
            # Penalises the top-k hardest pixels
            k = int(top_k_ratio * loss.shape[-1])
            loss, _ = torch.sort(loss, dim=-1, descending=True)
            loss = loss[:, :k]
        
        mask = torch.sum((loss > 0).float())
        mask = mask + (mask == 0).float()
        
        return torch.sum(loss) / mask

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        loss = 0.

        actor_occ, map_occ, comb_valid, actor_map_mask = self._build_topo_targets(data)
        actor_pred, map_pred = res['actor_occ'], res['actor_map_occ']

        actor_occ_loss = self._topo_loss(actor_pred[..., 0], actor_occ.detach(), comb_valid.detach(), top_k=False, top_k_ratio=0.25)
        map_occ_loss = self._topo_loss(map_pred[..., 0], map_occ.detach(), actor_map_mask.detach(), top_k=False, top_k_ratio=0.25)
        occ_loss = actor_occ_loss + map_occ_loss #+ map_actor_occ_loss
        loss += 50 * occ_loss
        
        prediction = res["prediction"]
        agent_mask = data["agent"]["valid_mask"][:, 1:, -80 :]
        agent_target = data["agent"]["target"][:, 1:]

        full_agent_target = torch.cat(
            [
                agent_target[..., :2],
                torch.stack(
                    [
                        agent_target[..., 2].cos(), agent_target[..., 2].sin(), 
                     ], dim=-1
                ),
            ],
            dim=-1,
        )

        pred_probability = res['pred_probability']
        num_mode = prediction.shape[2]

        only_agent_mask = agent_mask.sum(-1)!=0
        full_agent_mask = agent_mask.unsqueeze(2).expand(-1, -1, num_mode, -1)
          
        agent_dist = torch.linalg.norm(prediction[..., :2] - full_agent_target[..., None, :, :2], dim=-1)
        agent_dist = agent_dist * full_agent_mask.float()
        pred_best_mode = torch.argmin(agent_dist.sum(-1), dim=-1)

        pred_best_traj = prediction[torch.arange(prediction.shape[0])[:, None, None], torch.arange(prediction.shape[1])[None, :, None], pred_best_mode[:, :, None]]
        pred_best_traj = pred_best_traj[:, :, 0]
        agent_reg_loss = F.smooth_l1_loss(pred_best_traj[agent_mask], full_agent_target[agent_mask])
        agent_cls_loss = F.cross_entropy(
            pred_probability.permute(0, 2, 1), pred_best_mode.detach(),
            reduction='none', label_smoothing=0.2
            )
        agent_cls_loss = torch.mean(agent_cls_loss[only_agent_mask])
        agent_reg_loss = agent_reg_loss + agent_cls_loss


        loss += self._compute_single_objectives(res, data,
                actor_occ, map_occ, comb_valid, actor_map_mask, actor_pred,
                prediction, pred_probability, only_agent_mask)

        loss += agent_reg_loss

        return {'loss':loss}


    def _compute_single_objectives(self, res, data, 
            actor_occ, map_occ, comb_valid, actor_map_mask, actor_occ_pred,
            prediction, pred_probability, only_agent_mask):
        
        trajectory, probability = (
            res["trajectory"],
            res["probability"],
        )

        targets = data["agent"]["target"]
        valid_mask = data["agent"]["valid_mask"][:, :, -trajectory.shape[-2] :]
        
        ego_target_pos, ego_target_heading = targets[:, 0, :, :2], targets[:, 0, :, 2]

        ego_target = torch.cat(
            [
                ego_target_pos,
                torch.stack(
                    [
                        ego_target_heading.cos(), ego_target_heading.sin(),
                    ], dim=-1
                ),
            ],
            dim=-1,
        )

        dist = torch.linalg.norm(trajectory[..., :2] - ego_target[:, None, :, :2], dim=-1)
        dist = dist.sum(-1)

        best_mode = torch.argmin(dist, dim=-1)
        best_traj = trajectory[torch.arange(trajectory.shape[0]), best_mode]

        ego_reg_loss = F.smooth_l1_loss(best_traj, ego_target)
        ego_cls_loss = F.cross_entropy(probability, best_mode.detach(), label_smoothing=0.2)


        loss = ego_reg_loss + ego_cls_loss
        
        if res['conti_loss']:
            joint_plan = res['joint_plan']
            angle = torch.atan2(prediction[..., 3], prediction[..., 2])
            prediction_xy = prediction[..., :2] + data["agent"]["position"][:, 1:, 20, None, None]
            angle = angle + data["agent"]["heading"][:, 1:, 20, None, None]
            angle = wrap_to_pi(angle)
            full_pred = torch.cat([prediction_xy, angle[..., None]], dim=-1).detach()

            plan_angle = torch.atan2(joint_plan[..., 3], joint_plan[..., 2]).detach()
            full_joint_plan = torch.cat([joint_plan[..., :2], plan_angle[..., None]], dim=-1)

            ego_behave_occ = actor_occ_pred[:, 0, 1:, 0].sigmoid().detach()

            conti_loss = contigency_loss(full_joint_plan, trajectory, full_pred,
            pred_probability, only_agent_mask, ego_behave_occ, None)

            loss += 0.1*conti_loss

            
        return loss

    def _compute_metrics(self, output, data, prefix) -> Dict[str, torch.Tensor]:
        metrics = self.metrics[prefix](output, data["agent"]["target"][:, 0])
        return metrics
    
    def return_dict_str(self, 
        metrics: Dict[str, torch.Tensor],
        ):
        ret_str = ''
        for k, v in metrics.items():
            buf_str = f'{k}-' + "{:.4f}|".format(v.mean().item())
            ret_str += buf_str
        return ret_str
    
    def train_or_val(self, prefix, interv=1000):
        if prefix == 'val':
            flag = (self.val_cnt!= 0) and (self.val_cnt % 100==0)
            return flag, self.val_cnt
        flag = (self.train_cnt!= 0) and (self.train_cnt % interv==0)
        return flag, self.train_cnt

    def _log_step(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        flag, cnt = self.train_or_val(prefix)

        if flag:
            np_loss = loss.mean().item()
            full_str = f'{prefix}-step:{cnt}-loss:|' + '{:.4f}'.format(np_loss)

        for key, value in objectives.items():
            self.log(
                f"objectives/{prefix}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        if flag:
            full_str += self.return_dict_str(objectives)

        if metrics is not None:
            self.log_dict(
                metrics,
                prog_bar=(prefix == "val"),
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )
            if flag:
                full_str += self.return_dict_str(metrics)
        
        if flag:
            print(full_str)
            # sys.stdout.flush()

    def training_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        self.train_cnt += 1
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        self.val_cnt +=1
        return self._step(batch, "val")

    def test_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)
    

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        # Get optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_scheduler
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]
