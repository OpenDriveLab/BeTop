'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Pipeline developed upon Motion Transformer (MTR): 
https://arxiv.org/abs/2209.13508
'''

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from betopnet.models.utils.base_model import BaseModel
from .encoder import build_encoder
from .decoder import build_decoder


class MTRPP(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config

        self.encoder = build_encoder(
            self.model_cfg.ENCODER
        )
        self.decoder = build_decoder(
            in_channels=self.encoder.num_out_channels,
            config=self.model_cfg.DECODER
        )

    def forward(self, batch_dict):
        batch_dict = self.encoder(batch_dict)
        batch_dict = self.decoder(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_loss()
            tb_dict.update({'loss': loss.item()})
            disp_dict.update({'loss': loss.item()})
            return loss, tb_dict, disp_dict
        return batch_dict

    def get_loss(self):
        loss, tb_dict, disp_dict = self.decoder.get_loss()
        return loss, tb_dict, disp_dict