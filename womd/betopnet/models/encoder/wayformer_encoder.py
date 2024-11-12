'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''
'''
Mostly from UniTraj (ECCV'24): https://arxiv.org/abs/2403.15098
'''

from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
from einops import rearrange
from torch import nn as nn

from betopnet.models.utils.wayformer_utils import (TrainableQueryProvider, 
    CrossAttentionLayer, SelfAttentionBlock)


class WayformerEncoder(nn.Module):
    def __init__(
            self,
            config,
        ):  
        """Generic Perceiver IO encoder.

        :param input_adapter: Transforms and position-encodes task-specific input to generic encoder input of shape (B,
                M, C) where B is the batch size, M the input sequence length and C the number of key/value input
                channels. C is determined by the `num_input_channels` property of the `input_adapter`.
        :param num_latents: Number of latent variables (N).
        :param num_latent_channels: Number of latent channels (D).
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key channels for cross-attention
                (see`MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for cross-attention (see
                `MultiHeadAttention.num_v_channels` for details).
        :param num_cross_attention_layers: Number of cross-attention layers (alternating with self-attention blocks).
        :param first_cross_attention_layer_shared: Whether the first cross-attention layer should share its weights with
                subsequent cross-attention layers (if any).
        :param num_self_attention_heads: Number of self-attention heads.
        :param num_self_attention_qk_channels: Number of query and key channels for self-attention (see
                `MultiHeadAttention.num_qk_channels` for details).
        :param num_self_attention_v_channels: Number of value channels for self-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param num_self_attention_layers_per_block: Number of self-attention layers per self-attention block.
        :param num_self_attention_blocks: Number of self-attention blocks, with weights shared between corresponding
            self-attention layers.
        :param first_self_attention_block_shared: Whether the first self-attention block should share its weights with
                subsequent self-attention blocks (if any).
        :param dropout: Dropout probability for self- and cross-attention layers.
        :param residual_dropout: Dropout probability for residual connections.
        :param init_scale: Standard deviation for random normal initialization of parameters.
        :param activation_checkpointing: If True, implements an activation checkpoint for each self-attention layer and
                each cross-attention layer.
        :param activation_offloading: If True, offloads checkpointed activations to CPU.
        """
        super().__init__()

        num_latents = config.num_latents
        num_latent_channels = config.num_latent_channels
        num_cross_attention_heads = config.get('num_cross_attention_heads', 4)
        num_cross_attention_qk_channels = config.get('num_cross_attention_qk_channels', None)
        num_cross_attention_v_channels= config.get('num_cross_attention_v_channels', None)
        num_cross_attention_layers= config.get('num_cross_attention_layers', 1)
        first_cross_attention_layer_shared= config.get('first_cross_attention_layer_shared', False)
        cross_attention_widening_factor= config.get('cross_attention_widening_factor', 4)
        num_self_attention_heads= config.get('num_self_attention_heads', 4)
        num_self_attention_qk_channels= config.get('num_self_attention_qk_channels', None)
        num_self_attention_v_channels= config.get('num_self_attention_v_channels', None)
        num_self_attention_layers_per_block= config.get('num_self_attention_layers_per_block', 1)
        num_self_attention_blocks= config.get('num_self_attention_blocks', 2)
        first_self_attention_block_shared= config.get('first_self_attention_block_shared', False)
        self_attention_widening_factor= config.get('self_attention_widening_factor', 4)
        dropout= config.get('dropout', 0.1)
        residual_dropout= config.get('residual_dropout', 0.0)
        init_scale= config.get('init_scale', 0.02)

        self.latent_provider = TrainableQueryProvider(num_latents, num_latent_channels, init_scale=0.02)

        if num_cross_attention_layers <= 0:
            raise ValueError("num_cross_attention_layers must be > 0")

        if num_self_attention_blocks <= 0:
            raise ValueError("num_self_attention_blocks must be > 0")

        if num_cross_attention_layers > num_self_attention_blocks:
            raise ValueError("num_cross_attention_layers must be <= num_self_attention_blocks")

        self.num_cross_attention_layers = num_cross_attention_layers
        self.num_self_attention_blocks = num_self_attention_blocks

        self.first_cross_attention_layer_shared = first_cross_attention_layer_shared
        self.first_self_attention_block_shared = first_self_attention_block_shared

        def cross_attn():
            layer = CrossAttentionLayer(
                num_heads=num_cross_attention_heads,
                num_q_input_channels=num_latent_channels,
                num_kv_input_channels=num_latent_channels,
                num_qk_channels=num_cross_attention_qk_channels,
                num_v_channels=num_cross_attention_v_channels,
                widening_factor=cross_attention_widening_factor,
                dropout=dropout,
                residual_dropout=residual_dropout,
            )
            return (
                layer
            )

        def self_attn():
            return SelfAttentionBlock(
                num_layers=num_self_attention_layers_per_block,
                num_heads=num_self_attention_heads,
                num_channels=num_latent_channels,
                num_qk_channels=num_self_attention_qk_channels,
                num_v_channels=num_self_attention_v_channels,
                widening_factor=self_attention_widening_factor,
                dropout=dropout,
                residual_dropout=residual_dropout,
            )

        self.cross_attn_1 = cross_attn()
        self.self_attn_1 = self_attn()

        if self.extra_cross_attention_layer:
            self.cross_attn_n = cross_attn()

        if self.extra_self_attention_block:
            self.self_attn_n = self_attn()

        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            init_parameters(self, init_scale)

    @property
    def extra_cross_attention_layer(self):
        return self.num_cross_attention_layers > 1 and not self.first_cross_attention_layer_shared

    @property
    def extra_self_attention_block(self):
        return self.num_self_attention_blocks > 1 and not self.first_self_attention_block_shared

    def forward(self, x, pad_mask=None, return_adapted_input=False):
        b, *_ = x.shape

        x_adapted = x
        x_latent = self.latent_provider()

        x_latent = self.cross_attn_1(x_latent, x_adapted, pad_mask=pad_mask).last_hidden_state

        x_latent = self.self_attn_1(x_latent).last_hidden_state

        cross_attn_n = self.cross_attn_n if self.extra_cross_attention_layer else self.cross_attn_1
        self_attn_n = self.self_attn_n if self.extra_self_attention_block else self.self_attn_1

        for i in range(1, self.num_self_attention_blocks):
            if i < self.num_cross_attention_layers:
                x_latent = cross_attn_n(x_latent, x_adapted, pad_mask=pad_mask).last_hidden_state
            x_latent = self_attn_n(x_latent).last_hidden_state

        if return_adapted_input:
            return x_latent, x_adapted
        else:
            return x_latent