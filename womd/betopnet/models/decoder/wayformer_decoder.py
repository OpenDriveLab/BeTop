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

from betopnet.models.utils.wayformer_utils import (
    TrainableQueryProvider, CrossAttentionLayer, SelfAttentionBlock
    )

class WayformerDecoder(nn.Module):
    def __init__(
            self,
            config, 
            ):
        """Generic Perceiver IO decoder.

        :param output_adapter: Transforms generic decoder cross-attention output of shape (B, O, F) to task-specific
                output. B is the batch size, O the output sequence length and F the number of cross-attention output
                channels.
        :param output_query_provider: Provides the decoder's output query. Abstracts over output query details e.g. can
                be a learned query, a deterministic function of the model's input, etc. Configured by `PerceiverIO`
                subclasses.
        :param num_latent_channels: Number of latent channels of the Perceiver IO encoder output.
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key channels for cross-attention             (see
                `MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for cross-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param dropout: Dropout probability for cross-attention layer.
        :param init_scale: Standard deviation for random normal initialization of parameters.
        :param activation_checkpointing: If True, implements an activation checkpoint for the decoder's
            cross-attention layer.
        :param activation_offloading: If True, offloads checkpointed activations to CPU.
        """
        super().__init__()

        num_latent_channels = config.num_latent_channels
        num_cross_attention_heads = config.get('num_cross_attention_heads', 4)
        num_cross_attention_layers = config.get('num_cross_attention_layers', 8)
        num_cross_attention_qk_channels = config.get('num_cross_attention_qk_channels', None)
        num_cross_attention_v_channels = config.get('num_cross_attention_v_channels', None)
        cross_attention_widening_factor = config.get('cross_attention_widening_factor', 4)
        cross_attention_residual = config.get('cross_attention_residual', True)
        dropout = config.get('dropout', 0.1)
        init_scale = config.get('init_scale', 0.02)

        self.output_query_provider = TrainableQueryProvider(
            num_queries=config['num_queries_dec'],
            num_query_channels=config['hidden_size'],
            init_scale=0.1,
        )

        self.num_cross_attention_layers = num_cross_attention_layers
        self.self_attn = nn.ModuleList([SelfAttentionLayer(
            num_heads=num_cross_attention_heads,
            num_channels=num_latent_channels,
            num_qk_channels=num_latent_channels,
            num_v_channels=num_latent_channels,
            causal_attention=False,
            widening_factor=cross_attention_widening_factor,
            dropout=dropout
        ) for _ in range(num_cross_attention_layers)])
        
        self.cross_attn = nn.ModuleList([CrossAttentionLayer(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=self.output_query_provider.num_query_channels,
            num_kv_input_channels=num_latent_channels,
            num_qk_channels=num_cross_attention_qk_channels,
            num_v_channels=num_cross_attention_v_channels,
            widening_factor=cross_attention_widening_factor,
            attention_residual=cross_attention_residual,
            dropout=dropout,
        ) for _ in range(num_cross_attention_layers)])

        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            init_parameters(self, init_scale)

    def forward(self, x_latent, x_adapted=None, **kwargs):
        output_query = self.output_query_provider(x_adapted)

        output = self.cross_attn[0](output_query, x_latent).last_hidden_state

        for i in range(1, len(self.cross_attn)):
            output = self.self_attn[i - 1](output).last_hidden_state
            output = self.cross_attn[i](output, x_latent).last_hidden_state

        output = self.self_attn[-1](output).last_hidden_state
        return output