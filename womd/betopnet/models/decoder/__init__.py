'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''

from .mtr_plus_plus_decoder import MTRPPDecoder
from .betop_decoder import BeTopDecoder
from .wayformer_decoder import WayformerDecoder

__all__ = {
    'MTRPPDecoder': MTRPPDecoder,
    'BeTopDecoder': BeTopDecoder,
    'WayformerDecoder':WayformerDecoder
}

def build_decoder(in_channels, config):
    model = __all__[config.NAME](
        in_channels=in_channels,
        config=config
    )
    return model
