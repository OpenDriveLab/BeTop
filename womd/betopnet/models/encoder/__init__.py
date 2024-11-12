'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''

from .mtr_encoder import MTREncoder
from .wayformer_encoder import WayformerEncoder
from .mtr_plus_plus_encoder import MTRPPEncoder

__all__ = {
    'MTREncoder': MTREncoder,
    'WayformerEncoder':WayformerEncoder,
    'MTRPPEncoder': MTRPPEncoder,
}


def build_encoder(config):
    model = __all__[config.NAME](
        config=config
    )
    return model
