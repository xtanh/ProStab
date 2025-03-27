# https://github.com/BytedProtein/ByProt/blob/dd279dc85f76ee2c28c819b71bf3911b90159f0a/src/byprot/models/fixedbb/__init__.py
from omegaconf import OmegaConf
try:
    import esm
    ESM_INSTALLED = True
except:
    ESM_INSTALLED = False

from prostab.utils.config import compose_config, merge_config

import torch
from torch import nn
import numpy as np
import logging

log = logging.getLogger(__name__)

class BaseModel(nn.Module):
    _default_cfg = None

    def __init__(self, cfg) -> None:
        super().__init__()
        self._update_cfg(cfg)

    def _update_cfg(self, cfg):
        if self._default_cfg is None:
            self.cfg = cfg
        else:
            try:
                self.cfg = OmegaConf.merge(self._default_cfg, cfg)
            except Exception as e:
                log.error(f"Error merging configs: {e}")
                log.error(f"Default config: {self._default_cfg}")
                log.error(f"Input config: {cfg}")
                raise e

    @classmethod
    def from_config(cls, cfg):
        raise NotImplementedError

    def forward_encoder(self, batch):
        raise NotImplementedError

    def forward_decoder(self, prev_decoder_out, encoder_out):
        raise NotImplementedError

    def initialize_output_tokens(self, batch, encoder_out):
        raise NotImplementedError

    def forward(self, coords, coord_mask, tokens, token_padding_mask=None, **kwargs):
        raise NotImplementedError

    def sample(self, coords, coord_mask, tokens=None, token_padding_mask=None, **kwargs):
        raise NotImplementedError
