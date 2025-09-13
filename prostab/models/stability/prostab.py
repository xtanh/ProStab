from dataclasses import dataclass, field
from typing import List

import torch
from prostab.models import register_model
from prostab.models.stability.basemodel import BaseModel
from prostab.models.stability.protein_mpnn import ProteinMPNNConfig

from prostab.models.stability.modules.esm2 import ESM2
from prostab import utils
from prostab.models.stability.org_transfer_model import get_protein_mpnn
import torch.nn.functional as F
from prostab.models.stability.modules.esm2 import ESM2
# from prostab.models.stability.modules.cross_attention import CrossAttention
import torch.nn as nn
from ipdb import set_trace

log = utils.get_logger(__name__)
from .mlp import MLP, MLPConfig

'''
ProStab Model Implementation
This implementation builds upon the SPURS framework with key enhancements:
1. Cross-fusion feature integration mechanism
2. Explicit modeling of mutation-specific information differences


Based on SPURS foundation:
https://github.com/luo-group/SPURS/blob/9cf686eb8304740775c4cfdd2437d36a96e97732/spurs/models/stability/spurs.py

License: Same as SPURS (please refer to original repository)
'''



@dataclass
class FusionConfig:
    encoder_mpnn: ProteinMPNNConfig = field(default=ProteinMPNNConfig())
    esm_name: str = 'esm2_t33_650M_UR50D'
    mpnn_name: str = 'ProteinMPNN'
    dropout: float = 0.1
    mlp: MLPConfig = field(default=MLPConfig())
    esm_tune: bool = True
    mpnn_tune: bool = True


@register_model('esmfusion')
class FusionModel(BaseModel):
    _default_cfg = FusionConfig()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        
        # initialize ESM2
        self.esm_decoder = ESM2.from_pretrained(args=self.cfg, name=self.cfg.esm_name)
        self.padding_idx = self.esm_decoder.padding_idx
        self.mask_idx = self.esm_decoder.mask_idx
        self.cls_idx = self.esm_decoder.cls_idx
        self.eos_idx = self.esm_decoder.eos_idx
        
        # initialize ProteinMPNN
        self.mpnn_encoder = get_protein_mpnn(tune=cfg.mpnn_tune)
        self.use_input_decoding_order = cfg.encoder_mpnn.use_input_decoding_order
        self.mlp = MLP(self.cfg.mlp)
    
        
        encoder_layer_esm2 = nn.TransformerEncoderLayer(d_model=1280, nhead=8, dropout=0.1
                                                        ,dim_feedforward=2560,batch_first=True)
        
        self.transformer_encoder_esm2 = nn.TransformerEncoder(encoder_layer_esm2, num_layers=4)
        
        encoder_layer_protein = nn.TransformerEncoderLayer(d_model=1792, nhead=8, dropout=0.1
                                                          ,dim_feedforward=3584,batch_first=True)
        
        self.transformer_encoder_protein = nn.TransformerEncoder(encoder_layer_protein, num_layers=2)
        
        self.ddg_out = nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Dropout(0.1),nn.Linear(16,1))
        
        self.pos_reduction = nn.Sequential(nn.Linear(1280,640),nn.ReLU(),nn.Dropout(0.1),nn.Linear(640,320),nn.ReLU(),nn.Dropout(0.1),nn.Linear(320,160),nn.ReLU(),nn.Dropout(0.1),nn.Linear(160,1))
        
    def forward(self, batch, **kwargs):
        
        with torch.set_grad_enabled(self.cfg.mpnn_tune):
            mpnn_features = self.forward_mpnn(batch)
        
        batch['mut_ids'] = batch['mut_ids'] if isinstance(batch['mut_ids'], torch.Tensor) else torch.tensor(batch['mut_ids'])
        shifed_mut_ids = batch['mut_ids'].to(mpnn_features.device)
        
        with torch.set_grad_enabled(self.cfg.esm_tune):
            wt_esm2 = self.esm_decoder(
                tokens=batch['tokens'],
                encoder_out=None,
            )
            wt_esm2_features = wt_esm2['representations'][-1]
            wt_esm2_features = wt_esm2_features[:,1:-1]
            diff_features_list = []
            for i in range(0,len(batch['mut_tokens']),500):
                batch_mut_tokens_stacked = torch.cat(batch['mut_tokens'][i:i+500], dim=0)  # [num_mutations, seq_len]
               
                mt_esm2 = self.esm_decoder(
                    tokens=batch_mut_tokens_stacked,
                    encoder_out=None,
                )
                mt_features = mt_esm2['representations'][-1]  # [num_mutations, seq_len, hidden_dim]
                mt_esm2_features = mt_features[:,1:-1]
                wt_esm2_features_expanded = wt_esm2_features.expand(mt_esm2_features.shape[0], -1, -1)
                diff_features = mt_esm2_features - wt_esm2_features_expanded
                del mt_features, mt_esm2_features, wt_esm2_features_expanded
                diff_features_list.append(diff_features)
            diff_features = torch.cat(diff_features_list, dim=0)

        
        protein_feature = torch.cat([wt_esm2_features, mpnn_features], dim=-1)
        protein_feature = self.transformer_encoder_protein(protein_feature)
        batch['muted_id_representation'] = protein_feature[:,shifed_mut_ids]
        ddg_out = self.mlp(batch)
        ddg_out_aa = (ddg_out * batch['append_tensors'][:, 21:]).sum(-1)
        ddg_out_wt_aa = (ddg_out * batch['append_tensors'][:, :21]).sum(-1)
        ddg = ddg_out_aa - ddg_out_wt_aa
        mpnn_ddg =ddg.view(-1,1)
        delta_features = diff_features[torch.arange(diff_features.shape[0]),shifed_mut_ids]
        del diff_features
        delta = self.pos_reduction(delta_features)
        fusion = torch.cat([mpnn_ddg,delta],dim=-1)
        ddg = self.ddg_out(fusion)
        ddg = ddg.squeeze(-1)     
        return ddg
    
    def forward_mpnn(self, batch):
        X = batch['X']
        S = batch['S']
        mask = batch['mask']
        chain_M = batch['chain_M']
        chain_M_chain_M_pos = batch['chain_M_chain_M_pos']
        residue_idx = batch['residue_idx']
        chain_encoding_all = batch['chain_encoding_all']
        randn_1 = batch['randn_1']
        
        all_mpnn_hid, mpnn_embed, _ = self.mpnn_encoder(
            X, S, mask, chain_M, residue_idx, chain_encoding_all, None, 
            self.use_input_decoding_order
        )
        
        all_mpnn_hid = torch.cat([all_mpnn_hid[0],all_mpnn_hid[1], all_mpnn_hid[2],mpnn_embed], dim=-1)
        return all_mpnn_hid 