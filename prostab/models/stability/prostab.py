from dataclasses import dataclass, field

import torch
from prostab.models import register_model
from prostab.models.stability.basemodel import BaseModel
from prostab.models.stability.protein_mpnn import ProteinMPNNConfig

from prostab.models.stability.org_transfer_model import get_protein_mpnn
from prostab.models.stability.modules.esm2 import ESM2
import torch.nn as nn

from .mlp import MLP, MLPConfig

# Adapted from SPURS; source and license:
# https://github.com/luo-group/SPURS/blob/9cf686eb8304740775c4cfdd2437d36a96e97732/spurs/models/stability/spurs.py


@dataclass
class FusionConfig:
    encoder_mpnn: ProteinMPNNConfig = field(default_factory=ProteinMPNNConfig)
    esm_name: str = "esm2_t33_650M_UR50D"
    mpnn_name: str = "ProteinMPNN"
    dropout: float = 0.1
    mlp: MLPConfig = field(default_factory=MLPConfig)
    esm_tune: bool = True
    mpnn_tune: bool = True


@register_model("esmfusion")
class FusionModel(BaseModel):
    _default_cfg = FusionConfig()

    def __init__(self, cfg, *, mpnn_weights=None, esm_weights=None) -> None:
        super().__init__(cfg)

        self.esm_decoder = ESM2.from_pretrained(
            args=self.cfg, name=self.cfg.esm_name, weights_path=esm_weights
        )
        self.padding_idx = self.esm_decoder.padding_idx
        self.mask_idx = self.esm_decoder.mask_idx
        self.cls_idx = self.esm_decoder.cls_idx
        self.eos_idx = self.esm_decoder.eos_idx

        self.mpnn_encoder = get_protein_mpnn(
            tune=cfg.mpnn_tune, checkpoint_path=mpnn_weights
        )
        self.use_input_decoding_order = cfg.encoder_mpnn.use_input_decoding_order
        self.mlp = MLP(self.cfg.mlp)

        encoder_layer_esm2 = nn.TransformerEncoderLayer(
            d_model=1280, nhead=8, dropout=0.1, dim_feedforward=2560, batch_first=True
        )

        self.transformer_encoder_esm2 = nn.TransformerEncoder(
            encoder_layer_esm2, num_layers=4
        )

        encoder_layer_protein = nn.TransformerEncoderLayer(
            d_model=1792, nhead=8, dropout=0.1, dim_feedforward=3584, batch_first=True
        )

        self.transformer_encoder_protein = nn.TransformerEncoder(
            encoder_layer_protein, num_layers=2
        )

        self.ddg_out = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 1)
        )

        self.pos_reduction = nn.Sequential(
            nn.Linear(1280, 640),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(640, 320),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(320, 160),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(160, 1),
        )

    def forward(self, batch, **kwargs):

        with torch.set_grad_enabled(self.cfg.mpnn_tune):
            mpnn_features = self.forward_mpnn(batch)

        batch["mut_ids"] = (
            batch["mut_ids"]
            if isinstance(batch["mut_ids"], torch.Tensor)
            else torch.tensor(batch["mut_ids"])
        )
        mutation_indices = batch["mut_ids"].to(mpnn_features.device)

        with torch.set_grad_enabled(self.cfg.esm_tune):
            wt_esm2 = self.esm_decoder(
                tokens=batch["tokens"],
                encoder_out=None,
            )
            wt_esm2_features = wt_esm2["representations"][-1]
            wt_esm2_features = wt_esm2_features[:, 1:-1]
            diff_features_list = []
            for i in range(0, len(batch["mut_tokens"]), 500):
                batch_mut_tokens_stacked = torch.cat(
                    batch["mut_tokens"][i : i + 500], dim=0
                )  # [num_mutations, seq_len]

                mt_esm2 = self.esm_decoder(
                    tokens=batch_mut_tokens_stacked,
                    encoder_out=None,
                )
                mt_features = mt_esm2["representations"][
                    -1
                ]  # [num_mutations, seq_len, hidden_dim]
                mt_esm2_features = mt_features[:, 1:-1]
                wt_esm2_features_expanded = wt_esm2_features.expand(
                    mt_esm2_features.shape[0], -1, -1
                )
                diff_features = mt_esm2_features - wt_esm2_features_expanded
                del mt_features, mt_esm2_features, wt_esm2_features_expanded
                diff_features_list.append(diff_features)
            diff_features = torch.cat(diff_features_list, dim=0)

        protein_feature = torch.cat([wt_esm2_features, mpnn_features], dim=-1)
        protein_feature = self.transformer_encoder_protein(protein_feature)
        batch["muted_id_representation"] = protein_feature[:, mutation_indices]
        ddg_out = self.mlp(batch)
        ddg_out_aa = (ddg_out * batch["append_tensors"][:, 21:]).sum(-1)
        ddg_out_wt_aa = (ddg_out * batch["append_tensors"][:, :21]).sum(-1)
        ddg = ddg_out_aa - ddg_out_wt_aa
        mpnn_ddg = ddg.view(-1, 1)
        delta_features = diff_features[
            torch.arange(diff_features.shape[0]), mutation_indices
        ]
        del diff_features
        delta = self.pos_reduction(delta_features)
        fusion = torch.cat([mpnn_ddg, delta], dim=-1)
        ddg = self.ddg_out(fusion)
        ddg = ddg.squeeze(-1)
        return ddg

    def forward_mpnn(self, batch):
        X = batch["X"]
        S = batch["S"]
        mask = batch["mask"]
        chain_M = batch["chain_M"]
        residue_idx = batch["residue_idx"]
        chain_encoding_all = batch["chain_encoding_all"]

        all_mpnn_hid, mpnn_embed, _ = self.mpnn_encoder(
            X,
            S,
            mask,
            chain_M,
            residue_idx,
            chain_encoding_all,
            None,
            self.use_input_decoding_order,
        )

        all_mpnn_hid = torch.cat(
            [all_mpnn_hid[0], all_mpnn_hid[1], all_mpnn_hid[2], mpnn_embed], dim=-1
        )
        return all_mpnn_hid
