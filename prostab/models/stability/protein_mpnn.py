# form spurs
from dataclasses import dataclass



'''
Adapted from SPURS
https://github.com/luo-group/SPURS/blob/9cf686eb8304740775c4cfdd2437d36a96e97732/spurs/models/stability/protein_mpnn.py
'''

@dataclass
class ProteinMPNNConfig:
    d_model: int = 128
    d_node_feats: int = 128
    d_edge_feats: int = 128
    k_neighbors: int = 48
    augment_eps: float = 0.0
    n_enc_layers: int = 3
    dropout: float = 0.1
    
    tune: bool = False
    use_input_decoding_order: bool = False
    # decoder-only
    n_vocab: int = 22
    n_dec_layers: int = 3
    random_decoding_order: bool = True
    nar: bool = True
    crf: bool = False
    use_esm_alphabet: bool = False
