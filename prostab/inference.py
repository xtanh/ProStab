# import prostab
from omegaconf import OmegaConf
from prostab.utils import seed_everything
import torch
import os

from prostab.datamodules.datasets.utils import alt_parse_PDB
from prostab.datamodules.datasets.utils import get_pdb
from prostab.datamodules.datasets.data_utils import Alphabet
from prostab.models.stability.prostab import FusionModel



def get_prostab(ckpt_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.nn.Module:
    cfg = OmegaConf.load(os.path.join(ckpt_path,'.hydra/config.yaml'))

    del cfg['model']['_target_']
    seed_everything(cfg['train']['seed'])

    model = FusionModel(cfg['model']).to(device)
    ckpt = torch.load(os.path.join(ckpt_path,'./checkpoints/best.ckpt'), map_location=torch.device('cpu'))['state_dict']
    ckpt_remove_model = {k[6:]:v for k, v in ckpt.items() if 'model.' in k}
    model.load_state_dict(ckpt_remove_model, strict=False)    

    return model, cfg


    
    

def parse_pdb(pdb_path: str, pdb_name: str, chain: str, cfg, mutation=None ,device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """处理pdb结构，支持指定突变

    Args:
        pdb_path (str): pdb文件路径
        pdb_name (str): pdb名称
        chain (str): 链ID
        cfg (_type_): 配置信息
        mutation (_type_, optional): 突变信息，格式如“S11A”表示将第11位丝氨酸突变为丙氨酸
        device (str, optional): 计算设备. Defaults to 'cuda'iftorch.cuda.is_available()else'cpu'.
    """
    ALPAHBET = 'ACDEFGHIKLMNPQRSTVWYX'
    
    pdb_parsed = alt_parse_PDB(pdb_path, chain)
    resn_list = pdb_parsed[0]["resn_list"]
    original_seq = pdb_parsed[0]['seq']
    
    pdb = get_pdb(pdb_parsed[0], pdb_name, pdb_name, check_assert=False)
    
    if mutation:
        wtAA, mutAA = mutation[0], mutation[-1]
        pos = mutation[1:-1]
        
        try:
            pdb_idx = resn_list.index(pos)
            assert original_seq[pdb_idx] == wtAA, f"序列不匹配：期望 {wtAA}，实际为 {original_seq[pdb_idx]}"
            
            # 创建突变序列
            mut_sequence = list(original_seq)
            mut_sequence[pdb_idx] = mutAA
            mut_sequence = ''.join(mut_sequence)
            
            # 设置突变ID
            pdb['mut_ids'] = torch.tensor([pdb_idx])
            
            # 生成one-hot编码向量
            wt_onehot = torch.zeros((21))
            wt_onehot[ALPAHBET.index(wtAA)] = 1
            mt_onehot = torch.zeros((21))
            mt_onehot[ALPAHBET.index(mutAA)] = 1
            append_tensor = torch.cat([wt_onehot, mt_onehot])
            
            pdb['append_tensors'] = append_tensor.float().unsqueeze(0)
            pdb['mut_seq'] = [mut_sequence]
        except ValueError:
            print(f"警告：无法在PDB中找到残基位置 {pos}")
            pdb['mut_ids'] = torch.tensor([0])
            pdb['append_tensors'] = torch.zeros((1, 42))
            pdb['mut_seq'] = [original_seq]
    else:
        # 没有突变，使用默认值
        pdb['mut_ids'] = torch.tensor([0])
        pdb['append_tensors'] = torch.zeros((1, 42))
        pdb['mut_seq'] = [original_seq]
    pdb['ddG'] = torch.tensor([[0.0]])
    pdb['dataset'] = [pdb_name]
    alphabet = Alphabet(**cfg['datamodule']['alphabet'])
    pdb = alphabet.featurize([pdb])
    
    def move_tensors_to_device(d, device):
        if isinstance(d, dict):
            return {k: move_tensors_to_device(v, device) for k, v in d.items()}
        elif isinstance(d, list):
            return [move_tensors_to_device(v, device) for v in d]
        elif isinstance(d, torch.Tensor):
            return d.to(device)
        else:
            return d
    
    pdb = move_tensors_to_device(pdb, device)
    return pdb
        
        
                   