import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import shutil


import pyrootutils


'''
Adapted from SPURS
https://github.com/luo-group/SPURS/blob/9cf686eb8304740775c4cfdd2437732/spurs/train.py
'''


# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    # load environment variables from `.env` file if it exists
    # recursively searches for `.env` in all folders starting from work dir
    dotenv=True,
)


@hydra.main(config_path=f"{root}/configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """
    训练 SPURS 模型的主函数
    
    Args:
        cfg: Hydra 配置对象
    """
    # 导入必要的模块
    from prostab import utils
    from prostab.training_pipeline import train
    
    
    cfg = utils.resolve_experiment_config(cfg)
    
    cfg = utils.extras(cfg)
    
    # 开始训练
    return train(cfg)

if __name__ == "__main__":
    main()