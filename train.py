"""
Adapted from SPURS
https://github.com/luo-group/SPURS/blob/9cf686eb8304740775c4cfdd2437732/spurs/train.py
"""

import hydra
import pyrootutils
from omegaconf import DictConfig


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


@hydra.main(config_path=f"{root}/configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Train ProStab from a Hydra configuration."""
    from prostab import utils
    from prostab.training_pipeline import train

    cfg = utils.resolve_experiment_config(cfg)
    cfg = utils.extras(cfg)
    return train(cfg)


if __name__ == "__main__":
    main()
