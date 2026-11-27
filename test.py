"""
Adapted from SPURS
https://github.com/luo-group/SPURS/blob/9cf686eb8304740775c4cfdd2437732/spurs/test.py
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


@hydra.main(config_path=f"{root}/configs", config_name="test.yaml")
def main(config: DictConfig):
    from prostab import utils
    from prostab.testing_pipeline import test

    config = utils.resolve_experiment_config(config)
    config = utils.extras(config)

    return test(config)


if __name__ == "__main__":
    main()
