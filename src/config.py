from typing import Any

import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict[str, Any], config_path: str):
    with open(config_path, "w") as f:
        yaml.dump(config, f)
