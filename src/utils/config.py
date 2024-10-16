from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], config_path: str):
    with open(config_path, "w") as f:
        yaml.dump(config, f)
