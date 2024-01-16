import os
from pathlib import Path
import yaml


## ==================== ##
## Load config
## ==================== ##


def find_root_dir(config_filename:str ='config_global.yml', depth:int =4):
    """
    Looks for up to n levels of parent directories for a configuration directory.
    """
    current_path = Path("./").resolve()
    for _ in range(depth):  # Look into up to n levels of parent directories
        if current_path / config_filename in list(current_path.iterdir()):
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError(f"Could not find the config file '{config_filename}' after looking at maximum depth.")


def read_config(config_filename:str ='config_global.yml', depth:int =4):
    """Reads the YAML configuration file from the repository root."""
    repo_root = find_root_dir(config_filename, depth)

    config_path = repo_root / config_filename
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find the config file '{config_filename}'.")

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    return config