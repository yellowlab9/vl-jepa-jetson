"""
Configuration management utilities
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Union, cast
from omegaconf import OmegaConf


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Config saved to {path}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    base_omega = OmegaConf.create(base_config)
    override_omega = OmegaConf.create(override_config)
    merged = OmegaConf.merge(base_omega, override_omega)
    
    return cast(Dict[str, Any], OmegaConf.to_container(merged, resolve=True))


def print_config(config: Dict[str, Any], indent: int = 0):
    """
    Pretty print configuration.
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_config(value, indent + 2)
        else:
            print(" " * indent + f"{key}: {value}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        config = load_config(config_path)
        print("Configuration:")
        print_config(config)
    else:
        print("Usage: python config.py <config_path>")
