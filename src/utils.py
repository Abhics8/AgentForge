"""
Utility functions for AgentForge.

Provides:
  - Config loading from YAML files
  - (More utilities like plotting will be added here in Phase 2)
"""

import yaml
from typing import Dict, Any


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    Load hyperparameters from a YAML config file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary of configuration values
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
