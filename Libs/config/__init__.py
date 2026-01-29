"""Configuration management helpers for accessing YAML-based settings."""

import os
from typing import Any, Dict

import yaml


def _config_path(filename: str) -> str:
    """Builds the absolute path to a configuration file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, filename)


def load_yaml_config(filename: str) -> Dict[str, Any]:
    """Loads a YAML configuration file and returns its content as a dictionary."""
    config_file = _config_path(filename)
    with open(config_file, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset_config() -> Dict[str, Any]:
    """Returns the dataset configuration defined in ``dataset.yaml``."""
    return load_yaml_config("dataset.yaml")


def load_model_config() -> Dict[str, Any]:
    """Returns the model configuration defined in ``model.yaml``."""
    return load_yaml_config("model.yaml")


def load_experiment_config() -> Dict[str, Any]:
    """Returns the experiment configuration defined in ``exp.yaml``."""
    return load_yaml_config("exp.yaml")

