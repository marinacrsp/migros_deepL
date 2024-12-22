import argparse
import yaml
from datetime import datetime
from typing import Any, Dict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to configuration file.",
        default="./config/config.yaml",
    )

    return parser.parse_args()


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to a YAML file."""
    path = Path(config["output"]["path"]) / config["timestamp"]
    # Ensure path exists
    path.mkdir(parents=True, exist_ok=True)

    filename = path / "config.yaml"
    with open(filename, "w") as file:
        yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)


def load_config(file_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    config["timestamp"] = datetime.now().strftime("%m-%d_%Hh%Mm%Ss")
    save_config(config)
    print(config)
    return config