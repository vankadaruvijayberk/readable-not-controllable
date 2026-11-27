from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paper.yaml"


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def add_config_arg(parser):
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config file")
