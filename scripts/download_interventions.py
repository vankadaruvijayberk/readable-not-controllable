"""Download the per-example intervention outputs behind Table 5 from the Hugging Face Hub."""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from actground.config import add_config_arg, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--out", default="outputs/interventions")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in cfg["models"]:
        for dataset in cfg["datasets"]:
            name = f"{model}__{dataset}.parquet"
            cached = hf_hub_download(cfg["interventions_repo"], f"interventions/{name}", repo_type="dataset")
            shutil.copy(cached, out_dir / name)
            print(out_dir / name)


if __name__ == "__main__":
    main()
