"""Download the four medical QA datasets from the Hugging Face Hub into the local datasets cache."""

import argparse

from actground.config import add_config_arg, load_config
from actground.data import HF_SOURCES, load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    cfg = load_config(parser.parse_args().config)
    for dataset in cfg["datasets"]:
        split = load_split(dataset)
        path, name = HF_SOURCES[dataset]
        print(f"{dataset}: {len(split)} rows from {path}" + (f" ({name})" if name else ""))


if __name__ == "__main__":
    main()
