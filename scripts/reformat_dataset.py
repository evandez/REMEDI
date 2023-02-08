"""Reformat datasets.

This script handles reformatting and caching dataset from outside sources.
Callers usually should specify --dataset-file and point to the raw data
downloaded from its source.

Once the data is reformatted, you'll no longer have to specify --dataset-file
to any other scripts. The code will simply read it from the cache.
"""
import argparse

from remedi import data
from remedi.utils import logging_utils


def main(args: argparse.Namespace) -> None:
    """Do the reformatting by loading the dataset once."""
    data.disable_caching()
    logging_utils.configure(args=args)
    data.load_dataset(args.dataset, file=args.dataset_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reformat datasets on disk")
    data.add_dataset_args(parser)
    logging_utils.add_logging_args(parser)
    args = parser.parse_args()
    main(args)
