"""Reformat datasets."""
import argparse

from src import data


def main(args: argparse.Namespace) -> None:
    """Do the reformatting by loading the dataset once."""
    data.load_dataset(args.dataset, file=args.dataset_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reformat datasets on disk")
    data.add_dataset_args(parser)
    args = parser.parse_args()
    main(args)
