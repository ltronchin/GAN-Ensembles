import sys
sys.path.extend([
    "./"
])

import pandas as pd
import os
import argparse
import re

from src.general_utils import util_general
from src.general_utils import util_path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-save", "--save_dir", type=str, default="./")
    parser.add_argument("--dataset_name", type=str)
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args, unknown = parser.parse_known_args()

    # List of folders
    reports_dir = args.save_dir
    dataset_name = args.dataset_name
    filenames = os.listdir(reports_dir)

    # Pattern to capture GAN names and step sequences.
    # - GAN names: Any sequence of characters that are not hyphens.
    # - Step sequences: Sequences of digits separated by commas.
    pattern = fr"{dataset_name}-(?P<gan>.+?)-(?P<step>\d+(?:,\d+)*)"

    print(filenames)
    for filename in filenames:
        match = re.search(pattern, filename)
        if match:
            gan = match.group("gan")
            step = match.group("step")
            print(f"gan={gan}\nstep={step}\n")
        else:
            print('No match found.')

print('May the force be with you.')