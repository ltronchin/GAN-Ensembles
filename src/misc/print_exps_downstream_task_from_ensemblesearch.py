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

if __name__ == "__main__":

    # List of folders
    reports_dir = './reports/'
    dataset_name = 'pneumoniamnist'
    metric_name = 'fid'
    cost_name = 'ratio'

    reports_dir = os.path.join(reports_dir, dataset_name, 'ensemble', f'{metric_name}', f'ensemble_{cost_name}')
    filenames = os.listdir(reports_dir)

    exp_name_pattern = r"ensemble_search_([a-zA-Z0-9]+)-step_\d+-summary_[a-zA-Z0-9]+-cost_name_([a-zA-Z0-9_]+)-([a-zA-Z0-9_]+)_"

    for exp_name in filenames:
        exp_path = os.path.join(reports_dir, exp_name)

        # Check if it's a folder
        if os.path.isdir(exp_path):

            match = re.match(exp_name_pattern, exp_name)
            if match:
                fitness_name, cost_name, eval_backbone = match.groups()
                cost_name = cost_name.split('_')[0]
            else:
                print(f"Warning: Could not parse folder name {exp_name}")
                continue

            # Loop through each file in the folder
            for file_name in os.listdir(exp_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(exp_path, file_name)

                    # Open and read the file
                    with open(file_path, "r") as f:
                        content = f.read()

                    # Use regular expression to extract ensemble
                    ensemble_match = re.search(r"Params: {'ensemble': '([^']+)'}", content)

                    if ensemble_match:
                        ensemble = ensemble_match.group(1)

                        # Separate model names from their step numbers
                        models_with_steps = ensemble.split(',')
                        models_without_steps = []
                        steps_set = set()

                        for model_with_step in models_with_steps:
                            model, step = model_with_step.rsplit('_', 1)
                            models_without_steps.append(model)
                            steps_set.add(step)

                        # Ensure all models have the same step
                        if len(steps_set) != 1:
                            print("Warning: Multiple different steps found.")
                            continue

                        steps = steps_set.pop()

                        # Print the formatted string
                        print(
                            f"dataset_name='pneumoniamnist' config_file='pneumoniamnist.yaml' real_flag='0' fitness_name='{fitness_name}' cost_name='{cost_name}' eval_backbone='{eval_backbone}' gan_models='{','.join(models_without_steps)}' gan_steps='{steps}' gan_models_steps='None'")

print('May the force be with you.')