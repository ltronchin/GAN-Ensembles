import sys
sys.path.extend([
    "./"
])

import pandas as pd
import os
import argparse
import re
import numpy as np

from src.general_utils import util_general
from src.general_utils import util_path

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-save", "--save_dir", type=str, default="./reports/")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="pneumoniamnist")
    return parser

if __name__ == "__main__":

    # Configuration file
    parser = get_parser()
    args, unknown = parser.parse_known_args()

    # List of folders
    reports_dir = args.save_dir
    dataset_name = args.dataset_name
    folders = os.listdir(reports_dir)
    aggr_rule = 'mean'

    # Map datasets to their respective classes
    dataset_classes = {
        "pneumoniamnist": ["normal", "pneumonia"]
        # add other datasets here
    }

    # Create an empty dataframe with the desired columns
    #columns = ["folder", "gan", "step", "ACC", "std ACC"]
    columns = ["folder", "gan", "step", "fitness_name", "cost_name", "eval_backbone", "ACC", "std ACC"]
    for dataset, classes in dataset_classes.items():
        for cls in classes:
            columns.append(f"ACC {cls}")
            columns.append(f"std ACC {cls}")
    df = pd.DataFrame(columns=columns)

    # pattern = rf"{dataset_name}-(?P<gan>.+?)-(?P<step>\d+(?:,\d+)*)"
    pattern = re.compile(
        r'(?P<dataset>\w+)-'
        r'(?P<gan>.+?)-'
        r'(?P<step>[\d,]+)-'
        r'(?P<fitness_name>\w+)-'
        r'(?P<cost_name>\w+)-'
        r'(?P<eval_backbone>[\w_]+)'
    )
    for folder in folders:
        try:
            # Read the results.xlsx file
            results_path = os.path.join(reports_dir, folder, "results.xlsx")
            results_df = pd.read_excel(results_path)

            # Extract ACC values
            match = re.search(pattern, folder)
            if match:
                gan = match.group("gan")
                step = match.group("step")
                fitness_name = match.group("fitness_name")
                cost_name = match.group("cost_name")
                eval_backbone = match.group("eval_backbone")
                print(f"gan={gan}\nstep={step}\n")
                data = {
                    "folder": folder,
                    "gan": gan,
                    "step": step,
                    "fitness_name": fitness_name,
                    "cost_name": cost_name,
                    "eval_backbone": eval_backbone,
                    "ACC": results_df["ACC"].iloc[np.where(results_df == 'mean')[0][0]],
                    "std ACC": results_df["ACC"].iloc[np.where(results_df == 'std')[0][0]]
                }
            else:
                data = {
                    "folder": folder,
                    "gan": "",
                    "step": "",
                    "fitness_name": "",
                    "cost_name": "",
                    "eval_backbone": "",
                    "ACC": results_df["ACC"].iloc[np.where(results_df == 'mean')[0][0]],
                    "std ACC": results_df["ACC"].iloc[np.where(results_df == 'std')[0][0]]
                }

            for cls in dataset_classes[dataset_name]:
                data[f"ACC {cls}"] = results_df[f"ACC {cls}"].iloc[np.where(results_df == 'mean')[0][0]]
                data[f"std ACC {cls}"] = results_df[f"ACC {cls}"].iloc[np.where(results_df == 'std')[0][0]]

            # Append to the main dataframe
            df.loc[len(df)] = data
        except Exception as e:
            print(e)
            print("ERROR in folder:")
            print(folder)
            continue

    # Save the dataframe to a new Excel file
    if args.tag:
        df.to_excel(os.path.join(reports_dir, f"{dataset_name}_overall_reports_{args.tag}_{aggr_rule}.xlsx"), index=False)
    else:
        df.to_excel(os.path.join(reports_dir, f"{dataset_name}_overall_reports_{aggr_rule}.xlsx"), index=False)

    print("May the force be with you!")