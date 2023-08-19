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

    # Configuration file
    parser = get_parser()
    args, unknown = parser.parse_known_args()

    # List of folders
    reports_dir = args.save_dir
    dataset_name = args.dataset_name
    folders = os.listdir(reports_dir)

    # Map datasets to their respective classes
    dataset_classes = {
        "pneumoniamnist": ["normal", "pneumonia"]
        # add other datasets here
    }

    # Create an empty dataframe with the desired columns
    columns = ["folder", "gan", "step", "ACC"]
    for dataset, classes in dataset_classes.items():
        for cls in classes:
            columns.append(f"ACC {cls}")
    df = pd.DataFrame(columns=columns)

    pattern = rf"{dataset_name}-(?P<gan>.+?)-(?P<step>\d+(?:,\d+)*)"

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
                print(f"gan={gan}\nstep={step}\n")
                data = {
                    "folder": folder,
                    "gan": gan,
                    "step": step,
                    "ACC": results_df["ACC"].iloc[0]
                }
            else:
                data = {
                    "folder": folder,
                    "gan": "",
                    "step": "",
                    "ACC": results_df["ACC"].iloc[0]
                }

            for cls in dataset_classes[dataset_name]:
                data[f"ACC {cls}"] = results_df[f"ACC {cls}"].iloc[0]

            # Append to the main dataframe
            df.loc[len(df)] = data
        except:
            print("ERROR in folder:")
            print(folder)
            continue

    # Save the dataframe to a new Excel file
    df.to_excel(os.path.join(reports_dir, f"{dataset_name}_overall_reports.xlsx"), index=False)

    print("May the force be with you!")