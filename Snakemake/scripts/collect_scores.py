from snakemake.script import snakemake
import pandas as pd
from pathlib import Path

input_files = snakemake.input  
output_files = snakemake.output  

# Get output directory from one of the files
output_dir = Path(output_files[0]).parent
output_dir.mkdir(parents=True, exist_ok=True)

all_metrics = {}

# Collect all scores into per-metric dataframes
for file in input_files:
    df = pd.read_pickle(file)
    dataset_name = Path(file).stem.split('_')[0]  

    for metric in df.index:
        row = df.loc[metric].to_frame().T
        row["Dataset"] = dataset_name
        row.set_index("Dataset", inplace=True)

        if metric not in all_metrics:
            all_metrics[metric] = row
        else:
            all_metrics[metric] = pd.concat([all_metrics[metric], row])

# Save each metric to its corresponding file
for metric, df_metric in all_metrics.items():
    output_file = output_dir / f"{metric}.csv"
    df_metric.round(2).to_csv(output_file)
