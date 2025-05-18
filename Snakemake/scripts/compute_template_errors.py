from snakemake.script import snakemake
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

ap_templates = pd.read_pickle(snakemake.input.ap_templates)

# compute similairty, if more than two tracks, choose the one with highest similarity 
def compute_similarity(df):
    results = []
    for i in range(len(df)):
        track_i = df.iloc[i]['track']
        template_i = df.iloc[i]['template']

        best_match = None
        lowest_mse = float('inf')
        lowest_rmse = float('inf')
        lowest_mae = float('inf')

        for j in range(len(df)):
            if i == j:
                continue

            track_j = df.iloc[j]['track']
            template_j = df.iloc[j]['template']

            # Compute errors
            mse = mean_squared_error(template_i, template_j)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(template_i, template_j)
            results.append({
                'track_1': track_i,
                'track_2': track_j,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'two_tracks': len(df) == 2
            })

            if mse <= lowest_mse:
                lowest_mse = mse
                lowest_rmse = rmse
                lowest_mae = mae
                best_match = track_j

        results.append({
            'track_1': track_i,
            'track_2': 'Best Match',
            'mse': lowest_mse,
            'rmse': lowest_rmse,
            'mae': lowest_mae,
            'best_match_track': best_match,
            'two_tracks': len(df) == 2
        })

    return pd.DataFrame(results)

result_template_similarity = compute_similarity(ap_templates)

result_template_similarity.to_pickle(snakemake.output.template_similarity)