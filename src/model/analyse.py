import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error

# for reference, the code exports csv files with 4 columns: Date, Stock_symbol, target_return, prediction


# Redifine here, so model.py doesn't have to be run
def get_experiment_name(use_lstm: bool, use_mdgnn: bool, llm_mode: str) -> str:
    if use_lstm and use_mdgnn:
        return f"lstm_mdgnn_llm_{llm_mode}"
    if use_lstm and not use_mdgnn:
        return f"lstm_only_llm_{llm_mode}"
    if use_mdgnn and not use_lstm:
        return f"mdgnn_only_llm_{llm_mode}"
    raise ValueError("USE_LSTM and USE_MDGNN cannot both be False.")


experiment_name = get_experiment_name(use_lstm=True, use_mdgnn=True, llm_mode="mean")
num_splits = 1

output_dir = Path("data/model/output")

results = []
for s in range(1, num_splits + 1):
    csv_path = output_dir / f"preds_{experiment_name}_split_{s}.csv"
    res_df = pd.read_csv(csv_path)

    mse = mean_squared_error(y_true=res_df["target_return"], y_pred=res_df["prediction"])
    r2 = r2_score(y_true=res_df["target_return"], y_pred=res_df["prediction"])
    mae = median_absolute_error(y_true=res_df["target_return"], y_pred=res_df["prediction"])

    results.append({
        "split_idx": s,
        "mse": mse,
        "medae": mae,
        "r2": r2,
    })

metrics_df = pd.DataFrame(results)
metrics_file = output_dir / f"{experiment_name}_metrics.csv"
metrics_df.to_csv(metrics_file, index=False)