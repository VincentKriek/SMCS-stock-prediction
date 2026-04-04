import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

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

experiment_name = get_experiment_name(use_lstm=True, use_mdgnn=True, llm_mode=None)
num_splits = 1

output_dir = Path("data/model/output")

for s in range(1, num_splits + 1):
    # csv_path = output_dir / f"preds_{experiment_name}_split_{s}.csv"
    # df = pd.read_csv(csv_path)

    # TODO: remove this when using real results
    n = 10
    res_df = pd.DataFrame({
        "Date": pd.date_range(start="2024-01-01", periods=n, freq="D"),
        "Stock_symbol": np.random.choice(["A", "B", "C"], size=n),
        "target_return": np.random.rand(n),
        "prediction": np.random.rand(n),
    })
    print(res_df)

    mse = mean_squared_error(y_true=res_df["target_return"], y_pred=res_df["prediction"])
    r2 = r2_score(y_true=res_df["target_return"], y_pred=res_df["prediction"])

    print(f"Overall scores (Split {s})")
    print(f"- MSE: {mse:.6f}")
    print(f"- R² : {r2:.6f}\n")

    # Print results per stock
    results = []
    groups = res_df.groupby("Stock_symbol")
    for stock, group in groups:
        group_mse = mean_squared_error(y_true=group["target_return"], y_pred=group["prediction"])
        group_r2 = r2_score(y_true=group["target_return"], y_pred=group["prediction"])

        results.append({
            "Stock_symbol": stock,
            "MSE": group_mse,
            "R2": group_r2
        })
    results = pd.DataFrame(results)
    print("Metrics per Stock_symbol:")
    print(results)

    # Print results per day
    results = []
    groups = res_df.groupby("Date")
    for date, group in groups:
        group_mse = mean_squared_error(y_true=group["target_return"], y_pred=group["prediction"])
        group_r2 = r2_score(y_true=group["target_return"], y_pred=group["prediction"]) # note that r2 needs >= 2 values to not be NaN

        results.append({
            "Date": date,
            "MSE": group_mse,
            "R2": group_r2
        })
    results = pd.DataFrame(results)
    print("Metrics per Date:")
    print(results)

    # Plot losses over epochs
    # loss_df = pd.read_csv(f"losses_{experiment_name}_split_{s}")

    # TODO: remove this when using real results
    n = 25
    loss_df = pd.DataFrame({
        "epoch": range(n),
        "train_loss": np.random.rand(n),
        "val_loss": np.random.rand(n),
    })
    print(loss_df)

    plt.figure(figsize=(8, 6))
    plt.plot(loss_df["epoch"], loss_df["train_loss"], label="Train Loss")
    plt.plot(loss_df["epoch"], loss_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.show()


    

    




