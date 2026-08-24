# ============================================================
# Data Utilities
# Aircraft Engine RUL Prediction
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# ------------------------------------------------------------
# File Utilities
# ------------------------------------------------------------

def get_file_path(filename):
    """
    Locate a CMAPSS dataset file.

    Checks:
        1. Current working directory
        2. Dataset/ directory
    """
    if os.path.exists(filename):
        return filename

    elif os.path.exists(os.path.join("Dataset", filename)):
        return os.path.join("Dataset", filename)

    raise FileNotFoundError(
        f"Cannot find {filename} in current directory or Dataset/ folder."
    )


# ------------------------------------------------------------
# RUL Calculation
# ------------------------------------------------------------

def add_rul(df):
    """
    Calculate Remaining Useful Life (RUL) for each engine.

    RUL is calculated from the final operating cycle of each
    engine and capped at 125 cycles.
    """
    rul = (
        pd.DataFrame(
            df.groupby("unit_nr")["time_cycles"].max()
        )
        .reset_index()
    )

    rul.columns = ["unit_nr", "max"]

    df = df.merge(
        rul,
        on=["unit_nr"],
        how="left"
    )

    df["RUL"] = df["max"] - df["time_cycles"]

    df["RUL"] = df["RUL"].clip(upper=125)

    df.drop("max", axis=1, inplace=True)

    return df


# ------------------------------------------------------------
# Degradation Classification
# ------------------------------------------------------------

def get_class(rul):
    """
    Assign degradation-state class based on RUL.

    Returns:
        1 -> Degraded (RUL <= 50)
        0 -> Healthy  (RUL > 50)
    """
    return 1 if rul <= 50 else 0


# ------------------------------------------------------------
# Operating-Condition Clustering
# ------------------------------------------------------------

def cluster_operating_conditions(
    df,
    n_clusters=6,
    dataset_name=None
):
    """
    Cluster operating conditions using K-Means.

    FD001 and FD003 contain a single operating condition,
    so K-Means is bypassed and regime 0 is assigned.

    FD002 and FD004 use K-Means on:
        setting_1
        setting_2
        setting_3

    Returns:
        df
        KMeans model or integer 1 for a single-regime dataset
    """

    if dataset_name in ["FD001", "FD003"]:
        df["regime"] = 0
        return df, 1

    settings = df[
        ["setting_1", "setting_2", "setting_3"]
    ].values

    if np.std(settings) < 0.01:
        df["regime"] = 0
        return df, 1

    km = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    df["regime"] = km.fit_predict(settings)

    return df, km


# ------------------------------------------------------------
# Sequence Generation
# ------------------------------------------------------------

def generate_sequences(
    df,
    seq_length,
    feature_cols,
    regime_col="regime"
):
    """
    Generate all overlapping temporal windows.

    Returns:
        sequences
        classification labels
        RUL values
        regime sequences
    """

    seqs = []
    labels = []
    ruls = []
    regimes = []

    for unit in df["unit_nr"].unique():

        unit_data = df[
            df["unit_nr"] == unit
        ]

        data_matrix = unit_data[
            feature_cols
        ].values

        label_array = unit_data[
            "label"
        ].values

        rul_array = unit_data[
            "RUL"
        ].values

        reg_array = unit_data[
            regime_col
        ].values

        for i in range(
            len(unit_data) - seq_length + 1
        ):

            seqs.append(
                data_matrix[
                    i:i + seq_length
                ]
            )

            labels.append(
                label_array[
                    i + seq_length - 1
                ]
            )

            ruls.append(
                rul_array[
                    i + seq_length - 1
                ]
            )

            regimes.append(
                reg_array[
                    i:i + seq_length
                ]
            )

    return (
        np.array(seqs),
        np.array(labels),
        np.array(ruls),
        np.array(regimes)
    )


# ------------------------------------------------------------
# Weighted Sequence Generation
# ------------------------------------------------------------

def generate_sequences_weighted(
    df,
    seq_length,
    feature_cols,
    regime_col="regime"
):
    """
    Generate overlapping sequences with curriculum weights.

    Weighting:
        RUL <= 30       -> 2.5
        30 < RUL <= 65  -> 2.0
        RUL > 65        -> 1.2
    """

    seqs = []
    labels = []
    ruls = []
    weights = []
    regimes = []

    for unit in df["unit_nr"].unique():

        unit_data = df[
            df["unit_nr"] == unit
        ]

        data_matrix = unit_data[
            feature_cols
        ].values

        label_array = unit_data[
            "label"
        ].values

        rul_array = unit_data[
            "RUL"
        ].values

        reg_array = unit_data[
            regime_col
        ].values

        for i in range(
            len(unit_data) - seq_length + 1
        ):

            seqs.append(
                data_matrix[
                    i:i + seq_length
                ]
            )

            labels.append(
                label_array[
                    i + seq_length - 1
                ]
            )

            rul_val = rul_array[
                i + seq_length - 1
            ]

            ruls.append(rul_val)

            regimes.append(
                reg_array[
                    i:i + seq_length
                ]
            )

            if rul_val <= 30:
                weight = 2.5

            elif rul_val <= 65:
                weight = 2.0

            else:
                weight = 1.2

            weights.append(weight)

    return (
        np.array(seqs),
        np.array(labels),
        np.array(ruls),
        np.array(weights),
        np.array(regimes)
    )


# ------------------------------------------------------------
# NASA RUL Score
# ------------------------------------------------------------

def nasa_score(y_true, y_pred):
    """
    Calculate the NASA asymmetric scoring function.

    Under-prediction and over-prediction receive different
    exponential penalties.
    """

    d = y_pred - y_true

    scores = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1,
        np.exp(d / 10.0) - 1
    )

    return float(np.sum(scores))
