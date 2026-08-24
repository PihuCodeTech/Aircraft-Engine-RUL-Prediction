# ============================================================
# Data Preprocessing
# Aircraft Engine RUL Prediction
# ============================================================

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# ------------------------------------------------------------
# Condition-Aware Scaler
# ------------------------------------------------------------

class ConditionAwareScaler:
    """
    Condition-Aware Scaler.

    Sensors:
        s_1 ... s_21 are normalized separately within each
        operating regime using a MinMaxScaler.

    Settings:
        setting_1 ... setting_3 are normalized globally.

    This prevents sensor baseline shifts caused by changes
    in altitude and Mach while retaining absolute operating
    condition information from the settings.
    """

    def __init__(self):
        self.scalers = {}
        self.global_scaler = MinMaxScaler()

        self.setting_cols = []
        self.sensor_cols = []

    def fit_transform(
        self,
        df,
        features,
        regime_col="regime"
    ):
        """
        Fit the scaler on training data and transform it.
        """

        result = df.copy()

        self.setting_cols = [
            c for c in features
            if c.startswith("setting_")
        ]

        self.sensor_cols = [
            c for c in features
            if c.startswith("s_")
        ]

        # ----------------------------------------------------
        # Global normalization of operating settings
        # ----------------------------------------------------

        if self.setting_cols:
            result[self.setting_cols] = (
                self.global_scaler.fit_transform(
                    df[self.setting_cols]
                )
            )

        # ----------------------------------------------------
        # Per-regime normalization of sensors
        # ----------------------------------------------------

        for regime in df[regime_col].unique():

            mask = df[regime_col] == regime

            if self.sensor_cols:

                scaler = MinMaxScaler()

                result.loc[
                    mask,
                    self.sensor_cols
                ] = scaler.fit_transform(
                    df.loc[mask, self.sensor_cols]
                )

                self.scalers[regime] = scaler

        return result[features].values

    def transform(
        self,
        df,
        features,
        regime_col="regime"
    ):
        """
        Transform validation/test data using the scalers
        fitted on the training data.
        """

        result = df.copy()

        # ----------------------------------------------------
        # Global settings transformation
        # ----------------------------------------------------

        if self.setting_cols:
            result[self.setting_cols] = (
                self.global_scaler.transform(
                    df[self.setting_cols]
                )
            )

        # ----------------------------------------------------
        # Per-regime sensor transformation
        # ----------------------------------------------------

        for regime in df[regime_col].unique():

            mask = df[regime_col] == regime

            if self.sensor_cols:

                if regime in self.scalers:

                    result.loc[
                        mask,
                        self.sensor_cols
                    ] = self.scalers[regime].transform(
                        df.loc[mask, self.sensor_cols]
                    )

                else:
                    # Use the numerically closest fitted regime
                    closest = min(
                        self.scalers.keys(),
                        key=lambda r: abs(r - regime)
                    )

                    result.loc[
                        mask,
                        self.sensor_cols
                    ] = self.scalers[closest].transform(
                        df.loc[mask, self.sensor_cols]
                    )

        return result[features].values


# ------------------------------------------------------------
# EMA Denoising
# ------------------------------------------------------------

def apply_ema_denoising(
    df,
    features,
    span=5
):
    """
    Apply Exponential Moving Average (EMA) denoising
    to sensor features on an engine-by-engine basis.

    EMA is applied AFTER condition-aware normalization.
    """

    result = df.copy()

    sensor_cols = [
        c for c in features
        if c.startswith("s_")
    ]

    if sensor_cols:

        result[sensor_cols] = (
            result
            .groupby("unit_nr")[sensor_cols]
            .transform(
                lambda x: x.ewm(
                    span=span,
                    adjust=False
                ).mean()
            )
        )

    return result


# ------------------------------------------------------------
# Complete Preprocessing Pipeline
# ------------------------------------------------------------

def preprocess_train_validation(
    train_df,
    val_df,
    features,
    scaler=None,
    ema_span=5
):
    """
    Apply the complete preprocessing pipeline.

    Order:
        1. Condition-aware normalization
        2. EMA denoising

    Returns:
        processed_train_df
        processed_val_df
        scaler
    """

    if scaler is None:
        scaler = ConditionAwareScaler()

        train_df = train_df.copy()
        val_df = val_df.copy()

        train_df[features] = (
            scaler.fit_transform(
                train_df,
                features
            )
        )

        val_df[features] = (
            scaler.transform(
                val_df,
                features
            )
        )

    else:
        train_df = train_df.copy()
        val_df = val_df.copy()

        train_df[features] = (
            scaler.transform(
                train_df,
                features
            )
        )

        val_df[features] = (
            scaler.transform(
                val_df,
                features
            )
        )

    # EMA is intentionally applied AFTER normalization
    train_df = apply_ema_denoising(
        train_df,
        features,
        span=ema_span
    )

    val_df = apply_ema_denoising(
        val_df,
        features,
        span=ema_span
    )

    return train_df, val_df, scaler


# ------------------------------------------------------------
# Test-Set Preprocessing
# ------------------------------------------------------------

def preprocess_test(
    test_df,
    features,
    scaler,
    ema_span=5
):
    """
    Apply the already-fitted training scaler to test data,
    followed by EMA denoising.
    """

    result = test_df.copy()

    result[features] = scaler.transform(
        result,
        features
    )

    result = apply_ema_denoising(
        result,
        features,
        span=ema_span
    )

    return result
