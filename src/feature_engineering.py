# src/feature_engineering.py

import pandas as pd
import numpy as np


def compute_slope(series: np.ndarray) -> float:
    """
    Compute slope using simple linear regression (least squares).
    """
    x = np.arange(len(series))
    y = series

    # Avoid issues with constant values
    if len(set(y)) == 1:
        return 0.0

    slope = np.polyfit(x, y, 1)[0]
    return slope


def create_rolling_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.sort_values(by=['engine_id', 'cycle'])
    
    sensor_cols = [col for col in df.columns if 'sensor_' in col]
    
    feature_dfs = []

    for sensor in sensor_cols:
        grouped = df.groupby('engine_id')[sensor]

        temp = pd.DataFrame({
            f'{sensor}_mean': grouped.rolling(window).mean().reset_index(level=0, drop=True),
            f'{sensor}_std': grouped.rolling(window).std().reset_index(level=0, drop=True),
            f'{sensor}_min': grouped.rolling(window).min().reset_index(level=0, drop=True),
            f'{sensor}_max': grouped.rolling(window).max().reset_index(level=0, drop=True),
            f'{sensor}_slope': grouped.rolling(window).apply(
                lambda x: compute_slope(x.values), raw=False
            ).reset_index(level=0, drop=True)
        })

        feature_dfs.append(temp)

    # Concatenate all features at once
    features = pd.concat(feature_dfs, axis=1)

    df = pd.concat([df, features], axis=1)

    return df


def drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NaN values (caused by rolling window).
    """
    df = df.dropna()
    return df