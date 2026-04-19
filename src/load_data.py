import pandas as pd

def load_cmapss_data(file_path: str) -> pd.DataFrame:
    """
    Load CMAPSS dataset and assign column names.
    """

    cols = ['engine_id', 'cycle']
    cols += [f'op_setting_{i}' for i in range(1, 4)]
    cols += [f'sensor_{i}' for i in range(1, 22)]

    df = pd.read_csv(
        file_path,
        sep=' ',
        header=None
    )

    # Drop last two empty columns
    df = df.iloc[:, :26]
    df.columns = cols

    return df