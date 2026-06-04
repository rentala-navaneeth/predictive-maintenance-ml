import pandas as pd

def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) for each engine cycle.
    """
    max_cycle = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']

    df = df.merge(max_cycle, on='engine_id', how='left')

    df['RUL'] = df['max_cycle'] - df['cycle']

    df.drop(columns=['max_cycle'], inplace=True)
    
    return df


def create_binary_label(df: pd.DataFrame, threshold: int = 30) -> pd.DataFrame:
    """
    Convert RUL into binary classification label.
    
    Label = 1 → Failure within next `threshold` cycles
    Label = 0 → Otherwise
    """
    df['label'] = (df['RUL'] <= threshold).astype(int)
    return df
