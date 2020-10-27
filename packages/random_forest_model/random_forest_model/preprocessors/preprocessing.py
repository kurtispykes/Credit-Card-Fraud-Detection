import pandas as pd

def rename_id_col(df: pd.DataFrame):
    """Renaming the id column."""
    for col in df.columns:
        if "id" in col:
            df.rename(columns={col: col.replace("-", "_")}, inplace=True)
    return df
