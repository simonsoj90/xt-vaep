import numpy as np
import pandas as pd

def add_vaep_delta(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy()
    x["vaep_delta"]=x["vaep_p_score"].fillna(0)-x["vaep_p_concede"].fillna(0)
    return x

def aggregate_players(df: pd.DataFrame) -> pd.DataFrame:
    g=df.groupby("player",dropna=False)
    tot=g["vaep_delta"].sum().rename("vaep_total")
    mins=(df["minute"].max()+1) if "minute" in df.columns else 90
    per90=(tot/(mins/90)).rename("vaep_per90")
    return pd.concat([tot,per90],axis=1).reset_index()
