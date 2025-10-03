import numpy as np
import pandas as pd

def player_retention(df: pd.DataFrame) -> pd.DataFrame:
    p=df[df["event_type"].isin(["Pass","Carry"])].copy()
    p["kept"]=p["xT_delta"].fillna(0)>=0
    p["up"]=p["under_pressure"].fillna(False)
    actions=p.groupby("player",dropna=False).size().rename("actions").reset_index()
    retention=p.groupby("player",dropna=False)["kept"].mean().rename("retention_rate").reset_index()
    retention_up=p.loc[p["up"]].groupby("player",dropna=False)["kept"].mean().rename("retention_rate_under_pressure").reset_index()
    out=actions.merge(retention,how="left",on="player").merge(retention_up,how="left",on="player")
    return out


