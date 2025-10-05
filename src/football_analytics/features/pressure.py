import pandas as pd
import numpy as np

def add_pressure_features(ev: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    x=ev.sort_values(["match_id","time_seconds","period"]).copy()
    x["under_pressure"]=x.get("under_pressure",False).astype(bool)
    pc=x.groupby(["match_id","team"])["under_pressure"].rolling(window,min_periods=1).sum().reset_index(level=[0,1],drop=True)
    pr=x.groupby(["match_id","team"])["under_pressure"].rolling(window,min_periods=1).mean().reset_index(level=[0,1],drop=True)
    x["press_count"]=pd.to_numeric(pc,errors="coerce").fillna(0).astype("float32")
    x["press_rate"]=pd.to_numeric(pr,errors="coerce").fillna(0).astype("float32")
    return x
