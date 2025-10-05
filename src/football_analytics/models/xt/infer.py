import numpy as np
import pandas as pd
from .grid import XTGrid

class XTModel:
    def __init__(self, n_x: int = 16, n_y: int = 12, gamma: float = 0.97):
        self.grid=XTGrid(n_x,n_y,gamma)

    def fit(self, events: pd.DataFrame) -> None:
        self.grid.fit(events)

    def value_events(self, events: pd.DataFrame) -> pd.DataFrame:
        df=events.copy()
        is_pass=df["event_type"].astype(str).eq("Pass")
        is_carry=df["event_type"].astype(str).eq("Carry")
        start_v=self.grid.value(df["x"].to_numpy(),df["y"].to_numpy())
        end_x=np.where(is_pass,df.get("pass_end_x"),np.where(is_carry,df.get("carry_end_x"),np.nan))
        end_y=np.where(is_pass,df.get("pass_end_y"),np.where(is_carry,df.get("carry_end_y"),np.nan))
        end_v=np.where(np.isfinite(end_x) & np.isfinite(end_y), self.grid.value(end_x,end_y), np.nan)
        delta=np.where(is_pass|is_carry, np.nan_to_num(end_v, nan=0.0)-np.nan_to_num(start_v, nan=0.0), 0.0)
        df["xT_start"]=start_v
        df["xT_end"]=np.where(is_pass|is_carry, end_v, np.nan)
        df["xT_delta"]=delta
        return df


def aggregate_players(df:pd.DataFrame)->pd.DataFrame:
    g=df.groupby("player",dropna=False)
    out=g["xT_delta"].sum().to_frame("xT_total")
    minutes=(df["minute"].max()+1) if "minute" in df.columns else 90
    out["xT_per90"]=out["xT_total"]/(minutes/90)
    return out.reset_index()

