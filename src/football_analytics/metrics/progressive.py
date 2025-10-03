import numpy as np
import pandas as pd

def add_progressive_flags(df: pd.DataFrame, pitch_w: float = 120.0, pitch_h: float = 80.0) -> pd.DataFrame:
    g=np.sqrt((pitch_w-df["x"])**2+(pitch_h/2-df["y"])**2)
    end_x=np.where(df["event_type"].eq("Pass"),df["pass_end_x"],np.where(df["event_type"].eq("Carry"),df["carry_end_x"],np.nan))
    end_y=np.where(df["event_type"].eq("Pass"),df["pass_end_y"],np.where(df["event_type"].eq("Carry"),df["carry_end_y"],np.nan))
    ge=np.sqrt((pitch_w-end_x)**2+(pitch_h/2-end_y)**2)
    d=g-ge
    m=(df["event_type"].isin(["Pass","Carry"]))&pd.notna(end_x)&pd.notna(end_y)
    prog=(d>=10)|((end_x-df["x"])>=0.25*(pitch_w/2))
    out=df.copy()
    out["progressive"]=False
    out.loc[m&prog,"progressive"]=True
    return out

def player_progressive_value(df: pd.DataFrame) -> pd.DataFrame:
    x=df[df["progressive"]]
    g=x.groupby("player",dropna=False)["xT_delta"].sum().rename("progressive_xT_total")
    mins=(df["minute"].max()+1) if "minute" in df.columns else 90
    per90=(g/(mins/90)).rename("progressive_xT_per90")
    return pd.concat([g,per90],axis=1).reset_index()
