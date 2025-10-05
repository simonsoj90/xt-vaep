import pandas as pd
import numpy as np

def _safe_end_coords(x: pd.DataFrame):
    if "end_x" in x.columns and "end_y" in x.columns:
        return pd.to_numeric(x["end_x"],errors="coerce"), pd.to_numeric(x["end_y"],errors="coerce")
    is_pass=x["event_type"].astype(str).eq("Pass") if "event_type" in x.columns else pd.Series(False,index=x.index)
    is_carry=x["event_type"].astype(str).eq("Carry") if "event_type" in x.columns else pd.Series(False,index=x.index)
    px=x["pass_end_x"] if "pass_end_x" in x.columns else np.nan
    py=x["pass_end_y"] if "pass_end_y" in x.columns else np.nan
    cx=x["carry_end_x"] if "carry_end_x" in x.columns else np.nan
    cy=x["carry_end_y"] if "carry_end_y" in x.columns else np.nan
    ex=np.where(is_pass,px,np.where(is_carry,cx,np.nan))
    ey=np.where(is_pass,py,np.where(is_carry,cy,np.nan))
    return pd.to_numeric(pd.Series(ex,index=x.index),errors="coerce"), pd.to_numeric(pd.Series(ey,index=x.index),errors="coerce")

def add_progressive_flags(df: pd.DataFrame, min_dx: float = 10.0, min_dy: float = 5.0) -> pd.DataFrame:
    x=df.copy()
    x["x"]=pd.to_numeric(x.get("x"),errors="coerce")
    x["y"]=pd.to_numeric(x.get("y"),errors="coerce")
    end_x,end_y=_safe_end_coords(x)
    is_pass=x["event_type"].astype(str).eq("Pass")
    is_carry=x["event_type"].astype(str).eq("Carry")
    dx=(end_x-x["x"]).fillna(0.0)
    dy=(end_y-x["y"]).fillna(0.0)
    forward=dx>0
    long_enough=(dx.abs()>=min_dx)|(dy.abs()>=min_dy)
    x["is_progressive"]=(is_pass|is_carry)&forward&long_enough
    return x

def player_progressive_value(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    m=df["is_progressive"].fillna(False)
    tot=df.loc[m].groupby(keys,dropna=False)["xT_delta"].sum().rename("progressive_xT_total").reset_index()
    cnt=df.groupby(keys,dropna=False)["is_progressive"].sum().rename("progressive_actions").reset_index()
    out=tot.merge(cnt,how="outer",on=keys).fillna({"progressive_xT_total":0.0,"progressive_actions":0})
    return out

