import numpy as np
import pandas as pd

def add_end_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy()
    is_pass=x["event_type"].astype(str).eq("Pass")
    is_carry=x["event_type"].astype(str).eq("Carry")
    x["end_x"]=np.where(is_pass,x.get("pass_end_x"),np.where(is_carry,x.get("carry_end_x"),np.nan))
    x["end_y"]=np.where(is_pass,x.get("pass_end_y"),np.where(is_carry,x.get("carry_end_y"),np.nan))
    return x

def build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    x=add_end_coordinates(df)
    for c in ["x","y","end_x","end_y","pass_length","pass_angle","duration"]:
        if c in x.columns:
            x[c]=pd.to_numeric(x[c],errors="coerce")
    x["dx"]=x["end_x"]-x["x"]
    x["dy"]=x["end_y"]-x["y"]
    x["dist"]=np.sqrt((x["dx"]**2)+(x["dy"]**2))
    x["angle"]=np.arctan2(x["dy"],x["dx"])
    x["speed"]=np.where(x["duration"].fillna(0)>0,x["dist"]/x["duration"].fillna(0),0.0)
    x["is_home"]=x.apply(lambda r: 1 if str(r.get("team"))==str(r.get("home_team")) else 0,axis=1)
    x["under_pressure"]=x.get("under_pressure").fillna(False).astype(int)
    t=x["event_type"].astype(str)
    d=pd.get_dummies(t, prefix="act")
    feats=pd.concat([x[["x","y","end_x","end_y","dx","dy","dist","angle","speed","pass_length","pass_angle","duration","under_pressure","period","minute","is_home"]].fillna(0),d],axis=1)
    return feats
