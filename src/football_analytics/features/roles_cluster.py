import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def build_role_clusters(ev: pd.DataFrame, k: int = 8, min_actions: int = 200, random_state: int = 0) -> pd.DataFrame:
    x=ev.copy()
    x["event_type"]=x["event_type"].astype(str)
    x["is_pass"]=x["event_type"].eq("Pass").astype(bool)
    x["is_carry"]=x["event_type"].eq("Carry").astype(bool)
    x["is_shot"]=x["event_type"].eq("Shot").astype(bool)
    if {"pass_end_x","pass_end_y","carry_end_x","carry_end_y"}.issubset(x.columns):
        ex=np.where(x["is_pass"],x["pass_end_x"],np.where(x["is_carry"],x["carry_end_x"],np.nan))
        ey=np.where(x["is_pass"],x["pass_end_y"],np.where(x["is_carry"],x["carry_end_y"],np.nan))
        x["end_x"]=pd.to_numeric(pd.Series(ex,index=x.index),errors="coerce")
        x["end_y"]=pd.to_numeric(pd.Series(ey,index=x.index),errors="coerce")
    else:
        x["end_x"]=np.nan
        x["end_y"]=np.nan
    x["x"]=pd.to_numeric(x.get("x"),errors="coerce")
    x["y"]=pd.to_numeric(x.get("y"),errors="coerce")
    dx=(x["end_x"]-x["x"])
    dy=(x["end_y"]-x["y"])
    fwd_long=((dx.abs().fillna(0)>=10.0) | (dy.abs().fillna(0)>=5.0))
    x["is_progressive"]=((x["is_pass"]|x["is_carry"]) & fwd_long).astype(int)
    g=x.groupby("player",dropna=False)
    feat=g.agg(
        mx=("x","median"),
        my=("y","median"),
        pass_share=("is_pass","mean"),
        carry_share=("is_carry","mean"),
        shot_share=("is_shot","mean"),
        prog_share=("is_progressive","mean"),
        actions=("event_type","size")
    ).reset_index()
    feat=feat[feat["player"].notna()]
    feat=feat[feat["actions"]>=min_actions].copy()
    Z=feat[["mx","my","pass_share","carry_share","shot_share","prog_share"]].to_numpy(dtype=np.float32)
    Z=np.nan_to_num(Z,copy=False)
    Z=StandardScaler().fit_transform(Z)
    km=KMeans(n_clusters=k,random_state=random_state,n_init="auto").fit(Z)
    out=feat[["player"]].copy()
    out["role_cluster"]=km.labels_
    return out
