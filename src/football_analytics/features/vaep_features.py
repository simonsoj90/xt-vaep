import pandas as pd
import numpy as np
from .pressure import add_pressure_features

def build_vaep_features(ev: pd.DataFrame, n_back: int = 3) -> pd.DataFrame:
    x=add_pressure_features(ev).sort_values(["match_id","time_seconds","period"]).copy()
    x["x"]=pd.to_numeric(x.get("x"),errors="coerce")
    x["y"]=pd.to_numeric(x.get("y"),errors="coerce")
    x["under_pressure"]=x.get("under_pressure",False).astype(bool)
    for i in range(1,n_back+1):
        x[f"dx_{i}"]=(x["x"]-x["x"].shift(i)).astype("float32")
        x[f"dy_{i}"]=(x["y"]-x["y"].shift(i)).astype("float32")
        u=x["under_pressure"].shift(i)
        x[f"up_{i}"]=pd.Series(np.where(u.isna(),0,u.astype(bool).astype(int)),index=x.index).astype("int8")
    x["speed"]=(x["dx_1"].abs()+x["dy_1"].abs()).astype("float32")
    x["vert"]=x["dx_1"].astype("float32")
    xb=pd.cut(x["x"],bins=12,labels=False)
    yb=pd.cut(x["y"],bins=8,labels=False)
    x["x_bin"]=pd.Series(xb,index=x.index).fillna(-1).astype("int16")
    x["y_bin"]=pd.Series(yb,index=x.index).fillna(-1).astype("int16")
    feats=["x_bin","y_bin","speed","vert","press_count","press_rate"]+[f"dx_{i}" for i in range(1,n_back+1)]+[f"dy_{i}" for i in range(1,n_back+1)]+[f"up_{i}" for i in range(1,n_back+1)]
    keep=["match_id","poss_id","team","player","time_seconds","period","event_type"]+feats
    x=x[keep]
    for c in feats:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x[feats]=x[feats].fillna(0).astype("float32")
    return x
