import pandas as pd
import numpy as np

def player_roles(ev: pd.DataFrame, pitch_w: float = 120.0, pitch_h: float = 80.0) -> pd.DataFrame:
    x=ev[ev["event_type"].isin(["Pass","Carry"])].copy()
    for c in ["x","y"]:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    g=x.groupby("player",dropna=False)[["x","y"]].median().reset_index().rename(columns={"x":"mx","y":"my"})
    def bucket(row):
        x=row["mx"]; y=row["my"]
        if pd.isna(x) or pd.isna(y): return "UNK"
        lx=pitch_w/3; rx=2*pitch_w/3
        ly=pitch_h/3; ry=2*pitch_h/3
        horiz="D" if x<lx else ("M" if x<rx else "A")
        vert="L" if y<ly else ("C" if y<ry else "R")
        m={"DL":"LB","DC":"CB","DR":"RB","ML":"LM","MC":"CM","MR":"RM","AL":"LW","AC":"AM","AR":"RW"}
        return m.get(horiz+vert,"UNK")
    g["role"]=g.apply(bucket,axis=1)
    return g[["player","role"]]
