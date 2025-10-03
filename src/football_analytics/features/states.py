import pandas as pd

def ensure_basic_event_fields(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy()
    has_type=("event_type" in x.columns) and (x["event_type"].notna().any())
    if not has_type:
        et=pd.Series(index=x.index,dtype=object)
        pm=(x.get("pass_end_x").notna())&(x.get("pass_end_y").notna())
        cm=(x.get("carry_end_x").notna())&(x.get("carry_end_y").notna())
        sm=pd.Series(False,index=x.index)
        if "shot_outcome" in x.columns:
            so=x["shot_outcome"]
            sm=so.notna() & so.astype(str).str.lower().ne("nan")
        else:
            shot_cols=[c for c in x.columns if c.startswith("shot_")]
            if shot_cols:
                sm=x[shot_cols].notna().any(axis=1)
        et[pm]="Pass"
        et[cm]="Carry"
        et[sm]="Shot"
        x["event_type"]=et
    has_out=("event_outcome" in x.columns) and (x["event_outcome"].notna().any())
    if not has_out:
        eo=pd.Series(index=x.index,dtype=object)
        if "shot_outcome" in x.columns:
            so=x["shot_outcome"]
            m=so.notna() & so.astype(str).str.lower().ne("nan")
            eo[m]=so[m].astype(str)
        if "outcome" in x.columns:
            oc=x["outcome"]
            m=oc.notna() & oc.astype(str).str.lower().ne("nan")
            eo[m]=oc[m].astype(str)
        x["event_outcome"]=eo
    return x
