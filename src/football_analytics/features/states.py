import pandas as pd

def ensure_basic_event_fields(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy()
    et=x["event_type"] if "event_type" in x.columns else pd.Series(index=x.index,dtype=object)
    et=et.copy()
    pm=x.get("pass_end_x").notna() & x.get("pass_end_y").notna() if "pass_end_x" in x.columns and "pass_end_y" in x.columns else pd.Series(False,index=x.index)
    cm=x.get("carry_end_x").notna() & x.get("carry_end_y").notna() if "carry_end_x" in x.columns and "carry_end_y" in x.columns else pd.Series(False,index=x.index)
    is_shot=pd.Series(False,index=x.index)
    if "type" in x.columns:
        t=x["type"].apply(lambda d: d.get("name") if isinstance(d,dict) else d)
        is_shot=t.astype(str).str.lower().eq("shot")
    elif "shot_outcome" in x.columns:
        so=x["shot_outcome"]
        is_shot=so.notna() & ~so.astype(str).str.lower().isin(["nan","none",""])
    else:
        cols=[c for c in x.columns if str(c).startswith("shot_")]
        is_shot=x[cols].notna().any(axis=1) if cols else pd.Series(False,index=x.index)
    et.loc[pm]="Pass"
    et.loc[cm]="Carry"
    et.loc[~pm & ~cm & is_shot]="Shot"
    x["event_type"]=et
    if ("event_outcome" not in x.columns) or (x["event_outcome"].notna().sum()==0):
        eo=pd.Series(index=x.index,dtype=object)
        if "shot_outcome" in x.columns:
            so=x["shot_outcome"]
            m=so.notna() & ~so.astype(str).str.lower().isin(["nan","none",""])
            eo.loc[m]=so.loc[m].astype(str)
        if "outcome" in x.columns:
            oc=x["outcome"]
            m=oc.notna() & ~oc.astype(str).str.lower().isin(["nan","none",""])
            eo.loc[m]=oc.loc[m].astype(str)
        x["event_outcome"]=eo
    return x

