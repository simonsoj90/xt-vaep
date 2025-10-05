import pandas as pd
from pathlib import Path

def z(s: pd.Series) -> pd.Series:
    s=pd.to_numeric(s,errors="coerce")
    mu=s.mean()
    sd=s.std(ddof=0)
    return (s-mu)/sd if sd and pd.notna(sd) and sd!=0 else (s-mu)

def pick(df,base):
    if base in df: return base
    for suff in ("_x","_y"):
        c=base+suff
        if c in df: return c
    return None

def main():
    src="data/processed/player_table.parquet"
    out="data/processed/player_table_norm.parquet"
    df=pd.read_parquet(src).copy()
    for k in ("competition_name","season_name"):
        if k not in df: df[k]="ALL"
    mp=pd.to_numeric(df.get("minutes_played"),errors="coerce").fillna(0).replace(0,1)
    xt_tot=pick(df,"xT_total")
    if xt_tot and "xT_per90" not in df:
        df["xT_per90"]=pd.to_numeric(df[xt_tot],errors="coerce")/mp*90
    vaep_tot=pick(df,"vaep_total")
    vaep_p90=pick(df,"vaep_per90")
    if vaep_p90 and "vaep_per90" not in df:
        df["vaep_per90"]=pd.to_numeric(df[vaep_p90],errors="coerce")
    if "vaep_per90" not in df and vaep_tot:
        df["vaep_per90"]=pd.to_numeric(df[vaep_tot],errors="coerce")/mp*90
    if "progressive_actions_per90" not in df and "progressive_actions" in df:
        df["progressive_actions_per90"]=pd.to_numeric(df["progressive_actions"],errors="coerce")/mp*90
    to_norm=["xT_per90","vaep_per90","progressive_xT_per90","progressive_actions_per90"]
    for col in to_norm:
        if col in df:
            df[f"{col}_norm"]=df.groupby(["competition_name","season_name"])[col].transform(z)
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(out)
    df.to_csv("data/processed/player_table_norm.csv",index=False)

if __name__=="__main__":
    main()
