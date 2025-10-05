from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from football_analytics.metrics.progressive import add_progressive_flags,player_progressive_value
from football_analytics.metrics.retention import player_retention
from football_analytics.metrics.duels import player_duels
from football_analytics.features.minutes import minutes_from_subs
from football_analytics.features.roles import player_roles
from football_analytics.features.roles_cluster import build_role_clusters


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--events_with_xt",default="data/processed/events_with_xt.parquet")
    p.add_argument("--events_with_vaep",default="data/processed/events_with_vaep.parquet")
    p.add_argument("--out_parquet",default="data/processed/player_table.parquet")
    p.add_argument("--out_csv",default="data/processed/player_table.csv")
    return p.parse_args()

def keycols(ev: pd.DataFrame):
    cols=[c for c in ["competition_id","season_id","competition_name","season_name"] if c in ev.columns]
    cols.append("player")
    return cols

def agg_xt(ev: pd.DataFrame, mins: pd.DataFrame) -> pd.DataFrame:
    k=keycols(ev)
    g=ev.groupby(k,dropna=False)["xT_delta"].sum().rename("xT_total").reset_index()
    out=g.merge(mins,how="left",on=[c for c in k if c!="player"]+["player"])
    out["minutes_played"]=out["minutes_played"].fillna(0).astype(float)
    out["xT_per90"]=np.where(out["minutes_played"]>0, out["xT_total"]/out["minutes_played"]*90, np.nan)
    return out

def agg_vaep(ev: pd.DataFrame, mins: pd.DataFrame) -> pd.DataFrame:
    k=keycols(ev)
    g=ev.groupby(k,dropna=False)["vaep_delta"].sum().rename("vaep_total").reset_index()
    out=g.merge(mins,how="left",on=[c for c in k if c!="player"]+["player"])
    out["minutes_played"]=out["minutes_played"].fillna(0).astype(float)
    out["vaep_per90"]=np.where(out["minutes_played"]>0, out["vaep_total"]/out["minutes_played"]*90, np.nan)
    return out

def main():
    a=parse_args()
    Path(a.out_parquet).parent.mkdir(parents=True,exist_ok=True)
    ev_xt=pd.read_parquet(a.events_with_xt)
    mins=minutes_from_subs(ev_xt)
    ev_xt=add_progressive_flags(ev_xt)
    prog=player_progressive_value(ev_xt, keycols(ev_xt))
    prog=prog.merge(mins,how="left",on=[c for c in keycols(ev_xt) if c!="player"]+["player"])
    prog["minutes_played"]=prog["minutes_played"].fillna(0)
    prog["progressive_xT_per90"]=prog["progressive_xT_total"]/prog["minutes_played"].replace(0,1)*90
    ret=player_retention(ev_xt)
    due=player_duels(ev_xt)
    roles=player_roles(ev_xt)
    xt=agg_xt(ev_xt,mins)
    acts=ev_xt.groupby(keycols(ev_xt),dropna=False).size().rename("actions").reset_index()
    base=xt.merge(acts,how="left",on=keycols(ev_xt))
    base=base.merge(prog,how="left",on=keycols(ev_xt)+["minutes_played"])
    base=base.merge(ret,how="left",on="player").merge(due,how="left",on="player").merge(roles,how="left",on="player")
    base=base[base["player"].notna()]
    base=base[base["player"].astype(str).str.strip().ne("None")]
    
    base["minutes_played"]=pd.to_numeric(base["minutes_played"], errors="coerce").fillna(0).astype(float)
    base["xT_total"]=pd.to_numeric(base["xT_total"], errors="coerce")
    base["vaep_total"]=pd.to_numeric(base.get("vaep_total"), errors="coerce")

    base["xT_per90"]=np.where(base["minutes_played"]>0, base["xT_total"]/base["minutes_played"]*90, np.nan)
    if "vaep_total" in base.columns:
        base["vaep_per90"]=np.where(base["minutes_played"]>0, base["vaep_total"]/base["minutes_played"]*90, np.nan)

    if "progressive_actions" in base.columns:
        base["progressive_actions"]=pd.to_numeric(base["progressive_actions"], errors="coerce").fillna(0)
        base["progressive_actions_per90"]=np.where(base["minutes_played"]>0, base["progressive_actions"]/base["minutes_played"]*90, np.nan)

    for c in ["xT_per90","vaep_per90","progressive_actions_per90","progressive_xT_per90"]:
        if c in base.columns:
            base[c]=pd.to_numeric(base[c], errors="coerce")
    
    rc=build_role_clusters(ev_xt, k=8, min_actions=80)
    base=base.merge(rc, how="left", on="player")

    
    try:
        ev_v=pd.read_parquet(a.events_with_vaep)
        va=agg_vaep(ev_v,mins)
        base=base.merge(va,how="left",on=keycols(ev_xt)+["minutes_played"])
    except Exception:
        pass

    for base_col in ("vaep_total","vaep_per90"):
        alts=[c for c in base.columns if c==base_col or c.startswith(base_col+"_")]
    if alts:
        s=pd.to_numeric(base[alts[0]],errors="coerce")
        for c in alts[1:]:
            s=s.fillna(pd.to_numeric(base[c],errors="coerce"))
            base[base_col]=s
    
    drop=[c for c in base.columns if c.startswith("vaep_total_") or c.startswith("vaep_per90_")]
    base=base.drop(columns=drop,errors="ignore")
    mp=pd.to_numeric(base["minutes_played"],errors="coerce").fillna(0)
    base["xT_per90"]=np.where(mp>0,pd.to_numeric(base["xT_total"],errors="coerce")/mp*90,np.nan)
    if "vaep_total" in base.columns:
        base["vaep_per90"]=np.where(mp>0,pd.to_numeric(base["vaep_total"],errors="coerce")/mp*90,np.nan)
    if "progressive_actions" in base.columns:
        pa=pd.to_numeric(base["progressive_actions"],errors="coerce").fillna(0)
        base["progressive_actions_per90"]=np.where(mp>0,pa/mp*90,np.nan)

    base.to_parquet(a.out_parquet)
    base.to_csv(a.out_csv,index=False)

if __name__=="__main__":
    main()

