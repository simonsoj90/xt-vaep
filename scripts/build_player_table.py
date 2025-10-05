from pathlib import Path
import argparse
import pandas as pd
from football_analytics.metrics.progressive import add_progressive_flags,player_progressive_value
from football_analytics.metrics.retention import player_retention
from football_analytics.metrics.duels import player_duels
from football_analytics.features.minutes import minutes_from_subs
from football_analytics.features.roles import player_roles

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
    out["minutes_played"]=out["minutes_played"].fillna(0)
    out["xT_per90"]=out["xT_total"]/out["minutes_played"].replace(0,1)*90
    return out

def agg_vaep(ev: pd.DataFrame, mins: pd.DataFrame) -> pd.DataFrame:
    k=keycols(ev)
    g=ev.groupby(k,dropna=False)["vaep_delta"].sum().rename("vaep_total").reset_index()
    out=g.merge(mins,how="left",on=[c for c in k if c!="player"]+["player"])
    out["minutes_played"]=out["minutes_played"].fillna(0)
    out["vaep_per90"]=out["vaep_total"]/out["minutes_played"].replace(0,1)*90
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
    try:
        ev_v=pd.read_parquet(a.events_with_vaep)
        va=agg_vaep(ev_v,mins)
        base=base.merge(va,how="left",on=keycols(ev_xt)+["minutes_played"])
    except Exception:
        pass
    base.to_parquet(a.out_parquet)
    base.to_csv(a.out_csv,index=False)

if __name__=="__main__":
    main()

