from pathlib import Path
import argparse
import pandas as pd
from football_analytics.metrics.progressive import add_progressive_flags,player_progressive_value
from football_analytics.metrics.retention import player_retention
from football_analytics.metrics.duels import player_duels

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--events_with_xt",default="data/processed/events_with_xt.parquet")
    p.add_argument("--players_xt",default="data/processed/player_xt.parquet")
    p.add_argument("--players_vaep",default="data/processed/player_vaep.parquet")
    p.add_argument("--out_parquet",default="data/processed/player_table.parquet")
    p.add_argument("--out_csv",default="data/processed/player_table.csv")
    return p.parse_args()

def main():
    a=parse_args()
    Path(a.out_parquet).parent.mkdir(parents=True,exist_ok=True)
    ev=pd.read_parquet(a.events_with_xt)
    pl_xt=pd.read_parquet(a.players_xt)
    try:
        pl_vaep=pd.read_parquet(a.players_vaep)
    except Exception:
        pl_vaep=pd.DataFrame({"player":[],"vaep_total":[],"vaep_per90":[]})
    ev=add_progressive_flags(ev)
    prog=player_progressive_value(ev)
    ret=player_retention(ev)
    due=player_duels(ev)
    out=pl_xt.merge(prog,how="left",on="player").merge(ret,how="left",on="player").merge(due,how="left",on="player").merge(pl_vaep,how="left",on="player")
    out.to_parquet(a.out_parquet)
    out.to_csv(a.out_csv,index=False)

if __name__=="__main__":
    main()
