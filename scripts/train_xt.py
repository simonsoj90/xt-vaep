from pathlib import Path
import argparse
import pandas as pd
from football_analytics.models.xt.infer import XTModel,aggregate_players
from football_analytics.features.states import ensure_basic_event_fields

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--in_parquet",default="data/interim/events_all.feather")
    p.add_argument("--grid_out",default="data/processed/xt_grid.npz")
    p.add_argument("--events_out",default="data/processed/events_with_xt.parquet")
    p.add_argument("--players_out",default="data/processed/player_xt.parquet")
    p.add_argument("--nx",type=int,default=16)
    p.add_argument("--ny",type=int,default=12)
    return p.parse_args()

def main():
    a=parse_args()
    Path(a.events_out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.grid_out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.players_out).parent.mkdir(parents=True,exist_ok=True)
    ev=pd.read_feather(a.in_parquet)
    ev=ensure_basic_event_fields(ev)
    for c in ["x","y","pass_end_x","pass_end_y","carry_end_x","carry_end_y"]:
        if c in ev.columns:
            ev[c]=pd.to_numeric(ev[c],errors="coerce")
    m=XTModel(a.nx,a.ny)
    m.fit(ev)
    ve=m.value_events(ev)
    m.grid.save(a.grid_out)
    ve.to_parquet(a.events_out)
    pl=aggregate_players(ve)
    pl.to_parquet(a.players_out)

if __name__=="__main__":
    main()
