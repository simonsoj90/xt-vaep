import argparse, pandas as pd
from pathlib import Path
from football_analytics.models.xt.infer import XTModel

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_parquet",default="data/interim/events_all.feather")
    ap.add_argument("--use_grid",default="data/processed/xt_grid.npz")
    ap.add_argument("--out_events",default="data/processed/events_with_xt.parquet")
    args=ap.parse_args()
    if args.in_parquet.endswith(".feather"):
        ev=pd.read_feather(args.in_parquet)
    else:
        ev=pd.read_parquet(args.in_parquet)
    ev=ev[ev["event_type"].isin(["Pass","Carry","Shot"])].copy()
    m=XTModel(16,12)
    p=Path(args.use_grid)
    if p.exists():
        m.load_grid(str(p))
    else:
        m.fit(ev)
        Path(p.parent).mkdir(parents=True,exist_ok=True)
        m.grid.save(str(p))
    ve=m.value_events(ev)
    Path(args.out_events).parent.mkdir(parents=True,exist_ok=True)
    ve.to_parquet(args.out_events)

if __name__=="__main__":
    main()
