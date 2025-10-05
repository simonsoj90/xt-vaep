# from pathlib import Path
# import argparse
# import pandas as pd
# from football_analytics.models.vaep.value import add_vaep_delta,aggregate_players

# def parse_args():
#     p=argparse.ArgumentParser()
#     p.add_argument("--in_events",default="data/processed/events_with_xt.parquet")
#     p.add_argument("--k",type=int,default=5)
#     p.add_argument("--models_dir",default="models")
#     p.add_argument("--out_events",default="data/processed/events_with_vaep.parquet")
#     p.add_argument("--out_players",default="data/processed/player_vaep.parquet")
#     return p.parse_args()

# def main():
#     a=parse_args()
#     Path(a.out_events).parent.mkdir(parents=True,exist_ok=True)
#     Path(a.out_players).parent.mkdir(parents=True,exist_ok=True)
#     Path(a.models_dir).mkdir(parents=True,exist_ok=True)
#     ev=pd.read_parquet(a.in_events)
#     ev2,m1,m2,ls,lc=fit_vaep_lite(ev,k=a.k,models_dir=a.models_dir)
#     ev2=add_vaep_delta(ev2)
#     ev2.to_parquet(a.out_events)
#     pl=aggregate_players(ev2)
#     pl.to_parquet(a.out_players)

# if __name__=="__main__":
#     main()
