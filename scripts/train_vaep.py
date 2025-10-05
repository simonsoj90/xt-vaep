import pandas as pd
from football_analytics.features.labels import add_possession_ids, vaep_labels
from football_analytics.features.vaep_features import build_vaep_features
from football_analytics.models.vaep.train import fit_vaep, value_events

def main():
    ev=pd.read_parquet("data/processed/events_with_xt.parquet")
    ev=add_possession_ids(ev)
    lab=vaep_labels(ev)
    feats=build_vaep_features(ev)
    used=fit_vaep(ev,lab,feats,"models/vaep")
    value_events(ev,feats,used,"data/processed/events_with_vaep.parquet")

if __name__=="__main__":
    main()
