import pandas as pd
import numpy as np

def minutes_from_subs(ev: pd.DataFrame) -> pd.DataFrame:
    e=ev.copy()
    e["minute"]=pd.to_numeric(e["minute"],errors="coerce").fillna(0)
    keys=["competition_id","season_id","competition_name","season_name","match_id","team"]
    for k in ["competition_id","season_id","competition_name","season_name"]:
        if k not in e.columns:
            e[k]=None
    rows=[]
    for _,g in e.groupby(keys,dropna=False):
        ml=int(g["minute"].max())+1
        ml=int(np.clip(ml,90,120))
        subs=g[g["event_type"].astype(str)=="Substitution"]
        on_map={}
        off_map={}
        for _,r in subs.iterrows():
            off_player=str(r.get("player")) if pd.notna(r.get("player")) else None
            on_player=str(r.get("substitution_replacement")) if pd.notna(r.get("substitution_replacement")) else None
            m=int(r["minute"])
            if off_player and off_player!="nan":
                off_map[off_player]=min(off_map.get(off_player,ml),m)
            if on_player and on_player!="nan":
                on_map[on_player]=min(on_map.get(on_player,ml),m)
        players=set(g["player"].dropna().astype(str).tolist())|set(on_map.keys())|set(off_map.keys())
        for p in players:
            on=on_map.get(p,0)
            off=off_map.get(p,ml)
            mins=int(np.clip(off-on,0,120))
            rows.append(tuple(list(g[keys].iloc[0].values)+[p,mins]))
    per_match=pd.DataFrame(rows,columns=keys+["player","minutes_played"])
    per_match=per_match.groupby(keys+["player"],dropna=False,as_index=False)["minutes_played"].sum()
    season_keys=["competition_id","season_id","competition_name","season_name","player"]
    season=per_match.groupby(season_keys,dropna=False,as_index=False)["minutes_played"].sum()
    return season
