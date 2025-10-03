import pandas as pd

def player_duels(df: pd.DataFrame) -> pd.DataFrame:
    d=df[df["event_type"]=="Duel"].copy()
    if "event_outcome" in d.columns:
        d["won"]=d["event_outcome"].astype(str).str.contains("Won", case=False, na=False)
    else:
        d["won"]=False
    g=d.groupby("player",dropna=False)
    n=g.size().rename("duels")
    w=g["won"].mean().rename("duel_win_rate")
    return pd.concat([n,w],axis=1).reset_index()
