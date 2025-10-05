import numpy as np
import pandas as pd

def add_possession_ids(df: pd.DataFrame) -> pd.DataFrame:
    x=df.sort_values(["match_id","period","time_seconds","minute","second"],na_position="last").copy()
    x["team"]=x["team"].astype(str)
    turn=(x["match_id"].ne(x["match_id"].shift())|
          x["period"].ne(x["period"].shift())|
          x["team"].ne(x["team"].shift())|
          x["possession"].ne(x["possession"].shift()))
    x["poss_id"]=turn.cumsum()
    return x

def vaep_labels(df: pd.DataFrame) -> pd.DataFrame:
    x=add_possession_ids(df)
    is_goal=(x["event_type"].astype(str).eq("Shot") &
             x["event_outcome"].astype(str).str.contains("goal",case=False,na=False)).astype(int)
    poss=x.groupby(["match_id","poss_id","team"],dropna=False)["team"].size().reset_index(name="n")
    has_goal=is_goal.groupby([x["match_id"],x["poss_id"],x["team"]],dropna=False).max().reset_index(name="has_goal")
    poss=poss.merge(has_goal,on=["match_id","poss_id","team"],how="left").fillna({"has_goal":0})
    poss=poss.sort_values(["match_id","poss_id"])
    poss["next_poss_id"]=poss.groupby("match_id")["poss_id"].shift(-1)
    poss["next_poss_team"]=poss.groupby("match_id")["team"].shift(-1)
    poss["opp_next"]=(poss["next_poss_team"].astype(str)!=poss["team"].astype(str)).astype(int)
    nxt=poss.merge(
        poss[["match_id","poss_id","has_goal"]].rename(columns={"poss_id":"next_poss_id","has_goal":"next_has_goal"}),
        on=["match_id","next_poss_id"],how="left"
    )
    nxt["y_score"]=poss["has_goal"].astype(int)
    nxt["y_concede"]=np.where(nxt["opp_next"].eq(1),nxt["next_has_goal"].fillna(0).astype(int),0)
    lab=nxt[["match_id","poss_id","y_score","y_concede"]]
    out=x.merge(lab,on=["match_id","poss_id"],how="left")
    out["y_score"]=out["y_score"].fillna(0).astype(int)
    out["y_concede"]=out["y_concede"].fillna(0).astype(int)
    return out

def build_vaep_labels(df: pd.DataFrame):
    return vaep_labels(df)
