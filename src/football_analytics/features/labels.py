import numpy as np
import pandas as pd

def build_vaep_labels(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    x=df.sort_values(["match_id","time_seconds","period","minute","second"],na_position="last").reset_index(drop=True).copy()
    x["__idx"]=np.arange(len(x))
    x["__team_code"]=pd.factorize(x["team"].astype(str))[0]
    x["__is_goal"]=x["event_type"].astype(str).eq("Shot") & x["event_outcome"].astype(str).str.contains("goal",case=False,na=False)
    out=np.zeros((len(x),2),dtype=np.int8)
    for mid,g in x.groupby("match_id",sort=False):
        idx=g["__idx"].to_numpy()
        tm=g["__team_code"].to_numpy()
        gl=g["__is_goal"].to_numpy()
        n=len(g)
        t0=np.unique(tm)
        if len(t0)==0:
            continue
        ta=t0[0]
        tb=t0[1] if len(t0)>1 else -1
        ga=(gl & (tm==ta)).astype(np.int8)
        gb=(gl & (tm==tb)).astype(np.int8) if tb!=-1 else np.zeros(n,dtype=np.int8)
        cga=np.cumsum(ga)
        cgb=np.cumsum(gb)
        i=np.arange(n)
        j=np.minimum(i+k,n-1)
        sa=(cga[j]-cga[i])>0
        sb=(cgb[j]-cgb[i])>0
        score=np.where(tm==ta,sa,sb)
        concede=np.where(tm==ta,sb,sa)
        out[idx,0]=score.astype(np.int8)
        out[idx,1]=concede.astype(np.int8)
    y=pd.DataFrame({"will_score_k":out[:,0],"will_concede_k":out[:,1]})
    x=x.drop(columns=["__idx","__team_code","__is_goal"])
    return pd.concat([x,y],axis=1)
