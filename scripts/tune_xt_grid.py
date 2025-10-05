import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from football_analytics.models.xt.infer import XTModel

def eval_grid(ev_train,ev_val,nx,ny):
    m=XTModel(nx,ny)
    m.fit(ev_train)
    v=m.grid.v.copy()
    shots=ev_val[ev_val["event_type"].astype(str).eq("Shot")].copy()
    if len(shots)<10:
        return v, np.nan
    xs=m.grid.value(shots["x"].to_numpy(),shots["y"].to_numpy())
    y=(shots["event_outcome"].astype(str).str.contains("goal",case=False,na=False)).astype(int).to_numpy()
    if y.sum()==0 or y.sum()==len(y):
        return v, np.nan
    auc=roc_auc_score(y,xs)
    return v, auc

def bootstrap_ci(ev_train,nx,ny,B=30):
    n=len(ev_train)
    vs=[]
    for b in range(B):
        idx=np.random.randint(0,n,size=n)
        m=XTModel(nx,ny)
        m.fit(ev_train.iloc[idx])
        vs.append(m.grid.v.copy())
    arr=np.stack(vs,axis=0)
    lo=np.quantile(arr,0.05,axis=0)
    hi=np.quantile(arr,0.95,axis=0)
    mu=np.mean(arr,axis=0)
    return mu,lo,hi

def main():
    Path("models/xt_tuned").mkdir(parents=True,exist_ok=True)
    Path("reports/xt").mkdir(parents=True,exist_ok=True)
    ev=pd.read_feather("data/interim/events_all.feather")
    ev=ev[ev["event_type"].isin(["Pass","Carry","Shot"])].copy()
    gss=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    mid=ev["match_id"].values
    tr,va=next(gss.split(ev,groups=mid))
    ev_train=ev.iloc[tr]
    ev_val=ev.iloc[va]
    grids=[(12,9),(16,12),(24,16)]
    rows=[]
    best=None
    best_auc=-1
    for nx,ny in grids:
        v,auc=eval_grid(ev_train,ev_val,nx,ny)
        np.savez(f"models/xt_tuned/grid_{nx}x{ny}.npz",v=v)
        rows.append((nx,ny,auc))
        if auc==auc and auc>best_auc:
            best_auc=auc
            best=(nx,ny)
    df=pd.DataFrame(rows,columns=["n_x","n_y","shot_auc"])
    df.to_csv("reports/xt/xt_grid_metrics.csv",index=False)
    if best is not None:
        nx,ny=best
        mu,lo,hi=bootstrap_ci(ev_train,nx,ny,B=30)
        np.savez("models/xt_tuned/best_grid.npz",v=mu,lo=lo,hi=hi,n_x=nx,n_y=ny,auc=best_auc)
        np.savez("data/processed/xt_grid.npz",v=mu)

if __name__=="__main__":
    main()
