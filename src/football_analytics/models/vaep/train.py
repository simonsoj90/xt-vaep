import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss,brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from pathlib import Path
import joblib

def _split(X,y,groups,test_size=0.2,random_state=42):
    gss=GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=random_state)
    tr,va=next(gss.split(X,y,groups))
    return tr,va

def _sanitize(X):
    X=np.asarray(X,dtype=np.float32)
    X=np.nan_to_num(X,copy=False,posinf=0.0,neginf=0.0)
    return X

def _fit_model(X,y):
    X=_sanitize(X)
    m=HistGradientBoostingClassifier(random_state=0)
    m.fit(X,y)
    p=m.predict_proba(X)[:,1]
    cal=IsotonicRegression(out_of_bounds="clip").fit(p,y)
    return m,cal

def fit_vaep(events: pd.DataFrame, labels: pd.DataFrame, features: pd.DataFrame, out_dir: str = "models/vaep"):
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    df=features.merge(labels[["match_id","time_seconds","period","y_score","y_concede"]],on=["match_id","time_seconds","period"],how="left").fillna({"y_score":0,"y_concede":0})
    feats=[c for c in df.columns if c not in ["match_id","poss_id","team","player","time_seconds","period","event_type","y_score","y_concede"]]
    X=df[feats].values
    y1=df["y_score"].astype(int).values
    y2=df["y_concede"].astype(int).values
    groups=df["match_id"].values
    tr,va=_split(X,y1,groups)
    m1,cal1=_fit_model(X[tr],y1[tr])
    m2,cal2=_fit_model(X[tr],y2[tr])
    Xv=_sanitize(X[va])
    p1=cal1.predict(m1.predict_proba(Xv)[:,1])
    p2=cal2.predict(m2.predict_proba(Xv)[:,1])
    metrics=pd.DataFrame({
        "metric":["logloss_score","brier_score","logloss_concede","brier_concede"],
        "value":[log_loss(y1[va],p1,labels=[0,1]),brier_score_loss(y1[va],p1),log_loss(y2[va],p2,labels=[0,1]),brier_score_loss(y2[va],p2)]
    })
    Path("reports/validation").mkdir(parents=True,exist_ok=True)
    metrics.to_csv("reports/validation/vaep_metrics.csv",index=False)
    joblib.dump(m1,f"{out_dir}/score_hgb.joblib")
    joblib.dump(m2,f"{out_dir}/concede_hgb.joblib")
    joblib.dump(cal1,f"{out_dir}/score_iso.joblib")
    joblib.dump(cal2,f"{out_dir}/concede_iso.joblib")
    return feats

def value_events(events: pd.DataFrame, features: pd.DataFrame, feats: list, out_path: str = "data/processed/events_with_vaep.parquet"):
    import joblib, numpy as np, pandas as pd
    m1=joblib.load("models/vaep/score_hgb.joblib")
    m2=joblib.load("models/vaep/concede_hgb.joblib")
    c1=joblib.load("models/vaep/score_iso.joblib")
    c2=joblib.load("models/vaep/concede_iso.joblib")
    X=features[feats].values
    X=np.nan_to_num(X,copy=False,posinf=0.0,neginf=0.0).astype(np.float32)
    p1=c1.predict(m1.predict_proba(X)[:,1])
    p2=c2.predict(m2.predict_proba(X)[:,1])
    df=events.copy()
    df["vaep_p_score"]=p1
    df["vaep_p_concede"]=p2
    df["vaep_delta"]=df["vaep_p_score"]-df["vaep_p_concede"]
    df.to_parquet(out_path)
    return df
