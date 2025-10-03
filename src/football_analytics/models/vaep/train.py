import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from joblib import dump
from football_analytics.features.labels import build_vaep_labels
from football_analytics.features.basic_feats import build_basic_features

def make_dataset(events: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    ydf=build_vaep_labels(events,k=k)
    X=build_basic_features(ydf)
    y_score=ydf["will_score_k"].astype(int)
    y_concede=ydf["will_concede_k"].astype(int)
    X=X.replace([np.inf,-np.inf],np.nan).fillna(0)
    return X,y_score,y_concede

def train_models(X: pd.DataFrame, y_score: pd.Series, y_concede: pd.Series, random_state: int = 42) -> tuple[GradientBoostingClassifier, GradientBoostingClassifier]:
    m1=GradientBoostingClassifier(random_state=random_state)
    m2=GradientBoostingClassifier(random_state=random_state)
    m1.fit(X,y_score)
    m2.fit(X,y_concede)
    return m1,m2

def predict_probs(m1: GradientBoostingClassifier, m2: GradientBoostingClassifier, X: pd.DataFrame) -> tuple[np.ndarray,np.ndarray]:
    p1=m1.predict_proba(X)[:,1]
    p2=m2.predict_proba(X)[:,1]
    return p1,p2

def save_models(dir_path: str | Path, m1: GradientBoostingClassifier, m2: GradientBoostingClassifier) -> tuple[Path,Path]:
    d=Path(dir_path); d.mkdir(parents=True,exist_ok=True)
    p1=d/"vaep_score.joblib"
    p2=d/"vaep_concede.joblib"
    dump(m1,p1)
    dump(m2,p2)
    return p1,p2

def fit_vaep_lite(events: pd.DataFrame, k: int = 5, models_dir: str | Path = "models") -> tuple[pd.DataFrame, GradientBoostingClassifier, GradientBoostingClassifier, float, float]:
    X,y1,y2=make_dataset(events,k=k)
    m1,m2=train_models(X,y1,y2)
    p1,p2=predict_probs(m1,m2,X)
    ls=log_loss(y1,p1,labels=[0,1])
    lc=log_loss(y2,p2,labels=[0,1])
    out=events.copy()
    out["vaep_p_score"]=pd.Series(p1,index=out.index)
    out["vaep_p_concede"]=pd.Series(p2,index=out.index)
    save_models(models_dir,m1,m2)
    return out,m1,m2,ls,lc
