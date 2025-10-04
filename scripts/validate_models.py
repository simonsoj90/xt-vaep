from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from scipy.stats import spearmanr
from football_analytics.features.labels import build_vaep_labels

def ensure_dirs():
    Path("reports/validation").mkdir(parents=True,exist_ok=True)

def plot_xt_grid(npz_path:str, out_path:str):
    z=np.load(npz_path,allow_pickle=True)
    v=z["v"]
    plt.figure()
    plt.imshow(v,origin="lower")
    plt.colorbar()
    plt.title("xT grid")
    plt.savefig(out_path,bbox_inches="tight")
    plt.close()

def vaep_reliability(events_path:str,k:int,out_png:str,out_csv:str):
    ev=pd.read_parquet(events_path)
    y=build_vaep_labels(ev,k=k)
    p_score=y["vaep_p_score"] if "vaep_p_score" in y.columns else ev["vaep_p_score"]
    p_concede=y["vaep_p_concede"] if "vaep_p_concede" in y.columns else ev["vaep_p_concede"]
    ys=y["will_score_k"].astype(int)
    yc=y["will_concede_k"].astype(int)
    p_score=np.clip(p_score,0,1)
    p_concede=np.clip(p_concede,0,1)
    bs_score=brier_score_loss(ys,p_score)
    bs_concede=brier_score_loss(yc,p_concede)
    ll_score=log_loss(ys,p_score,labels=[0,1])
    ll_concede=log_loss(yc,p_concede,labels=[0,1])
    fr_score, mr_score = calibration_curve(ys, p_score, n_bins=20, strategy="quantile")
    fr_concede, mr_concede = calibration_curve(yc, p_concede, n_bins=20, strategy="quantile")
    plt.figure()
    plt.plot(mr_score,fr_score,marker="o")
    plt.plot([0,1],[0,1])
    plt.title("VAEP score reliability")
    plt.savefig(out_png.replace(".png","_score.png"),bbox_inches="tight")
    plt.close()
    plt.figure()
    plt.plot(mr_concede,fr_concede,marker="o")
    plt.plot([0,1],[0,1])
    plt.title("VAEP concede reliability")
    plt.savefig(out_png.replace(".png","_concede.png"),bbox_inches="tight")
    plt.close()
    pd.DataFrame({
        "metric":["brier_score","log_loss"],
        "score":[bs_score,ll_score],
        "concede":[bs_concede,ll_concede]
    }).to_csv(out_csv,index=False)

def split_half_stability(events_xt_path:str, out_csv:str, min_actions:int=100):
    ev=pd.read_parquet(events_xt_path)
    ev["half_tag"]=ev["match_id"].astype(int)%2
    def agg(e):
        g=e.groupby("player",dropna=False)["xT_delta"].sum()
        m=(e["minute"].max()+1) if "minute" in e.columns else 90
        return (g/(m/90)).rename("xT_per90")
    a=agg(ev[ev["half_tag"]==0]).to_frame()
    b=agg(ev[ev["half_tag"]==1]).to_frame()
    c=ev.groupby("player",dropna=False).size().rename("actions").to_frame()
    df=a.merge(b,left_index=True,right_index=True,how="inner",suffixes=("_even","_odd")).merge(c,left_index=True,right_index=True,how="left")
    df=df[df["actions"]>=min_actions]
    r,s=spearmanr(df["xT_per90_even"],df["xT_per90_odd"],nan_policy="omit")
    df["rank_even"]=df["xT_per90_even"].rank(ascending=False,method="average")
    df["rank_odd"]=df["xT_per90_odd"].rank(ascending=False,method="average")
    df.to_csv(out_csv,index=True)
    Path(out_csv.replace(".csv","_spearman.txt")).write_text(f"{r}")

def main():
    ensure_dirs()
    plot_xt_grid("data/processed/xt_grid.npz","reports/validation/xt_grid.png")
    vaep_reliability("data/processed/events_with_vaep.parquet",5,"reports/validation/vaep_reliability.png","reports/validation/vaep_metrics.csv")
    split_half_stability("data/processed/events_with_xt.parquet","reports/validation/xt_stability.csv",min_actions=100)

if __name__=="__main__":
    main()
