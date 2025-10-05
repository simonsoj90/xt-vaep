import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="xT / VAEP", layout="wide")

p_raw=Path("data/processed/player_table.parquet")
p_norm=Path("data/processed/player_table_norm.parquet")

st.title("xT / VAEP")

use_norm=st.checkbox("use normalized per90", value=True)
if use_norm and p_norm.exists():
    df=pd.read_parquet(p_norm)
elif p_raw.exists():
    df=pd.read_parquet(p_raw)
else:
    st.write("no data")
    st.stop()

def uniq(col):
    return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

leagues=uniq("competition_name")
seasons=uniq("season_name")
roles=uniq("role")
clusters=uniq("role_cluster")

c0,c1,c2,c3=st.columns(4)
with c0:
    pick_league=st.multiselect("league", leagues, leagues)
with c1:
    pick_season=st.multiselect("season", seasons, seasons)
with c2:
    pick_role=st.multiselect("role", roles, [])
with c3:
    pick_cluster=st.multiselect("role cluster", clusters, [])

min_actions=st.slider("min actions", 0, 300, 100, 10)
min_minutes=st.slider("min minutes", 0, 3000, 900, 30)
q=st.text_input("filter", "")
topn=st.slider("top N", 10, 300, 50, 10)

m=pd.Series(True, index=df.index)
if "competition_name" in df.columns and len(pick_league)>0:
    m=m & df["competition_name"].isin(pick_league)
if "season_name" in df.columns and len(pick_season)>0:
    m=m & df["season_name"].isin(pick_season)
if "role" in df.columns and len(pick_role)>0:
    m=m & df["role"].isin(pick_role)
if "role_cluster" in df.columns and len(pick_cluster)>0:
    m=m & df["role_cluster"].isin(pick_cluster)

d=df[m].copy()
if "actions" in d.columns:
    d=d[d["actions"].fillna(0)>=min_actions]
if "minutes_played" in d.columns:
    d=d[d["minutes_played"].fillna(0)>=min_minutes]
if q:
    d=d[d["player"].astype(str).str.contains(q, case=False, na=False)]

have_norm=("xT_per90_norm" in d.columns) and use_norm

cols_xt=["competition_name","season_name","role_cluster","role","player","minutes_played","actions","xT_total","xT_per90","xT_per90_norm","progressive_xT_total","progressive_xT_per90","progressive_xT_per90_norm","progressive_actions","progressive_actions_per90","progressive_actions_per90_norm","retention_rate","retention_rate_under_pressure","duels","duel_win_rate"]
cols_vaep=["competition_name","season_name","role_cluster","role","player","minutes_played","actions","vaep_total","vaep_per90","vaep_per90_norm"]

for c in ["xT_per90_norm","progressive_xT_per90_norm","progressive_actions_per90_norm","vaep_per90_norm"]:
    if c not in d.columns:
        d[c]=np.nan

t1,t2,t3=st.tabs(["xT","VAEP","Compare"])

with t1:
    dx=d[[c for c in cols_xt if c in d.columns]].copy()
    sort_col="xT_per90_norm" if have_norm else "xT_per90"
    if sort_col in dx.columns:
        dx=dx.sort_values(sort_col, ascending=False)
    st.dataframe(dx.head(topn).reset_index(drop=True), width="stretch")
    if len(dx)>0:
        st.download_button("download csv", dx.head(topn).to_csv(index=False).encode(), file_name="xt_top.csv")

with t2:
    dv=d[[c for c in cols_vaep if c in d.columns]].copy()
    sort_col="vaep_per90_norm" if have_norm else "vaep_per90"
    if sort_col in dv.columns:
        dv=dv.sort_values(sort_col, ascending=False)
    st.dataframe(dv.head(topn).reset_index(drop=True), width="stretch")
    if len(dv)>0:
        st.download_button("download csv", dv.head(topn).to_csv(index=False).encode(), file_name="vaep_top.csv")

with t3:
    xcol="xT_per90_norm" if have_norm else "xT_per90"
    ycol="vaep_per90_norm" if have_norm else "vaep_per90"
    if {xcol,ycol}.issubset(d.columns):
        st.scatter_chart(pd.DataFrame({xcol:d[xcol], ycol:d[ycol]}))
        r=np.corrcoef(d[xcol].fillna(0), d[ycol].fillna(0))[0,1]
        st.metric("corr", f"{r:.3f}")
    else:
        st.write("comparison unavailable")
