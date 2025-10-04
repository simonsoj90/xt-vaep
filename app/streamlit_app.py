import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="xt-vaep",layout="wide")
p=Path("data/processed/player_table.parquet")
if not p.exists():
    st.write("no data"); st.stop()
df=pd.read_parquet(p)

st.title("xT / VAEP")

leagues=sorted(df["competition_name"].dropna().unique().tolist()) if "competition_name" in df.columns else []
seasons=sorted(df["season_name"].dropna().unique().tolist()) if "season_name" in df.columns else []

col1,col2,col3=st.columns(3)
with col1:
    pick_league=st.multiselect("league",leagues,leagues)
with col2:
    pick_season=st.multiselect("season",seasons,seasons)
with col3:
    min_actions=st.slider("min actions",0,300,50,10)

min_minutes=st.slider("min minutes",0,3000,450,30)
q=st.text_input("filter","")
topn=st.slider("top N",10,200,50,10)

m=pd.Series(True,index=df.index)
if "competition_name" in df.columns and pick_league:
    m=m & df["competition_name"].isin(pick_league)
if "season_name" in df.columns and pick_season:
    m=m & df["season_name"].isin(pick_season)
d=df[m].copy()

if "actions" in d.columns:
    d=d[d["actions"]>=min_actions]
if "minutes_played" in d.columns:
    d=d[d["minutes_played"]>=min_minutes]
if q:
    d=d[d["player"].astype(str).str.contains(q,case=False,na=False)]

cols_xt=["competition_name","season_name","player","minutes_played","actions","xT_total","xT_per90","progressive_xT_total","progressive_xT_per90","retention_rate","retention_rate_under_pressure","duels","duel_win_rate"]
cols_vaep=["competition_name","season_name","player","minutes_played","actions","vaep_total","vaep_per90"]

t1,t2,t3=st.tabs(["xT","VAEP","Compare"])
with t1:
    dx=d[[c for c in cols_xt if c in d.columns]].copy()
    if "xT_per90" in dx.columns: dx=dx.sort_values("xT_per90",ascending=False)
    st.dataframe(dx.head(topn).reset_index(drop=True),use_container_width=True)
    st.download_button("download csv",dx.head(topn).to_csv(index=False).encode(),file_name="xt_top.csv")
with t2:
    dv=d[[c for c in cols_vaep if c in d.columns]].copy()
    if "vaep_per90" in dv.columns: dv=dv.sort_values("vaep_per90",ascending=False)
    st.dataframe(dv.head(topn).reset_index(drop=True),use_container_width=True)
    st.download_button("download csv",dv.head(topn).to_csv(index=False).encode(),file_name="vaep_top.csv")
with t3:
    if {"xT_per90","vaep_per90"}.issubset(d.columns):
        st.scatter_chart(pd.DataFrame({"xT_per90":d["xT_per90"],"vaep_per90":d["vaep_per90"]}))
        r=np.corrcoef(d["xT_per90"].fillna(0),d["vaep_per90"].fillna(0))[0,1]
        st.metric("corr",f"{r:.3f}")
