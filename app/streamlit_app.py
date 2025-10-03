import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="xt-vaep",layout="wide")
p=Path("data/processed/player_table.parquet")
if not p.exists():
    st.write("no data")
    st.stop()
df=pd.read_parquet(p)
cols_xt=["player","xT_total","xT_per90","progressive_xT_total","progressive_xT_per90","retention_rate","retention_rate_under_pressure","duels","duel_win_rate"]
cols_vaep=["player","vaep_total","vaep_per90"]
df_xt=df[[c for c in cols_xt if c in df.columns]].copy()
df_vaep=df[[c for c in cols_vaep if c in df.columns]].copy()
st.title("xT / VAEP")
q=st.text_input("filter","")
n=st.slider("top N",10,200,50,10)
t1,t2,t3=st.tabs(["xT","VAEP","Compare"])
with t1:
    d=df_xt
    if q:
        d=d[d["player"].astype(str).str.contains(q,case=False,na=False)]
    if "xT_per90" in d.columns:
        d=d.sort_values("xT_per90",ascending=False)
    st.dataframe(d.head(n).reset_index(drop=True),use_container_width=True)
    st.download_button("download csv",d.head(n).to_csv(index=False).encode(),file_name="xt_top.csv")
with t2:
    d=df_vaep
    if q:
        d=d[d["player"].astype(str).str.contains(q,case=False,na=False)]
    if "vaep_per90" in d.columns:
        d=d.sort_values("vaep_per90",ascending=False)
    st.dataframe(d.head(n).reset_index(drop=True),use_container_width=True)
    st.download_button("download csv",d.head(n).to_csv(index=False).encode(),file_name="vaep_top.csv")
with t3:
    d=df.merge(df_vaep,on="player",how="left") if "vaep_per90" in df.columns else df.copy()
    if q:
        d=d[d["player"].astype(str).str.contains(q,case=False,na=False)]
    x=d.get("xT_per90")
    y=d.get("vaep_per90")
    if x is not None and y is not None:
        st.scatter_chart(pd.DataFrame({"xT_per90":x,"vaep_per90":y}))
        r=np.corrcoef(x.fillna(0),y.fillna(0))[0,1]
        st.metric("corr",f"{r:.3f}")
