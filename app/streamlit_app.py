import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="xt-vaep",layout="wide")
p=Path("data/processed/player_table.parquet")
df=pd.read_parquet(p) if p.exists() else pd.DataFrame()
cols=["player","xT_total","xT_per90","progressive_xT_total","progressive_xT_per90","retention_rate","retention_rate_under_pressure","duels","duel_win_rate"]
df=df[cols].copy() if len(df)>0 else df
st.title("Player Finder")
q=st.text_input("Filter","")
if q:
    df=df[df["player"].astype(str).str.contains(q,case=False,na=False)]
topn=st.slider("Top N",10,200,50,10)
if len(df)>0:
    df=df.sort_values("xT_per90",ascending=False).reset_index(drop=True)
    st.dataframe(df.head(topn),use_container_width=True)
    st.download_button("Download CSV",df.to_csv(index=False).encode(),file_name="player_table.csv")
else:
    st.write("No data found")
