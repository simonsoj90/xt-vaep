import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="xt-vaep",layout="wide")
st.title("Player Finder")

p=Path("data/processed/player_table.parquet")
if p.exists():
    df=pd.read_parquet(p)
else:
    df=pd.DataFrame()

cols=["player","xT_total","xT_per90","progressive_xT_total","progressive_xT_per90","retention_rate","retention_rate_under_pressure","duels","duel_win_rate"]
df=df[cols] if len(df)>0 else df

pos=st.text_input("Filter player substring","")
if pos:
    df=df[df["player"].astype(str).str.contains(pos,case=False,na=False)]

st.dataframe(df.sort_values("xT_per90",ascending=False).reset_index(drop=True),use_container_width=True)
