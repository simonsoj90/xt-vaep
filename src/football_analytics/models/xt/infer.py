import numpy as np
import pandas as pd
from .grid import XTGrid

class XTModel:
    def __init__(self,n_x=16,n_y=12):
        self.grid=XTGrid(n_x,n_y)

    def fit(self,events:pd.DataFrame,pitch_w=120.0,pitch_h=80.0):
        self.grid.fit(events,pitch_w,pitch_h)

    def value_events(self,events:pd.DataFrame)->pd.DataFrame:
        df=events.copy()
        df["xT_start"]=np.nan
        df["xT_end"]=np.nan
        m=df["x"].notna()&df["y"].notna()
        if m.any():
            df.loc[m,"xT_start"]=self.grid.value_xy(df.loc[m,"x"],df.loc[m,"y"])
        is_pass=df["event_type"].astype(str).eq("Pass")
        is_carry=df["event_type"].astype(str).eq("Carry")
        pm=is_pass&df["pass_end_x"].notna()&df["pass_end_y"].notna()
        if pm.any():
            df.loc[pm,"xT_end"]=self.grid.value_xy(df.loc[pm,"pass_end_x"],df.loc[pm,"pass_end_y"])
        cm=is_carry&df["carry_end_x"].notna()&df["carry_end_y"].notna()
        if cm.any():
            df.loc[cm,"xT_end"]=self.grid.value_xy(df.loc[cm,"carry_end_x"],df.loc[cm,"carry_end_y"])
        sm=df["event_type"].astype(str).eq("Shot")
        if sm.any():
            df.loc[sm,"xT_end"]=self.grid.value_xy(df.loc[sm,"x"],df.loc[sm,"y"])
            xi,yi,_=self.grid._bin_valid(df.loc[sm,"x"],df.loc[sm,"y"])
            ok=(xi>=0)&(yi>=0)
            if ok.any():
                df.loc[sm[sm].index[ok],"xT_end"]=self.grid.p_goal[yi[ok],xi[ok]]
        df["xT_delta"]=df["xT_end"].fillna(0)-df["xT_start"].fillna(0)
        return df

def aggregate_players(df:pd.DataFrame)->pd.DataFrame:
    g=df.groupby("player",dropna=False)
    out=g["xT_delta"].sum().to_frame("xT_total")
    minutes=(df["minute"].max()+1) if "minute" in df.columns else 90
    out["xT_per90"]=out["xT_total"]/(minutes/90)
    return out.reset_index()

