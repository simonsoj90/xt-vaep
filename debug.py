import pandas as pd
ev=pd.read_feather("data/interim/events_all.feather")
print(ev["event_type"].value_counts().head(10).to_string())
m=ev["event_type"].astype(str).str.lower().eq("shot")
print("shots",int(m.sum()))
g=m & ev["event_outcome"].astype(str).str.contains("goal",case=False,na=False)
print("goals",int(g.sum()))
print("pass_end_x_notna",int(ev["pass_end_x"].notna().sum()))
print("carry_end_x_notna",int(ev["carry_end_x"].notna().sum()))