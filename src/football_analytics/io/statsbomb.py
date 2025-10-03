from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd
from statsbombpy import sb
from dataclasses import dataclass

@dataclass
class SBSelection:
    competition: str
    season: str

def list_competitions() -> pd.DataFrame:
    comps = sb.competitions()
    comps["competition_name"] = comps["competition_name"].astype(str)
    comps["season_name"] = comps["season_name"].astype(str)
    return comps

def resolve_comp_season(selection: SBSelection) -> tuple[int, int]:
    comps = list_competitions()
    df = comps[(comps["competition_name"].str.lower() == selection.competition.lower()) & (comps["season_name"].str.lower() == selection.season.lower())]
    row = df.iloc[0]
    return int(row["competition_id"]), int(row["season_id"])

def fetch_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    m = sb.matches(competition_id=competition_id, season_id=season_id)
    for c in ["match_id", "home_team_id", "away_team_id"]:
        if c in m.columns:
            m[c] = m[c].astype(int)
    return m

def _safe_get(d, k, default=None):
    return d.get(k, default) if isinstance(d, dict) else default

def _split_location(row: pd.Series, field: str):
    loc = row.get(field)
    if isinstance(loc, (list, tuple)) and len(loc) == 2:
        return float(loc[0]), float(loc[1])
    return None, None

def _safe_sb_events(match_id: int) -> pd.DataFrame:
    try:
        return sb.events(match_id=int(match_id), fix_missing_players=True)
    except TypeError:
        return sb.events(match_id=int(match_id))

def fetch_events_for_matches(match_ids: Iterable[int]) -> pd.DataFrame:
    frames = []
    for mid in match_ids:
        ev = _safe_sb_events(int(mid))
        ev["match_id"] = int(mid)
        frames.append(ev)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def normalize_events(events: pd.DataFrame, match_meta: pd.DataFrame) -> pd.DataFrame:
    cols=["match_id","team","player","type","minute","second","period","timestamp","possession","play_pattern","under_pressure","location","pass_end_location","carry_end_location","pass_length","pass_angle","outcome","shot_outcome","duration"]
    use=[c for c in cols if c in events.columns]
    df=events[use].copy()
    if "type" in df.columns:
        df["event_type"]=df["type"].apply(lambda d: d.get("name") if isinstance(d,dict) else d)
    else:
        df["event_type"]=pd.Series(index=df.index,dtype=object)
    if "shot_outcome" in df.columns:
        so=df["shot_outcome"]
        df["shot_outcome"]=so.apply(lambda d: d.get("name") if isinstance(d,dict) else d)
    def outc(row):
        a=row.get("shot_outcome")
        b=row.get("outcome")
        if isinstance(a,dict):
            a=a.get("name")
        if isinstance(b,dict):
            b=b.get("name")
        return a if pd.notna(a) and str(a)!="" else b
    df["event_outcome"]=df.apply(outc,axis=1)
    if "shot_outcome" in df.columns:
        m=df["shot_outcome"].astype(str).str.len()>0
        df.loc[m,"event_type"]="Shot"
    for c,k in [("team","name"),("player","name"),("play_pattern","name")]:
        if c in df.columns:
            df[c]=df[c].apply(lambda d: d.get(k) if isinstance(d,dict) else d)
    if "under_pressure" in df.columns:
        df["under_pressure"]=df["under_pressure"].fillna(False).astype(bool)
    else:
        df["under_pressure"]=False
    def split_loc(v):
        if isinstance(v,(list,tuple)) and len(v)==2:
            return float(v[0]),float(v[1])
        return None,None
    df["x"],df["y"]=zip(*df.apply(lambda r: split_loc(r.get("location")),axis=1))
    df["pass_end_x"],df["pass_end_y"]=zip(*df.apply(lambda r: split_loc(r.get("pass_end_location")),axis=1))
    df["carry_end_x"],df["carry_end_y"]=zip(*df.apply(lambda r: split_loc(r.get("carry_end_location")),axis=1))
    df["time_seconds"]=df["minute"].fillna(0)*60+df["second"].fillna(0)
    keep=["match_id","team","player","event_type","event_outcome","period","minute","second","time_seconds","possession","x","y","pass_end_x","pass_end_y","carry_end_x","carry_end_y","pass_length","pass_angle","under_pressure","duration"]
    keep=[c for c in keep if c in df.columns]
    df=df[keep].copy()
    mm=match_meta.copy()
    if "home_team" in mm.columns and isinstance(mm["home_team"].iloc[0],dict):
        mm["home_team"]=mm["home_team"].apply(lambda d: d.get("home_team_name") or d.get("name") or d)
    if "away_team" in mm.columns and isinstance(mm["away_team"].iloc[0],dict):
        mm["away_team"]=mm["away_team"].apply(lambda d: d.get("away_team_name") or d.get("name") or d)
    for c in ["home_team","away_team"]:
        if c not in mm.columns and f"{c}_name" in mm.columns:
            mm[c]=mm[f"{c}_name"]
    meta_cols=["match_id","competition_id","season_id","match_date","home_team","away_team","home_score","away_score"]
    mm=mm[[c for c in meta_cols if c in mm.columns]].drop_duplicates("match_id")
    out=df.merge(mm,on="match_id",how="left")
    return out


def build_events_table(selection: SBSelection, out_parquet: Path) -> pd.DataFrame:
    comp_id, season_id = resolve_comp_season(selection)
    matches = fetch_matches(comp_id, season_id)
    events_raw = fetch_events_for_matches(matches["match_id"].tolist())
    events_norm = normalize_events(events_raw, matches)
    if "competition_id" not in events_norm.columns:
        events_norm["competition_id"] = comp_id
    if "season_id" not in events_norm.columns:
        events_norm["season_id"] = season_id
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    events_norm.to_parquet(out_parquet)
    return events_norm

