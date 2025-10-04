from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Iterable, List
import pandas as pd
try:
    import yaml
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False
from football_analytics.io.statsbomb import SBSelection, build_events_table

def _sanitize(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("/", "-").replace(":", "-").replace("&", "and")

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("-p","--pair",action="append",default=[])
    p.add_argument("-c","--config",type=str,default=None)
    p.add_argument("-o","--outdir",default="data/interim")
    p.add_argument("-m","--merged",default="events_all.parquet")
    p.add_argument("-f","--force",action="store_true")
    return p.parse_args()

def load_pairs_from_yaml(path: Path) -> List[SBSelection]:
    if not HAVE_YAML:
        raise RuntimeError("pyyaml required")
    data = yaml.safe_load(path.read_text())
    sels=[]
    for item in data:
        sels.append(SBSelection(competition=str(item["competition"]),season=str(item["season"])))
    return sels

def load_pairs_from_cli(pair_args: Iterable[str]) -> List[SBSelection]:
    sels=[]
    for raw in pair_args:
        comp, season = raw.split(":",1)
        sels.append(SBSelection(competition=comp.strip(),season=season.strip()))
    return sels

def main():
    a=parse_args()
    outdir=Path(a.outdir); outdir.mkdir(parents=True,exist_ok=True)
    sels=[]
    if a.config: sels+=load_pairs_from_yaml(Path(a.config))
    if a.pair:   sels+=load_pairs_from_cli(a.pair)
    if not sels:
        print('provide --pair "League:Season" or --config data/leagues.yml',file=sys.stderr); sys.exit(2)
    frames=[]
    for sel in sels:
        tag=f"{_sanitize(sel.competition)}_{_sanitize(sel.season)}"
        p=outdir/f"events_{tag}.parquet"
        if p.exists() and not a.force:
            df=pd.read_parquet(p)
        else:
            df=build_events_table(sel,p)
        frames.append(df)
        print(f"{sel.competition} {sel.season} -> {p}")
    merged=pd.concat(frames,ignore_index=True).reset_index(drop=True)
    mp=outdir/a.merged
    mf=outdir/"events_all.feather"
    merged.to_parquet(mp)
    merged.to_feather(mf)
    print(f"merged_parquet: {mp}")
    print(f"merged_feather: {mf}")
    print("done")

if __name__=="__main__":
    main()
