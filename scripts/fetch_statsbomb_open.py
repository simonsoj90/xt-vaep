from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

try:
    import yaml  # optional, only needed if you pass --config
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

from football_analytics.io.statsbomb import SBSelection, build_events_table


def _sanitize_for_path(s: str) -> str:
    """Make a safe filename chunk (lowercase, replace spaces/slashes/colons, etc.)."""
    return (
        s.strip()
         .lower()
         .replace(" ", "_")
         .replace("/", "-")
         .replace(":", "-")
         .replace("&", "and")
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch & normalize StatsBomb Open Data for multiple competitions/seasons."
    )

    p.add_argument(
        "--pair",
        "-p",
        action="append",
        default=[],
        help='Repeatable "Competition:Season" (e.g., -p "Premier League:2017/2018").',
    )

    p.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="YAML file with a list of objects: [{competition: 'La Liga', season: '2017/2018'}, ...]",
    )

    p.add_argument(
        "--outdir",
        "-o",
        default="data/interim",
        help="Directory to write parquet files (default: data/interim).",
    )
    p.add_argument(
        "--merged",
        "-m",
        default="events_all.parquet",
        help="Merged parquet filename inside outdir (default: events_all.parquet).",
    )

    return p.parse_args()


def load_pairs_from_yaml(path: Path) -> List[SBSelection]:
    if not HAVE_YAML:
        raise RuntimeError("PyYAML not installed. Run `pip install pyyaml` or use --pair flags.")

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, list):
        raise ValueError("YAML must be a list of {competition, season} objects.")

    sels: List[SBSelection] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"YAML entry {i} is not a dict: {item}")
        comp = item.get("competition")
        season = item.get("season")
        if not comp or not season:
            raise ValueError(f"YAML entry {i} missing competition/season: {item}")
        sels.append(SBSelection(competition=str(comp), season=str(season)))
    return sels


def load_pairs_from_cli(pair_args: Iterable[str]) -> List[SBSelection]:
    sels: List[SBSelection] = []
    for raw in pair_args:
        if ":" not in raw:
            raise ValueError(f'Invalid --pair value "{raw}". Expected "Competition:Season".')
        comp, season = raw.split(":", 1)
        sels.append(SBSelection(competition=comp.strip(), season=season.strip()))
    return sels


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selections: List[SBSelection] = []
    if args.config:
        selections.extend(load_pairs_from_yaml(Path(args.config)))
    if args.pair:
        selections.extend(load_pairs_from_cli(args.pair))

    if not selections:
        print(
            "No competitions/seasons provided.\n"
            "Use --pair \"Premier League:2017/2018\" (repeatable), or --config data/leagues.yml",
            file=sys.stderr,
        )
        sys.exit(2)

    written_files: List[Path] = []
    frames: List[pd.DataFrame] = []

    for sel in selections:
        tag = f"{_sanitize_for_path(sel.competition)}_{_sanitize_for_path(sel.season)}"
        out_path = outdir / f"events_{tag}.parquet"
        print(f"Fetching {sel.competition} — {sel.season}")
        df = build_events_table(sel, out_path)
        print(f"  ↳ Saved {len(df):,} rows to {out_path}")
        frames.append(df)
        written_files.append(out_path)

    if frames:
        merged = pd.concat(frames, ignore_index=True).sort_values(
            ["competition_id", "season_id", "match_id", "time_seconds"],
            axis=0,
            ascending=True,
            inplace=False,
        )
        merged=merged.reset_index(drop=True)
        out_parq=outdir/args.merged
        out_feather=outdir/"events_all.feather"
        merged.to_parquet(out_parq)
        merged.to_feather(out_feather)
        print(f"merged_feather: {out_feather}")

    print("\nDone.")
    if written_files:
        print("Per-season files:")
        for p in written_files:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
