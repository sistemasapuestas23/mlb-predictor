import datetime
import os

import numpy as np
import pandas as pd
from pybaseball import statcast

# Pesos wOBA estandar (frozen de FanGraphs)
W_WALK, W_HBP = 0.69, 0.72
W_1B, W_2B, W_3B, W_HR = 0.88, 1.25, 1.58, 2.08
W_OBA_LG = 0.320

_EVENT_W = {
    "walk": W_WALK, "hit_by_pitch": W_HBP, "single": W_1B,
    "double": W_2B, "triple": W_3B, "home_run": W_HR,
}
_PAS = list(_EVENT_W.keys()) + ["field_out", "strikeout", "force_out", "grounded_into_double_play",
                                "field_error", "sac_fly", "sac_bunt", "double_play", "caught_stealing",
                                "other_out", "strikeout_double_play"]

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")


def _statcast_day(day, retries=3):
    # statcast por dia con cache parquet (las ventanas de 14/3 dias reusan dias)
    os.makedirs(CACHE_DIR, exist_ok=True)
    f = os.path.join(CACHE_DIR, "statcast_%s.parquet" % day)
    if os.path.exists(f):
        try:
            return pd.read_parquet(f)
        except Exception:
            pass
    last = None
    for i in range(retries):
        try:
            df = statcast(day, day, verbose=False)
            df.to_parquet(f)
            return df
        except Exception as e:
            last = e
    raise RuntimeError("statcast fallo para %s: %s" % (day, last))


def statcast_window(start_date, end_date):
    dfs = []
    d = start_date
    while d <= end_date:
        try:
            dfs.append(_statcast_day(d.strftime("%Y-%m-%d")))
        except Exception:
            pass
        d += datetime.timedelta(days=1)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def team_woba_map(end_date, days=14, team_park=None):
    # wOBA por equipo (bateo) en la ventana [end-days, end-1], convertido a
    # wRC+ aproximado: (wOBA_team / wOBA_lg) / park_factor * 100
    start = end_date - datetime.timedelta(days=days)
    df = statcast_window(start, end_date - datetime.timedelta(days=1))
    out = {}
    if df.empty or "batting_team" not in df.columns:
        return out
    ev = df[df["events"].notna() & df["events"].isin(_PAS)]
    w = ev["events"].map(lambda e: _EVENT_W.get(e, 0.0)).fillna(0.0)
    pa_mask = ev["events"].isin(_PAS)
    grp = ev[pa_mask].groupby("batting_team")
    num = w[pa_mask].groupby(ev[pa_mask]["batting_team"]).sum()
    den = grp.size().replace(0, np.nan)
    for team in den.index:
        team = str(team).upper()
        woba = float(num.get(team, np.nan) / den[team]) if den[team] > 0 else W_OBA_LG
        pf = (team_park or {}).get(team, 1.0)
        out[team] = (woba / W_OBA_LG) / pf * 100.0
    return out


def bullpen_usage_map(end_date, days=3):
    # Lanzamientos de relevistas (apariciones <= 60 pitches) ultimos N dias
    start = end_date - datetime.timedelta(days=days)
    df = statcast_window(start, end_date - datetime.timedelta(days=1))
    out = {}
    if df.empty or "pitching_team" not in df.columns:
        return out
    pit = df[df["pitcher"].notna() & df["pitch_type"].notna()]
    cnt = pit.groupby(["pitching_team", "pitcher"]).size().reset_index(name="n")
    rel = cnt[cnt["n"] <= 60]
    for team, tot in rel.groupby("pitching_team")["n"].sum().items():
        out[str(team).upper()] = float(tot)
    return out
