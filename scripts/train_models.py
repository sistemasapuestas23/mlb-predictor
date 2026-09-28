import argparse
import datetime
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config_loader import load_team_config          # noqa: E402
from src.data_ingestion import get_schedule, get_pitching_stats, match_pitcher  # noqa: E402
from src.features import team_woba_map, bullpen_usage_map  # noqa: E402
from src.models.ensemble import FEATURE_COLS            # noqa: E402
from src.models.poisson_bivariate import BivariatePoisson  # noqa: E402
from src.models.lstm_momentum import LSTMMomentum, build_sequences  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS = os.path.join(os.path.dirname(__file__), "..", "models")


def build_dataset(seasons):
    # Construye dataset historico: por cada juego, features + marcador real.
    # Nota: SIERA usado = temporada completa (proxy con leakage aceptado y
    # documentado; el wRC+14d y bullpen3 si son causales).
    rows = []
    cfg = load_team_config()
    pf = {k: v["park_factor"] for k, v in cfg.items()}
    for season in seasons:
        pitch_df = get_pitching_stats(season)
        d = datetime.date(season, 4, 1)
        stop = min(datetime.date(season, 10, 5), datetime.date.today())
        while d <= stop:
            sched = get_schedule(d.strftime("%Y-%m-%d"))
            if sched.empty:
                d += datetime.timedelta(days=1)
                continue
            wrc_map = team_woba_map(d, days=14, team_park=pf)
            bp_map = bullpen_usage_map(d, days=3)
            for _, g in sched.iterrows():
                sph = match_pitcher(g["home_probable"], pitch_df)
                spa = match_pitcher(g["away_probable"], pitch_df)
                if g["home_abbr"] not in pf or g["away_abbr"] not in pf:
                    continue
                rows.append(dict(
                    date=d.strftime("%Y-%m-%d"),
                    home_team=g["home_abbr"], away_team=g["away_abbr"],
                    siera_home=float(sph["SIERA"]) if sph is not None else 4.20,
                    siera_away=float(spa["SIERA"]) if spa is not None else 4.20,
                    xfip_home=float(sph["xFIP"]) if sph is not None else 4.20,
                    xfip_away=float(spa["xFIP"]) if spa is not None else 4.20,
                    era_home=float(sph["ERA"]) if sph is not None else 4.30,
                    era_away=float(spa["ERA"]) if spa is not None else 4.30,
                    park_factor=pf[g["home_abbr"]],
                    wind_to_out=0.0, temp=72.0, humidity=55.0,  # historico: se omite clima real
                    wrc_home14=wrc_map.get(g["home_abbr"], 100.0),
                    wrc_away14=wrc_map.get(g["away_abbr"], 100.0),
                    bullpen_home3=bp_map.get(g["home_abbr"], 200.0),
                    bullpen_away3=bp_map.get(g["away_abbr"], 200.0),
                    home_score=int(g.get("home_score") or -1),
                    away_score=int(g.get("away_score") or -1),
                ))
            d += datetime.timedelta(days=1)
    df = pd.DataFrame(rows)
    df = df[(df.home_score >= 0) & (df.away_score >= 0)].reset_index(drop=True)
    df["home_win"] = (df.home_score > df.away_score).astype(int)
    os.makedirs(DATA, exist_ok=True)
    df.to_parquet(os.path.join(DATA, "training_games.parquet"))
    print("dataset:", df.shape, "guardado en", DATA)
    return df


def train(df):
    from catboost import CatBoostClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss, brier_score_loss
    from xgboost import XGBClassifier

    os.makedirs(MODELS, exist_ok=True)
    X = df[FEATURE_COLS].astype(float)
    y = df["home_win"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

    models = {
        "catboost": CatBoostClassifier(iterations=600, depth=5, learning_rate=0.05,
                                       verbose=0, random_seed=42),
        "xgboost": XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                                 subsample=0.9, eval_metric="logloss", random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    }
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        print("%-9s logloss=%.4f brier=%.4f" % (name, log_loss(y_te, p), brier_score_loss(y_te, p)))
        joblib.dump(m, os.path.join(MODELS, name + ".pkl"))

    # Poisson bivariada sobre marcadores
    pois = BivariatePoisson().fit(df.rename(columns={"home_team": "home_team"}))
    pois.save(os.path.join(MODELS, "poisson.npz"))

    # LSTM momentum sobre secuencias por equipo
    seq_x, seq_y = build_sequences(df)
    lstm = LSTMMomentum().fit(seq_x, seq_y)
    lstm.save(os.path.join(MODELS, "lstm.pt"))
    print("modelos guardados en", MODELS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "train"])
    ap.add_argument("--seasons", nargs="+", type=int,
                    default=[datetime.date.today().year - 2, datetime.date.today().year - 1,
                             datetime.date.today().year])
    args = ap.parse_args()
    if args.cmd == "build":
        build_dataset(args.seasons)
    else:
        df = pd.read_parquet(os.path.join(DATA, "training_games.parquet"))
        train(df)


if __name__ == "__main__":
    main()
