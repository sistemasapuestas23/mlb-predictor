import os

import joblib
import numpy as np

from .poisson_bivariate import BivariatePoisson

WEIGHTS = {"catboost": 0.30, "xgboost": 0.20, "mlp": 0.15, "lstm": 0.15, "poisson": 0.20}
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


class Ensemble:
    # Combina los 5 modelos con pesos fijos (calibrables con backtesting)
    def __init__(self, models_dir=MODELS_DIR):
        self.dir = models_dir
        self.cat = joblib.load(os.path.join(models_dir, "catboost.pkl"))
        self.xgb = joblib.load(os.path.join(models_dir, "xgboost.pkl"))
        self.mlp = joblib.load(os.path.join(models_dir, "mlp.pkl"))
        self.pois = BivariatePoisson.load(os.path.join(models_dir, "poisson.npz"))
        from .lstm_momentum import LSTMMomentum
        self.lstm = LSTMMomentum.load(os.path.join(models_dir, "lstm.pt"))

    def predict_game(self, feats, lstm_seq=None):
        # feats: dict con FEATURE_COLS; lstm_seq: array (1, 2*window, n_feat) o None
        x = np.array([[feats[c] for c in FEATURE_COLS]])
        p_cat = float(self.cat.predict_proba(x)[0, 1])
        p_xgb = float(self.xgb.predict_proba(x)[0, 1])
        p_mlp = float(self.mlp.predict_proba(x)[0, 1])
        p_home_p, _ = self.pois.predict_proba(feats["home_team"], feats["away_team"])
        probs = {"catboost": p_cat, "xgboost": p_xgb, "mlp": p_mlp, "poisson": p_home_p}
        w = {k: v for k, v in WEIGHTS.items() if k != "lstm"}
        if lstm_seq is not None:
            probs["lstm"] = float(self.lstm.predict_proba(lstm_seq)[0])
        else:
            # sin secuencia real: redistribuye el peso del LSTM entre el resto
            total = sum(WEIGHTS[k] for k in probs)
            w = {k: WEIGHTS[k] / total for k in probs}
        p_home = sum(w[k] * probs[k] for k in w)
        return p_home, probs


FEATURE_COLS = [
    "siera_home", "siera_away", "xfip_home", "xfip_away",
    "era_home", "era_away", "park_factor",
    "wind_to_out", "temp", "humidity",
    "wrc_home14", "wrc_away14", "bullpen_home3", "bullpen_away3",
]


def heuristic_fallback(feats):
    # Si aun no hay modelos entrenados: logistica manual sobre los pesos
    # conceptuales del modelo (pitcheo 35%, parque 25%, clima 15%, wRC+ 15%, bullpen 10%)
    z = (0.35 * (feats["siera_away"] - feats["siera_home"]) * 0.55
         + 0.25 * (feats["park_factor"] - 1.0) * 3.0
         + 0.15 * feats["wind_to_out"] * 0.02
         + 0.15 * (feats["wrc_home14"] - feats["wrc_away14"]) * 0.02
         + 0.10 * (feats["bullpen_away3"] - feats["bullpen_home3"]) * 0.004
         + 0.22)  # ventaja de localia
    return 1.0 / (1.0 + np.exp(-z))
