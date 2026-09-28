import numpy as np
import torch
import torch.nn as nn


class LSTMWin(nn.Module):
    # LSTM sobre la secuencia de los ultimos N partidos del equipo:
    # [r_wRC+ propio, r_wRC+ rival, r_SIERA abridor propio, r_SIERA rival,
    #  park_factor, victoria(0/1)] -> probabilidad de ganar el siguiente
    def __init__(self, n_feat=6, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.head(out[:, -1, :])).squeeze(-1)


def build_sequences(games, window=10, n_feat=6):
    # games: DataFrame ordenado por fecha con columnas
    # home_team, away_team, home_win, wrc_home14, wrc_away14, siera_home, siera_away, park_factor
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    hist = {t: [] for t in teams}
    X, y = [], []
    for _, g in games.sort_values("date").iterrows():
        h, a = g["home_team"], g["away_team"]
        if len(hist[h]) >= 3 and len(hist[a]) >= 3:
            seq_h = np.array(hist[h][-window:], dtype=np.float32)
            seq_a = np.array(hist[a][-window:], dtype=np.float32)
            pad = lambda s: np.vstack([np.zeros((window - len(s), n_feat)), s]) if len(s) < window else s[-window:]
            X.append(np.concatenate([pad(seq_h), pad(seq_a)]))  # 2*window pasos
            y.append(float(g["home_win"]))
        f = lambda w, l, s_w, s_l: [w, l, s_w, s_l, g["park_factor"], float(g["home_win"])]
        hist[h].append(f(g["wrc_home14"], g["wrc_away14"], g["siera_home"], g["siera_away"]))
        hist[a].append(f(g["wrc_away14"], g["wrc_home14"], g["siera_away"], g["siera_home"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LSTMMomentum:
    def fit(self, X, y, epochs=40, lr=1e-3, seed=42):
        torch.manual_seed(seed)
        n_feat = X.shape[2]
        self.model_ = LSTMWin(n_feat=n_feat)
        opt = torch.optim.Adam(self.model_.parameters(), lr=lr)
        lossf = nn.BCELoss()
        Xt = torch.tensor(X)
        yt = torch.tensor(y)
        for _ in range(epochs):
            self.model_.train()
            opt.zero_grad()
            loss = lossf(self.model_(Xt), yt)
            loss.backward()
            opt.step()
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.tensor(X, dtype=torch.float32)).numpy()

    def save(self, path):
        torch.save(self.model_.state_dict(), path)

    @classmethod
    def load(cls, path, n_feat=6):
        m = cls()
        m.model_ = LSTMWin(n_feat=n_feat)
        m.model_.load_state_dict(torch.load(path, map_location="cpu"))
        m.model_.eval()
        return m
