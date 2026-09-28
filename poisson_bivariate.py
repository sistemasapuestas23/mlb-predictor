import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

RHO = -0.08  # correlacion tipica de goles/carreras en Dixon-Coles


def _tau(x, y, lh, la):
    # Ajuste Dixon-Coles para marcadores bajos
    if x == 0 and y == 0:
        return 1.0 - lh * la * RHO
    if x == 0 and y == 1:
        return 1.0 + lh * RHO
    if x == 1 and y == 0:
        return 1.0 + la * RHO
    if x == 1 and y == 1:
        return 1.0 - RHO
    return 1.0


class BivariatePoisson:
    # Fuerzas de ataque/defensa por equipo + ventaja de localia,
    # ajustadas por maxima verosimilitud sobre marcadores historicos
    def fit(self, games):
        teams = sorted(set(games["home_team"]) | set(games["away_team"]))
        self.teams_, self.idx_ = teams, {t: i for i, t in enumerate(teams)}
        n = len(teams)
        x0 = np.concatenate([np.zeros(n), -np.zeros(n), [0.25]])
        bounds = [(-3, 3)] * (2 * n) + [(0.0, 1.0)]

        def nll(p):
            atk, dfc, ha = p[:n], p[n:2 * n], p[-1]
            ll = 0.0
            hs = games["home_score"].to_numpy()
            aws = games["away_score"].to_numpy()
            hi = games["home_team"].map(self.idx_).to_numpy()
            ai = games["away_team"].map(self.idx_).to_numpy()
            lh = np.exp(atk[hi] + dfc[ai] + ha)
            la = np.exp(atk[ai] + dfc[hi])
            ll += np.sum(poisson.logpmf(hs, lh) + poisson.logpmf(aws, la))
            return -ll

        res = minimize(nll, x0, bounds=bounds, method="L-BFGS-B")
        p = res.x
        self.attack_, self.defense_, self.home_adv_ = p[:n], p[n:2 * n], float(p[-1])
        return self

    def lambdas(self, home, away):
        i_h, i_a = self.idx_[home], self.idx_[away]
        lh = float(np.exp(self.attack_[i_h] + self.defense_[i_a] + self.home_adv_))
        la = float(np.exp(self.attack_[i_a] + self.defense_[i_h]))
        return lh, la

    def predict_proba(self, home, away, max_goals=16):
        lh, la = self.lambdas(home, away)
        ph = poisson.pmf(np.arange(max_goals + 1), lh)
        pa = poisson.pmf(np.arange(max_goals + 1), la)
        m = np.outer(ph, pa)
        for x in range(2):
            for y in range(2):
                m[x, y] *= _tau(x, y, lh, la)
        m = np.clip(m, 0, None)
        m /= m.sum()
        p_home = np.tril(m, -1).sum()
        p_away = np.triu(m, 1).sum()
        return p_home, p_away

    def save(self, path):
        np.savez(path, teams=self.teams_, attack=self.attack_,
                 defense=self.defense_, home_adv=self.home_adv_)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        m = cls()
        m.teams_ = list(z["teams"])
        m.idx_ = {t: i for i, t in enumerate(m.teams_)}
        m.attack_, m.defense_, m.home_adv_ = z["attack"], z["defense"], float(z["home_adv"])
        return m
