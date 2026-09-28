import argparse
import datetime
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config_loader import load_team_config          # noqa: E402
from src.data_ingestion import get_schedule, get_pitching_stats, match_pitcher  # noqa: E402
from src.features import team_woba_map, bullpen_usage_map  # noqa: E402
from src.weather import get_stadium_weather             # noqa: E402
from src.telegram_bot import send_message               # noqa: E402
from src.models.ensemble import Ensemble, FEATURE_COLS, heuristic_fallback  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler("logs/run.log", mode="a"),
                              logging.StreamHandler()])
log = logging.getLogger("mlb-daily")


def _g(row, col, default=np.nan):
    v = row.get(col)
    return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default


def build_game_features(game, pitch_df, wrc_map, bp_map, team_cfg_map, today):
    home, away = game["home_abbr"], game["away_abbr"]
    c_home, c_away = team_cfg_map[home], team_cfg_map[away]

    sp_home = match_pitcher(game["home_probable"], pitch_df)
    sp_away = match_pitcher(game["away_probable"], pitch_df)
    siera_home = _g(sp_home, "SIERA", 4.20) if sp_home is not None else 4.20
    siera_away = _g(sp_away, "SIERA", 4.20) if sp_away is not None else 4.20
    xfip_home = _g(sp_home, "xFIP", 4.20) if sp_home is not None else 4.20
    xfip_away = _g(sp_away, "xFIP", 4.20) if sp_away is not None else 4.20
    era_home = _g(sp_home, "ERA", 4.30) if sp_home is not None else 4.30
    era_away = _g(sp_away, "ERA", 4.30) if sp_away is not None else 4.30

    w = get_stadium_weather(c_home["lat"], c_home["lon"], c_home["dome"], c_home["wind_out_dir"])

    feats = dict(
        home_team=home, away_team=away,
        siera_home=siera_home, siera_away=siera_away,
        xfip_home=xfip_home, xfip_away=xfip_away,
        era_home=era_home, era_away=era_away,
        park_factor=float(c_home["park_factor"]),
        wind_to_out=float(w["wind_to_out"]), temp=float(w["temp"]), humidity=float(w["humidity"]),
        wrc_home14=wrc_map.get(home, 100.0), wrc_away14=wrc_map.get(away, 100.0),
        bullpen_home3=bp_map.get(home, 200.0), bullpen_away3=bp_map.get(away, 200.0),
    )
    meta = dict(weather=w, sp_home=sp_home, sp_away=sp_away)
    return feats, meta


def format_message(date_str, rows):
    lines = ["⚾ <b>PREDICCIÓN MLB — %s</b>" % date_str, ""]
    for r in rows:
        lines += [
            "<b>%s @ %s</b>" % (r["away"], r["home"]),
            "└ Pick Proyectado: <b>%s</b> (%.1f%%)" % (r["pick"], r["prob"] * 100),
            "└ Duelo de Abridores: %s (%.2f) vs %s (%.2f)" % (
                r["sp_away_name"], r["siera_away"], r["sp_home_name"], r["siera_home"]),
            "└ Factor Estadio/Clima: %.3f / %s" % (r["park_factor"], r["weather_desc"]),
            "└ Ventaja Ofensiva: wRC+14d %s vs %s" % (r["wrc_away"], r["wrc_home"]),
            "└ Justificación: %s" % r["just"],
            "",
        ]
    return "\n".join(lines)


def justify(r):
    # 2 lineas: como interactua el SIERA del abridor con el park factor y el clima
    parts = []
    if r["siera_home"] < r["siera_away"] - 0.3:
        parts.append("El SIERA de %s (%.2f) marca ventaja de pitcheo local sobre %.2f." % (r["sp_home_name"], r["siera_home"], r["siera_away"]))
    elif r["siera_away"] < r["siera_home"] - 0.3:
        parts.append("El SIERA de %s (%.2f) supera al local (%.2f)." % (r["sp_away_name"], r["siera_away"], r["siera_home"]))
    else:
        parts.append("Duelo de abridores parejo en SIERA (%.2f vs %.2f)." % (r["siera_away"], r["siera_home"]))
    if r["park_factor"] > 1.05:
        parts.append("Parque pro-ofensivo (PF %.3f)" % r["park_factor"])
    elif r["park_factor"] < 0.95:
        parts.append("Parque pro-pitcheo (PF %.3f)" % r["park_factor"])
    else:
        parts.append("Parque neutral (PF %.3f)" % r["park_factor"])
    if r["wind_to_out"] > 8:
        parts.append("con viento a favor hacia el CF (%.0f mph) que enciende la ofensiva." % r["wind_to_out"])
    elif r["wind_to_out"] < 2 and r["humidity"] > 70:
        parts.append("y aire pesado/húmedo (%.0f%% HR) que frena el batazo." % r["humidity"])
    else:
        parts.append("y clima sin efecto extremo.")
    if abs(r["wrc_home"] - r["wrc_away"]) > 8:
        parts.append("wRC+14d %d vs %d inclina la balanza." % (r["wrc_home"], r["wrc_away"]))
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: hoy)")
    args = ap.parse_args()
    os.makedirs("logs", exist_ok=True)

    date_str = args.date or datetime.date.today().strftime("%Y-%m-%d")
    today = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    log.info("Iniciando prediccion para %s", date_str)

    sched = get_schedule(date_str)
    if sched.empty:
        log.warning("Sin juegos para %s", date_str)
        return

    season = today.year
    pitch_df = get_pitching_stats(season)
    team_cfg_map = load_team_config()

    # ventanas moviles terminando ayer (evita look-ahead del dia del juego)
    wrc_map = team_woba_map(today, days=14, team_park={k: v["park_factor"] for k, v in team_cfg_map.items()})
    bp_map = bullpen_usage_map(today, days=3)

    try:
        ens = Ensemble()
        have_models = True
        log.info("Modelos cargados desde models/")
    except Exception as e:
        log.warning("Sin modelos entrenados (%s). Usando fallback heuristico.", e)
        ens, have_models = None, False

    rows = []
    for _, game in sched.iterrows():
        try:
            feats, meta = build_game_features(game, pitch_df, wrc_map, bp_map, team_cfg_map, today)
        except Exception as e:
            log.error("Fallo features %s @ %s: %s", game["away_abbr"], game["home_abbr"], e)
            continue

        if have_models:
            p_home, _ = ens.predict_game(feats, lstm_seq=None)  # LSTM se omite en vivo (sin secuencia); su peso se redistribuye
        else:
            p_home = heuristic_fallback(feats)

        pick = "HOME" if p_home >= 0.5 else "AWAY"
        sp_h = meta["sp_home"]
        sp_a = meta["sp_away"]
        rows.append(dict(
            away=game["away_abbr"], home=game["home_abbr"],
            pick=pick, prob=max(p_home, 1 - p_home),
            sp_home_name=game["home_probable"], sp_away_name=game["away_probable"],
            siera_home=feats["siera_home"], siera_away=feats["siera_away"],
            park_factor=feats["park_factor"], weather_desc=meta["weather"]["desc"],
            wrc_home=int(round(feats["wrc_home14"])), wrc_away=int(round(feats["wrc_away14"])),
            wind_to_out=feats["wind_to_out"], humidity=feats["humidity"],
        ))
        rows[-1]["just"] = justify(rows[-1])

    msg = format_message(date_str, rows)
    if not have_models:
        msg += "\n<i>(Fallback heuristico: aun no hay modelos entrenados en models/)</i>"
    log.info("Enviando %d picks a Telegram", len(rows))
    send_message(msg)
    log.info("Listo.")


if __name__ == "__main__":
    main()
