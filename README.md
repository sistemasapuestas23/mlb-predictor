# MLB Quant Predictor

Sistema de prediccion diaria de MLB inspirado en el analisis cuantitativo del
23 de septiembre de 2026 (contexto de estadio + clima neutralizando la
efectividad superficial).

## Pipeline diario (GitHub Actions)
1. Se dispara 30 minutos antes del primer juego de la jornada (11:34 AM ET).
2. `scripts/run_daily.py`:
   - Cartelera + abridores probables via `MLB-StatsAPI`.
   - SIERA / xFIP via `pybaseball` (FanGraphs leaderboards).
   - wRC+ ultimos 14 dias aproximado con wOBA desde `statcast` (Baseball Savant).
   - Fatiga de bullpen (lanzamientos ultimos 3 dias) via statcast.
   - Clima en tiempo real (OpenWeatherMap) o 'Entorno Controlado' en domos.
   - Park Factor desde `config/park_factors.yaml`.
   - Inferencia con ensamble: CatBoost + XGBoost + MLP + LSTM (momentum) + Poisson bivariada.
   - Envio de picks formateados a Telegram.

## Pesos del ensamble
| Modelo              | Peso |
|---------------------|------|
| CatBoostClassifier  | 0.30 |
| XGBoostClassifier   | 0.20 |
| MLP (red neuronal)  | 0.15 |
| LSTM momentum       | 0.15 |
| Poisson bivariada   | 0.20 |

## Setup
1. Crea el repo en GitHub y sube este codigo.
2. Secrets del repositorio (Settings -> Secrets -> Actions):
   - `TELEGRAM_TOKEN`      : token de @BotFather
   - `TELEGRAM_CHAT_ID`    : tu chat id
   - `OPENWEATHER_API_KEY` : api key de OpenWeatherMap (gratis)
3. Entrena la primera vez los modelos:
   ```bash
   pip install -r requirements.txt
   python scripts/train_models.py build --seasons 2024 2025 2026
   python scripts/train_models.py train
   git add models/ && git commit -m "models" && git push
   ```
   O deja que el workflow semanal `weekly_training.yml` lo haga y haga push
   automatico de `models/` al repo.

## Workflows
- `.github/workflows/daily_predictions.yml` : diario, cron `34 15 * * *` UTC
  (11:34 AM ET, ~30 min antes del primer pitch). Soporta `workflow_dispatch`
  con parametro `date` (YYYY-MM-DD) para fechas pasadas.
- `.github/workflows/weekly_training.yml`   : reentrena cada lunes 04:00 UTC y
  hace push de `models/` al repositorio.

## Avisos honestos
- Los cron de GitHub Actions pueden retrasarse varios minutos; para produccion
  critica se recomienda un runner propio (cron + self-hosted) o un disparador
  externo (scheduler + repository_dispatch).
- pybaseball hace scraping de FanGraphs: puede romperse si FanGraphs cambia el
  HTML. Hay reintentos con backoff, pero monitoriza los logs.
- El wRC+ de 14 dias se aproxima con wOBA de statcast (formula estandar
  wRC+ ~= (wOBA_team / wOBA_lg) / park_factor * 100), no es el wRC+ exacto de
  FanGraphs (que requiere datos que no son descargables por fecha).
- La primera ejecucion diaria sin `models/` usa un fallback heuristico
  (SIERA + wRC+ + park factor) y lo indica en el mensaje de Telegram.

## Disclaimer
Proyecto con fines educativos/analiticos. No es asesoria de apuestas.
