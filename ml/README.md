# Machine learning

Predicts whether a symbol's price will be higher 30 seconds from now, from its latest
10-second bar, and serves those predictions.

| File | What it does |
|---|---|
| `features.py` | The 8 feature columns and the label. Loads the training frame from the dbt mart, and holds `with_indicators()`, the Python twin of the dbt indicator SQL used for live bars. |
| `train.py` | Purged walk-forward CV, then fit (first 60% of time), calibrate (next 20%) and test (last 20%). Logs to MLflow and replaces the served model only if the new one scores at least as well on rows neither has trained on. |
| `live.py` | Keeps recent bars from the `crypto.bars_10s` Kafka topic in memory and computes their indicators. |
| `monitor.py` | The live track record: a call on every finished live bar the model never trained on, settled 30 seconds later. |
| `api/main.py` | FastAPI: `/predict`, `/predict/{symbol}`, `/model/info`, `/model/live`, `/symbols`, `/bars/{symbol}`, `/health`. |
| `experiments/cross_symbol.py` | A feature experiment, not part of the pipeline (see below). |

## The task

- **Label:** 1 if the close of the bar exactly 30 seconds later is higher than this
  bar's close (`label_up_3`). A move of exactly zero counts as 0.
- **Features:** `ret_1`, `log_ret_1`, `px_vs_sma30`, `volatility_30`,
  `order_flow_imbalance`, `ofi_6`, `rel_volume`, `trade_count`. Windows are measured in
  time (300 s and 60 s), so gaps in the feed never stretch them.
- **Model:** LightGBM, Platt-calibrated so `prob_up` means how often "up" really
  happened at that score.

On a week of real BTC, ETH and SOL bars the held-out AUC is about 0.525: a small edge
over guessing, reported next to baselines in every run. See the main README.

## Tried and not adopted

**Cross-symbol features** (`experiments/cross_symbol.py`): BTC's last two moves, the
other symbols' average move and order flow, and the symbol's move relative to all of
them, all from bars that finish at the same moment. On 9 days of real bars (208,000
rows, November 2026), with train.py's splits and settings:

| | Served features | With cross-symbol |
|---|---|---|
| Walk-forward CV AUC (5 folds) | 0.5447 | 0.5499 |
| Held-out AUC (last 20%) | 0.5349 | 0.5351 |
| Held-out Brier | 0.24807 | 0.24803 |

They helped in every CV fold, but less with each newer one: +0.0009 in the latest, and
nothing on the held-out days. Serving them would also need the join in dbt, in the live
Python path and in the API. Rerun the script as more history collects.

## Files it writes

`MODEL_DIR` (default `ml/models/`, `/models` in Docker):

| File | Contents |
|---|---|
| `direction_lgbm.pkl` | the served model and its feature list |
| `model_meta.json` | the served model's metrics, feature importance, `trained_through` and promotion verdict |
| `last_training.json` | the same for the latest run, promoted or not |
| `live_track_record.json` | the API's live track record, saved every minute |

None of these are tracked in git; `make train` or the Dagster pipeline creates them.

## Running

```bash
make train   # needs a warehouse: make seed && make dbt, or a backfill + make refresh
make api     # http://localhost:8000/docs
```
