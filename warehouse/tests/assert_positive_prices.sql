-- Singular data-quality test: no non-positive prices should reach the ML mart.
select symbol, window_start, close
from {{ ref('mart_ml_features') }}
where close <= 0
