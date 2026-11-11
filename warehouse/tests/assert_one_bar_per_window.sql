-- Singular data-quality test: staging must hold at most one bar per symbol and window.
select symbol, window_start, count(*) as copies
from {{ ref('stg_ohlcv') }}
group by 1, 2
having count(*) > 1
