-- Rolling technical indicators computed with window functions over the 10s bars.
-- This is the analytics feature table that both the dashboard and ML pipeline consume.
with base as (
    select * from {{ ref('stg_ohlcv') }}
),

with_returns as (
    select
        *,
        close / nullif(lag(close) over w, 0) - 1                        as ret_1,
        ln(close / nullif(lag(close) over w, 0))                        as log_ret_1
    from base
    window w as (partition by symbol order by window_start)
),

with_ma as (
    select
        *,
        avg(close) over (partition by symbol order by window_start
            rows between 5 preceding and current row)                  as sma_6,
        avg(close) over (partition by symbol order by window_start
            rows between 29 preceding and current row)                 as sma_30,
        stddev_samp(log_ret_1) over (partition by symbol order by window_start
            rows between 29 preceding and current row)                 as volatility_30,
        avg(order_flow_imbalance) over (partition by symbol order by window_start
            rows between 5 preceding and current row)                  as ofi_6,
        avg(volume) over (partition by symbol order by window_start
            rows between 29 preceding and current row)                 as avg_volume_30
    from with_returns
)

select
    symbol,
    window_start,
    window_end,
    open, high, low, close, volume, vwap, trade_count,
    net_signed_volume, order_flow_imbalance,
    ret_1, log_ret_1,
    sma_6, sma_30,
    close / nullif(sma_30, 0) - 1                                       as px_vs_sma30,
    volatility_30,
    ofi_6,
    volume / nullif(avg_volume_30, 0)                                   as rel_volume
from with_ma
