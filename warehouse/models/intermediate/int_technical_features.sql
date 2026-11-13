-- Rolling technical indicators computed with window functions over the 10s bars.
-- This is the analytics feature table that both the dashboard and ML pipeline consume.
-- Keep in sync with with_indicators() in ml/features.py; a test compares the two.
--
-- Windows are measured in time, not rows, so a gap in the feed (a quiet symbol, or the
-- stack being off) never stretches "the last 30 bars" across hours, and a return only
-- exists when the previous bar is exactly 10 seconds earlier.
with base as (
    select * from {{ ref('stg_ohlcv') }}
),

with_returns as (
    select
        *,
        case when window_start - lag(window_start) over w = interval 10 seconds
             then close / nullif(lag(close) over w, 0) - 1 end          as ret_1,
        case when window_start - lag(window_start) over w = interval 10 seconds
             then ln(close / nullif(lag(close) over w, 0)) end          as log_ret_1
    from base
    window w as (partition by symbol order by window_start)
),

with_ma as (
    select
        *,
        avg(close) over w6                                             as sma_6,
        avg(close) over w30                                            as sma_30,
        stddev_samp(log_ret_1) over w30                                as volatility_30,
        avg(order_flow_imbalance) over w6                              as ofi_6,
        avg(volume) over w30                                           as avg_volume_30
    from with_returns
    window
        -- 30 bars when none are missing: this one and the 29 before it
        w30 as (partition by symbol order by window_start
                range between interval 290 seconds preceding and current row),
        -- 6 bars when none are missing
        w6 as (partition by symbol order by window_start
               range between interval 50 seconds preceding and current row)
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
