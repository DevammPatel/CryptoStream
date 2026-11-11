-- Cleaned, typed staging view over the raw 10s bars: one row per symbol and window.
-- The same bar can land in the lake twice: Spark replays a batch after a restart, and
-- the history backfill overlaps bars Spark also wrote. The most complete copy wins.
-- Keep in sync with one_bar_per_window() in ml/features.py.
with source as (
    select * from {{ source('raw', 'ohlcv_10s') }}
),

renamed as (
    select
        symbol,
        cast(window_start as timestamp)            as window_start,
        cast(window_end as timestamp)              as window_end,
        cast(open as double)                       as open,
        cast(high as double)                       as high,
        cast(low as double)                        as low,
        cast(close as double)                      as close,
        cast(volume as double)                     as volume,
        cast(vwap as double)                       as vwap,
        cast(trade_count as bigint)                as trade_count,
        cast(net_signed_volume as double)          as net_signed_volume,
        cast(order_flow_imbalance as double)       as order_flow_imbalance
    from source
    -- guard against degenerate bars
    where close > 0 and volume > 0
),

deduped as (
    select *
    from renamed
    qualify row_number() over (
        partition by symbol, window_start
        order by trade_count desc, volume desc
    ) = 1
)

select * from deduped
