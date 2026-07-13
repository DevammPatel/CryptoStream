-- Model-ready mart: features + the supervised label (next-bar up/down).
-- Label = 1 if the close 3 bars ahead (~30s) is higher than the current close.
with feats as (
    select * from {{ ref('int_technical_features') }}
),

labelled as (
    select
        *,
        lead(close, 3) over (partition by symbol order by window_start)   as future_close_3
    from feats
)

select
    symbol,
    window_start,
    close,
    ret_1,
    log_ret_1,
    px_vs_sma30,
    volatility_30,
    order_flow_imbalance,
    ofi_6,
    rel_volume,
    trade_count,
    future_close_3,
    case when future_close_3 > close then 1 else 0 end                    as label_up_3
from labelled
where future_close_3 is not null
  and volatility_30 is not null
