-- Model-ready mart: features + the supervised label.
-- Label = 1 if the close of the bar exactly 30s later is higher than this close. Rows
-- with no bar at +30s (the end of the data, or a gap) get no label and are left out,
-- rather than borrowing whichever bar happens to come three rows later.
with feats as (
    select * from {{ ref('int_technical_features') }}
),

labelled as (
    select
        f.*,
        future.close                                                    as future_close_3
    from feats f
    left join feats future
        on future.symbol = f.symbol
       and future.window_start = f.window_start + interval 30 seconds
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
