-- Latest snapshot per symbol — powers the dashboard KPI tiles and the LLM agent.
with feats as (
    select * from {{ ref('int_technical_features') }}
),

ranked as (
    select
        *,
        row_number() over (partition by symbol order by window_start desc) as rn
    from feats
),

daily as (
    select
        symbol,
        max(high)                                       as high_window,
        min(low)                                        as low_window,
        sum(volume)                                     as total_volume,
        count(*)                                        as bars
    from feats
    group by 1
)

select
    r.symbol,
    r.window_start                                       as as_of,
    r.close                                              as last_price,
    r.ret_1                                              as last_return,
    r.px_vs_sma30,
    r.volatility_30,
    r.order_flow_imbalance,
    r.rel_volume,
    d.high_window,
    d.low_window,
    d.total_volume,
    d.bars
from ranked r
join daily d using (symbol)
where r.rn = 1
