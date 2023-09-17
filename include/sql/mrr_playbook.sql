--------------------------------------
-- mrr playbook
--------------------------------------
SET day_count = (select datediff('days', to_date('2018-01-011'), current_date));
with subscription_periods as (
    select subscription_id, 
           customer_id, 
           cast(start_date as date) as start_date, 
           cast(end_date as date) as end_date, 
           monthly_amount 
        from stg_subscription_periods),
months as (    
    SELECT distinct(date_trunc('month', DATEADD(day, row_number() OVER (order by seq4())-1, TO_DATE('2018-01-01')))) AS date_month
    FROM table(generator(rowCount => $day_count))),
customers_list as (
    select
        customer_id,
        date_trunc('month', min(start_date)) as date_month_start,
        date_trunc('month', max(end_date)) as date_month_end
    from subscription_periods
    group by 1),
customer_months as (
    select
        customers_list.customer_id,
        months.date_month
    from customers_list
    inner join months
        on  months.date_month >= customers_list.date_month_start
        and months.date_month < customers_list.date_month_end),
joined as (
    select
        customer_months.date_month,
        customer_months.customer_id,
        coalesce(subscription_periods.monthly_amount, 0) as mrr
    from customer_months
    left join subscription_periods
        on customer_months.customer_id = subscription_periods.customer_id
        and customer_months.date_month >= subscription_periods.start_date
        and (customer_months.date_month < subscription_periods.end_date
                or subscription_periods.end_date is null)),
customer_revenue_by_month as (
    select
        date_month,
        customer_id,
        mrr,
        mrr > 0 as is_active,
        min(case when mrr > 0 then date_month end) over (
            partition by customer_id
        ) as first_active_month,
        max(case when mrr > 0 then date_month end) over (
            partition by customer_id
        ) as last_active_month,
        case
          when min(case when mrr > 0 then date_month end) over (
            partition by customer_id
        ) = date_month then true
          else false end as is_first_month,
        case
          when max(case when mrr > 0 then date_month end) over (
            partition by customer_id
        ) = date_month then true
          else false end as is_last_month
    from joined),
customer_churn_month as (
    select
        date_month + interval '1 month' as date_month,
        customer_id,
        0::float as mrr,
        false as is_active,
        first_active_month,
        last_active_month,
        false as is_first_month,
        false as is_last_month
    from customer_revenue_by_month
    where is_last_month
),
unioned as (
    select * from customer_revenue_by_month union all select * from customer_churn_month
),
mrr_with_changes as (
    select
        *,
        coalesce(lag(is_active) over (partition by customer_id order by date_month), false) as previous_month_is_active,
        coalesce(lag(mrr) over (partition by customer_id order by date_month), 0) as previous_month_mrr,
        mrr - coalesce(lag(mrr) over (partition by customer_id order by date_month), 0) as mrr_change
    from unioned),
mrr as (
    select
        uuid_string() as id,
        *,
        case
            when is_first_month
                then 'new'
            when not(is_active) and previous_month_is_active
                then 'churn'
            when is_active and not(previous_month_is_active)
                then 'reactivation'
            when mrr_change > 0 then 'upgrade'
            when mrr_change < 0 then 'downgrade'
        end as change_category,
        least(mrr, previous_month_mrr) as renewal_amount
    from mrr_with_changes),

create table mrr as select * from mrr;