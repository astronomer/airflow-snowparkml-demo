--------------------------------------
-- jaffle shop
--------------------------------------

with
  customers as (select * from stg_customers),
  orders as (select * from stg_orders),
  payments as (select * from stg_payments),
  customer_orders as (
    select
      customer_id,
      min(order_date) as first_order,
      max(order_date) as most_recent_order,
      count(order_id) as number_of_orders
    from orders
    group by customer_id),
  customer_payments as (
    select
      orders.customer_id,
      sum(amount / 100) as total_amount
    from payments
    left join orders on payments.order_id = orders.order_id
    group by orders.customer_id),
  customers as (
    select
      customers.customer_id,
      customers.first_name,
      customers.last_name,
      customer_orders.first_order,
      customer_orders.most_recent_order,
      customer_orders.number_of_orders,
      customer_payments.total_amount as customer_lifetime_value
    from customers
    left join customer_orders on customers.customer_id = customer_orders.customer_id
    left join customer_payments on customers.customer_id = customer_payments.customer_id
  ),
  order_payments as (
    select
        order_id,
        sum(case when payment_method = 'credit_card' then amount / 100 else 0 end) as credit_card_amount,
        sum(case when payment_method = 'coupon' then amount / 100 else 0 end) as coupon_amount,
        sum(case when payment_method = 'bank_transfer' then amount / 100 else 0 end) as bank_transfer_amount,
        sum(case when payment_method = 'gift_card' then amount / 100 else 0 end) as gift_card_amount,
        sum(amount / 100) as total_amount
    from payments
    group by order_id),
orders as (
    select
        orders.order_id,
        orders.customer_id,
        orders.order_date,
        orders.status,
        order_payments.credit_card_amount,
        order_payments.coupon_amount,
        order_payments.bank_transfer_amount,
        order_payments.gift_card_amount,
        order_payments.total_amount as amount
    from orders
    left join order_payments
        on orders.order_id = order_payments.order_id
),

create table customers as select * from customers;