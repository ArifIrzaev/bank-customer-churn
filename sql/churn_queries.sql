-- =====================================================================
-- Отток клиентов банка: ключевые расчёты на SQL (SQLite)
--
-- Таблица: customers (одна строка = один клиент), её создаёт ноутбук
-- notebooks/churn_analysis.ipynb в файле data/churn.db.
--
-- Столбцы: customer_id, credit_score, geography, gender, age, tenure,
-- balance, num_products, has_cr_card, is_active, est_salary, exited,
-- complain, satisfaction, card_type, points
--
-- Каждый запрос начинается с маркера "-- name: <имя>": по нему ноутбук
-- находит запрос, выполняет его и сверяет результат с pandas.
-- Отток = AVG(exited), потому что exited принимает только значения 0/1.
-- =====================================================================


-- name: q01_overall
-- Общий отток: сколько клиентов, сколько ушло, какая доля
SELECT
    COUNT(*)                        AS clients,
    SUM(exited)                     AS churned,
    ROUND(AVG(exited) * 100, 2)     AS churn_rate_pct
FROM customers;


-- name: q02_by_geography
-- Отток по странам
SELECT
    geography,
    COUNT(*)                        AS clients,
    SUM(exited)                     AS churned,
    ROUND(AVG(exited) * 100, 2)     AS churn_rate_pct
FROM customers
GROUP BY geography
ORDER BY churn_rate_pct DESC;


-- name: q03_by_age_group
-- Отток по возрастным группам: группы строим через CASE WHEN
SELECT
    CASE
        WHEN age < 30 THEN '18-29'
        WHEN age < 40 THEN '30-39'
        WHEN age < 50 THEN '40-49'
        WHEN age < 60 THEN '50-59'
        ELSE '60+'
    END                             AS age_group,
    COUNT(*)                        AS clients,
    SUM(exited)                     AS churned,
    ROUND(AVG(exited) * 100, 2)     AS churn_rate_pct
FROM customers
GROUP BY age_group
ORDER BY age_group;


-- name: q04_by_products
-- Отток по числу продуктов (3 и 4 объединены: клиентов с 4 продуктами всего 60)
SELECT
    CASE WHEN num_products >= 3 THEN '3-4'
         ELSE CAST(num_products AS TEXT)
    END                             AS products_grp,
    COUNT(*)                        AS clients,
    SUM(exited)                     AS churned,
    ROUND(AVG(exited) * 100, 2)     AS churn_rate_pct
FROM customers
GROUP BY products_grp
ORDER BY products_grp;


-- name: q05_by_activity
-- Активные и неактивные клиенты
SELECT
    is_active,
    COUNT(*)                        AS clients,
    SUM(exited)                     AS churned,
    ROUND(AVG(exited) * 100, 2)     AS churn_rate_pct
FROM customers
GROUP BY is_active
ORDER BY is_active;


-- name: q06_by_balance
-- Нулевой и ненулевой баланс
SELECT
    CASE WHEN balance = 0 THEN 'zero' ELSE 'non_zero' END AS balance_grp,
    COUNT(*)                        AS clients,
    SUM(exited)                     AS churned,
    ROUND(AVG(exited) * 100, 2)     AS churn_rate_pct
FROM customers
GROUP BY balance_grp
ORDER BY balance_grp;


-- name: q07_by_credit_card
-- Наличие кредитной карты
SELECT
    has_cr_card,
    COUNT(*)                        AS clients,
    SUM(exited)                     AS churned,
    ROUND(AVG(exited) * 100, 2)     AS churn_rate_pct
FROM customers
GROUP BY has_cr_card
ORDER BY has_cr_card;


-- name: q08_risk_matrix
-- Матрица риска: возраст x активность (условная агрегация через CASE WHEN)
SELECT
    CASE
        WHEN age < 30 THEN '18-29'
        WHEN age < 40 THEN '30-39'
        WHEN age < 50 THEN '40-49'
        WHEN age < 60 THEN '50-59'
        ELSE '60+'
    END                                                         AS age_group,
    ROUND(AVG(CASE WHEN is_active = 1 THEN exited END) * 100, 1) AS active_churn_pct,
    ROUND(AVG(CASE WHEN is_active = 0 THEN exited END) * 100, 1) AS inactive_churn_pct,
    ROUND(
        AVG(CASE WHEN is_active = 0 THEN exited END)
        / AVG(CASE WHEN is_active = 1 THEN exited END), 2)       AS inactive_to_active_ratio
FROM customers
GROUP BY age_group
ORDER BY age_group;


-- name: q09_top_risk_segments
-- Топ-10 сегментов страна x возраст x активность по оттоку.
-- CTE считает сегменты, подзапросы дают средний отток и общее число ушедших,
-- HAVING отсекает слишком маленькие сегменты (меньше 100 клиентов).
WITH segments AS (
    SELECT
        geography,
        CASE
            WHEN age < 30 THEN '18-29'
            WHEN age < 40 THEN '30-39'
            WHEN age < 50 THEN '40-49'
            WHEN age < 60 THEN '50-59'
            ELSE '60+'
        END                         AS age_group,
        CASE WHEN is_active = 1 THEN 'active' ELSE 'inactive' END AS activity,
        COUNT(*)                    AS clients,
        SUM(exited)                 AS churned,
        AVG(exited)                 AS churn_rate
    FROM customers
    GROUP BY geography, age_group, activity
    HAVING COUNT(*) >= 100
)
SELECT
    geography,
    age_group,
    activity,
    clients,
    churned,
    ROUND(churn_rate * 100, 1)                                        AS churn_rate_pct,
    ROUND(churn_rate / (SELECT AVG(exited) FROM customers), 2)        AS lift_vs_avg,
    ROUND(100.0 * churned / (SELECT SUM(exited) FROM customers), 1)   AS share_of_all_churned_pct
FROM segments
ORDER BY churn_rate DESC
LIMIT 10;


-- name: q10_segment_rank
-- Оконная функция: ранжируем возрастные группы по оттоку внутри каждой страны
WITH g AS (
    SELECT
        geography,
        CASE
            WHEN age < 30 THEN '18-29'
            WHEN age < 40 THEN '30-39'
            WHEN age < 50 THEN '40-49'
            WHEN age < 60 THEN '50-59'
            ELSE '60+'
        END                         AS age_group,
        COUNT(*)                    AS clients,
        ROUND(AVG(exited) * 100, 1) AS churn_rate_pct
    FROM customers
    GROUP BY geography, age_group
)
SELECT
    geography,
    age_group,
    clients,
    churn_rate_pct,
    RANK() OVER (PARTITION BY geography ORDER BY churn_rate_pct DESC) AS rank_in_country
FROM g
ORDER BY geography, rank_in_country;


-- name: q11_complain_check
-- Проверка на утечку: как связаны жалоба и уход
SELECT
    complain,
    COUNT(*)                        AS clients,
    SUM(exited)                     AS churned,
    ROUND(AVG(exited) * 100, 2)     AS churn_rate_pct
FROM customers
GROUP BY complain
ORDER BY complain;
