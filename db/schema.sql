-- ============================================================================
-- Digikala Customer Churn Prediction - Database Schema
-- PostgreSQL 15+
-- ============================================================================
--
-- Schema Design Decisions (based on EDA):
--
-- 1. orders.order_id is UNIQUE -> PRIMARY KEY
-- 2. crm has 1:1 relationship with orders -> order_id as PK and FK
-- 3. comments can have multiple per order -> needs auto-increment ID
-- 4. CRM.order_date is redundant -> NOT included in schema
-- 5. Orphan comments (5,037) will be filtered during data load
-- 6. is_otd has values {-1, 0, 1} -> SMALLINT, -1 means unknown/pending
--
-- ============================================================================

-- Drop existing tables (in correct order due to FK constraints)
DROP TABLE IF EXISTS comments CASCADE;
DROP TABLE IF EXISTS crm CASCADE;
DROP TABLE IF EXISTS orders CASCADE;

-- ============================================================================
-- TABLE: orders (Parent table - 2,720,059 rows)
-- ============================================================================
-- Contains order transactions with delivery information
-- order_id is unique and serves as PRIMARY KEY

CREATE TABLE orders (
    order_id        BIGINT      PRIMARY KEY,
    user_id         BIGINT      NOT NULL,
    is_otd          SMALLINT    NULL,  -- Values: -1 (unknown), 0 (late), 1 (on-time)
    order_date      DATE        NOT NULL,
    delivery_status VARCHAR(50) NULL   -- ~1.6% NULL is acceptable
);

-- Indexes for common query patterns
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_order_date ON orders(order_date);
CREATE INDEX idx_orders_user_date ON orders(user_id, order_date);

-- Comments
COMMENT ON TABLE orders IS 'Order transactions - parent table for CRM and comments';
COMMENT ON COLUMN orders.order_id IS 'Unique order identifier (PK)';
COMMENT ON COLUMN orders.user_id IS 'Customer identifier - multiple orders per user possible';
COMMENT ON COLUMN orders.is_otd IS 'On-time delivery: 1=on-time, 0=late, -1=unknown/pending';
COMMENT ON COLUMN orders.order_date IS 'Date when order was placed';
COMMENT ON COLUMN orders.delivery_status IS 'Delivery status text (e.g., delivered, pending)';


-- ============================================================================
-- TABLE: crm (Child table - 2,720,059 rows, 1:1 with orders)
-- ============================================================================
-- Customer relationship data: complaints and ratings per order
-- order_id is both PK (unique) and FK (references orders)
-- NOTE: order_date column from CSV is NOT included (redundant with orders.order_date)

CREATE TABLE crm (
    order_id                        BIGINT      PRIMARY KEY,
    crm_delivery_request_count      SMALLINT    NOT NULL DEFAULT 0,
    crm_fake_delivery_request_count SMALLINT    NOT NULL DEFAULT 0,
    rate_to_shop                    DECIMAL(2,1) NULL,  -- Range: 1.0-5.0, ~58% NULL
    rate_to_courier                 DECIMAL(2,1) NULL,  -- Range: 1.0-5.0, ~74% NULL

    -- Foreign key constraint
    CONSTRAINT fk_crm_orders
        FOREIGN KEY (order_id)
        REFERENCES orders(order_id)
        ON DELETE CASCADE
);

-- Constraints for data validation
ALTER TABLE crm ADD CONSTRAINT chk_crm_delivery_count
    CHECK (crm_delivery_request_count >= 0);

ALTER TABLE crm ADD CONSTRAINT chk_crm_fake_count
    CHECK (crm_fake_delivery_request_count >= 0);

ALTER TABLE crm ADD CONSTRAINT chk_rate_shop
    CHECK (rate_to_shop IS NULL OR (rate_to_shop >= 1.0 AND rate_to_shop <= 5.0));

ALTER TABLE crm ADD CONSTRAINT chk_rate_courier
    CHECK (rate_to_courier IS NULL OR (rate_to_courier >= 1.0 AND rate_to_courier <= 5.0));

-- Comments
COMMENT ON TABLE crm IS 'CRM data: complaints and ratings per order (1:1 with orders)';
COMMENT ON COLUMN crm.order_id IS 'Order identifier - PK and FK to orders';
COMMENT ON COLUMN crm.crm_delivery_request_count IS 'Number of delivery complaint tickets';
COMMENT ON COLUMN crm.crm_fake_delivery_request_count IS 'Number of fake delivery complaint tickets';
COMMENT ON COLUMN crm.rate_to_shop IS 'Customer rating for shop (1-5), NULL if not rated';
COMMENT ON COLUMN crm.rate_to_courier IS 'Customer rating for courier (1-5), NULL if not rated';


-- ============================================================================
-- TABLE: comments (Child table - ~89K rows after filtering orphans)
-- ============================================================================
-- Customer comments/feedback for orders
-- One order can have multiple comments -> needs auto-increment ID
-- NOTE: Orphan comments (order_id not in orders) will be filtered during load

CREATE TABLE comments (
    id              SERIAL      PRIMARY KEY,
    order_id        BIGINT      NOT NULL,
    description     TEXT        NULL,

    -- Foreign key constraint
    CONSTRAINT fk_comments_orders
        FOREIGN KEY (order_id)
        REFERENCES orders(order_id)
        ON DELETE CASCADE
);

-- Index for FK and common queries
CREATE INDEX idx_comments_order_id ON comments(order_id);

-- Comments
COMMENT ON TABLE comments IS 'Customer comments per order (1:N with orders)';
COMMENT ON COLUMN comments.id IS 'Auto-increment primary key';
COMMENT ON COLUMN comments.order_id IS 'FK to orders - one order can have multiple comments';
COMMENT ON COLUMN comments.description IS 'Free-text customer comment (Persian text)';


-- ============================================================================
-- USEFUL VIEWS FOR ANALYSIS
-- ============================================================================

-- View: User-level order summary
CREATE OR REPLACE VIEW v_user_order_summary AS
SELECT
    user_id,
    COUNT(*) as total_orders,
    MIN(order_date) as first_order_date,
    MAX(order_date) as last_order_date,
    SUM(CASE WHEN is_otd = 1 THEN 1 ELSE 0 END) as on_time_orders,
    SUM(CASE WHEN is_otd = 0 THEN 1 ELSE 0 END) as late_orders,
    SUM(CASE WHEN is_otd = -1 THEN 1 ELSE 0 END) as unknown_otd_orders,
    ROUND(
        AVG(CASE WHEN is_otd IN (0, 1) THEN is_otd::NUMERIC ELSE NULL END),
        3
    ) as on_time_ratio
FROM orders
GROUP BY user_id;

COMMENT ON VIEW v_user_order_summary IS 'Aggregated order statistics per user';


-- View: User-level CRM summary
CREATE OR REPLACE VIEW v_user_crm_summary AS
SELECT
    o.user_id,
    SUM(c.crm_delivery_request_count) as total_delivery_requests,
    SUM(c.crm_fake_delivery_request_count) as total_fake_requests,
    ROUND(AVG(c.rate_to_shop), 2) as avg_shop_rating,
    ROUND(AVG(c.rate_to_courier), 2) as avg_courier_rating,
    COUNT(c.rate_to_shop) as shop_rating_count,
    COUNT(c.rate_to_courier) as courier_rating_count
FROM orders o
JOIN crm c ON o.order_id = c.order_id
GROUP BY o.user_id;

COMMENT ON VIEW v_user_crm_summary IS 'Aggregated CRM statistics per user';


-- View: User-level comments summary
CREATE OR REPLACE VIEW v_user_comments_summary AS
SELECT
    o.user_id,
    COUNT(c.id) as total_comments,
    COUNT(CASE WHEN c.description IS NOT NULL AND LENGTH(c.description) > 0 THEN 1 END) as non_empty_comments
FROM orders o
LEFT JOIN comments c ON o.order_id = c.order_id
GROUP BY o.user_id;

COMMENT ON VIEW v_user_comments_summary IS 'Aggregated comment statistics per user';


-- ============================================================================
-- VERIFICATION QUERIES (run after data load)
-- ============================================================================
/*
-- Check row counts
SELECT 'orders' as table_name, COUNT(*) as row_count FROM orders
UNION ALL
SELECT 'crm', COUNT(*) FROM crm
UNION ALL
SELECT 'comments', COUNT(*) FROM comments;

-- Check is_otd distribution
SELECT is_otd, COUNT(*) as count,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM orders
GROUP BY is_otd
ORDER BY is_otd;

-- Check delivery_status values
SELECT delivery_status, COUNT(*) as count
FROM orders
GROUP BY delivery_status
ORDER BY count DESC;

-- Check rating distributions
SELECT
    rate_to_shop,
    COUNT(*) as count
FROM crm
WHERE rate_to_shop IS NOT NULL
GROUP BY rate_to_shop
ORDER BY rate_to_shop;

-- Verify FK integrity
SELECT COUNT(*) as orphan_crm_records
FROM crm c
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = c.order_id);

SELECT COUNT(*) as orphan_comment_records
FROM comments c
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = c.order_id);
*/