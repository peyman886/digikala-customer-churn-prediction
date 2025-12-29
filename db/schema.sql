-- =====================================================
-- Digikala Churn Prediction Database Schema
-- PostgreSQL 15+
-- Author: Peyman
-- =====================================================

-- Drop existing tables if they exist
DROP TABLE IF EXISTS comments CASCADE;
DROP TABLE IF EXISTS crm CASCADE;
DROP TABLE IF EXISTS orders CASCADE;

-- =====================================================
-- ORDERS TABLE (Main Fact Table)
-- =====================================================
CREATE TABLE orders (
    order_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    is_otd BOOLEAN,
    order_date DATE NOT NULL,
    delivery_status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_order_date ON orders(order_date);
CREATE INDEX idx_orders_user_date ON orders(user_id, order_date);

-- =====================================================
-- CRM TABLE (Customer Relationship Management)
-- =====================================================
CREATE TABLE crm (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) NOT NULL UNIQUE,
    crm_delivery_request_count INTEGER DEFAULT 0,
    crm_fake_delivery_request_count INTEGER DEFAULT 0,
    rate_to_shop DECIMAL(3,2),
    rate_to_courier DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_crm_order
        FOREIGN KEY (order_id) 
        REFERENCES orders(order_id)
        ON DELETE CASCADE
);

-- Create index for foreign key
CREATE INDEX idx_crm_order_id ON crm(order_id);

-- =====================================================
-- COMMENTS TABLE (Customer Feedback)
-- =====================================================
CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_comments_order
        FOREIGN KEY (order_id) 
        REFERENCES orders(order_id)
        ON DELETE CASCADE
);

-- Create index for foreign key
CREATE INDEX idx_comments_order_id ON comments(order_id);

-- =====================================================
-- VIEWS FOR ANALYTICS
-- =====================================================

-- User Order Summary View
CREATE OR REPLACE VIEW v_user_order_summary AS
SELECT 
    user_id,
    COUNT(DISTINCT order_id) as total_orders,
    MAX(order_date) as last_order_date,
    MIN(order_date) as first_order_date,
    ROUND(AVG(CASE WHEN is_otd THEN 1 ELSE 0 END)::numeric, 2) as on_time_delivery_rate,
    COUNT(CASE WHEN is_otd THEN 1 END) as on_time_orders,
    COUNT(CASE WHEN NOT is_otd THEN 1 END) as late_orders
FROM orders
GROUP BY user_id;

-- User CRM Summary View
CREATE OR REPLACE VIEW v_user_crm_summary AS
SELECT 
    o.user_id,
    COUNT(DISTINCT c.order_id) as orders_with_crm,
    SUM(c.crm_delivery_request_count) as total_delivery_requests,
    SUM(c.crm_fake_delivery_request_count) as total_fake_requests,
    ROUND(AVG(c.rate_to_shop)::numeric, 2) as avg_shop_rating,
    ROUND(AVG(c.rate_to_courier)::numeric, 2) as avg_courier_rating
FROM orders o
LEFT JOIN crm c ON o.order_id = c.order_id
GROUP BY o.user_id;

-- User Comments Summary View
CREATE OR REPLACE VIEW v_user_comments_summary AS
SELECT 
    o.user_id,
    COUNT(DISTINCT co.id) as total_comments,
    COUNT(CASE WHEN co.description IS NOT NULL AND LENGTH(co.description) > 0 THEN 1 END) as comments_with_text
FROM orders o
LEFT JOIN comments co ON o.order_id = co.order_id
GROUP BY o.user_id;

-- =====================================================
-- HELPER FUNCTIONS
-- =====================================================

-- Function to calculate days since last order
CREATE OR REPLACE FUNCTION days_since_last_order(p_user_id VARCHAR)
RETURNS INTEGER AS $$
DECLARE
    last_date DATE;
    days_diff INTEGER;
BEGIN
    SELECT MAX(order_date) INTO last_date
    FROM orders
    WHERE user_id = p_user_id;
    
    IF last_date IS NULL THEN
        RETURN NULL;
    END IF;
    
    days_diff := CURRENT_DATE - last_date;
    RETURN days_diff;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- DATA VALIDATION CONSTRAINTS
-- =====================================================

-- Ensure ratings are between 0 and 5
ALTER TABLE crm ADD CONSTRAINT check_shop_rating 
    CHECK (rate_to_shop IS NULL OR (rate_to_shop >= 0 AND rate_to_shop <= 5));

ALTER TABLE crm ADD CONSTRAINT check_courier_rating 
    CHECK (rate_to_courier IS NULL OR (rate_to_courier >= 0 AND rate_to_courier <= 5));

-- Ensure counts are non-negative
ALTER TABLE crm ADD CONSTRAINT check_delivery_requests 
    CHECK (crm_delivery_request_count >= 0);

ALTER TABLE crm ADD CONSTRAINT check_fake_requests 
    CHECK (crm_fake_delivery_request_count >= 0);

-- =====================================================
-- COMMENTS
-- =====================================================

COMMENT ON TABLE orders IS 'Main orders table containing all customer orders';
COMMENT ON TABLE crm IS 'Customer relationship management data for each order';
COMMENT ON TABLE comments IS 'Customer comments and feedback for orders';

COMMENT ON COLUMN orders.order_id IS 'Unique identifier for each order';
COMMENT ON COLUMN orders.user_id IS 'Unique identifier for each user/customer';
COMMENT ON COLUMN orders.is_otd IS 'Boolean flag indicating if order was delivered on time';
COMMENT ON COLUMN orders.order_date IS 'Date when the order was placed';
COMMENT ON COLUMN orders.delivery_status IS 'Current delivery status of the order';

-- =====================================================
-- SAMPLE QUERY FOR TESTING
-- =====================================================

-- Query to test the schema after data loading:
/*
SELECT 
    o.total_orders,
    o.on_time_delivery_rate,
    c.total_delivery_requests,
    c.avg_shop_rating,
    co.total_comments
FROM v_user_order_summary o
LEFT JOIN v_user_crm_summary c ON o.user_id = c.user_id
LEFT JOIN v_user_comments_summary co ON o.user_id = co.user_id
LIMIT 10;
*/
