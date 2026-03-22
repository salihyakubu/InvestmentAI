-- Enable extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Enums
CREATE TYPE asset_class AS ENUM ('stock', 'crypto');
CREATE TYPE order_side AS ENUM ('buy', 'sell');
CREATE TYPE order_type AS ENUM ('market', 'limit', 'stop', 'stop_limit', 'twap', 'vwap');
CREATE TYPE order_status AS ENUM (
    'pending', 'submitted', 'partial_fill', 'filled',
    'cancelled', 'rejected', 'expired'
);
CREATE TYPE trading_mode AS ENUM ('backtest', 'paper', 'live');
CREATE TYPE signal_direction AS ENUM ('long', 'short', 'flat');

-- OHLCV market data (hypertable)
CREATE TABLE ohlcv (
    time         TIMESTAMPTZ    NOT NULL,
    symbol       VARCHAR(20)    NOT NULL,
    asset_class  asset_class    NOT NULL,
    open         NUMERIC(20,8)  NOT NULL,
    high         NUMERIC(20,8)  NOT NULL,
    low          NUMERIC(20,8)  NOT NULL,
    close        NUMERIC(20,8)  NOT NULL,
    volume       NUMERIC(20,8)  NOT NULL,
    vwap         NUMERIC(20,8),
    trade_count  INTEGER,
    timeframe    VARCHAR(5)     NOT NULL DEFAULT '1d',
    source       VARCHAR(30)    NOT NULL,
    ingested_at  TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);
SELECT create_hypertable('ohlcv', 'time');
CREATE INDEX idx_ohlcv_symbol_time ON ohlcv (symbol, time DESC);
CREATE INDEX idx_ohlcv_timeframe ON ohlcv (timeframe, time DESC);

-- Feature store (hypertable)
CREATE TABLE features (
    time         TIMESTAMPTZ      NOT NULL,
    symbol       VARCHAR(20)      NOT NULL,
    feature_name VARCHAR(100)     NOT NULL,
    value        DOUBLE PRECISION NOT NULL,
    version      INTEGER          NOT NULL DEFAULT 1
);
SELECT create_hypertable('features', 'time');
CREATE INDEX idx_features_symbol ON features (symbol, feature_name, time DESC);

-- Predictions (hypertable - composite PK includes time for partitioning)
CREATE TABLE predictions (
    id               UUID             NOT NULL DEFAULT gen_random_uuid(),
    time             TIMESTAMPTZ      NOT NULL,
    symbol           VARCHAR(20)      NOT NULL,
    model_id         VARCHAR(100)     NOT NULL,
    model_version    INTEGER          NOT NULL,
    direction        signal_direction NOT NULL,
    confidence       DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    expected_return  DOUBLE PRECISION,
    horizon_minutes  INTEGER          NOT NULL,
    features_used    JSONB,
    explanation      JSONB,
    created_at       TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, time)
);
SELECT create_hypertable('predictions', 'time');
CREATE INDEX idx_predictions_symbol ON predictions (symbol, time DESC);

-- Orders
CREATE TABLE orders (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id      VARCHAR(100),
    symbol           VARCHAR(20)    NOT NULL,
    asset_class      asset_class    NOT NULL,
    side             order_side     NOT NULL,
    order_type       order_type     NOT NULL,
    quantity         NUMERIC(20,8)  NOT NULL,
    limit_price      NUMERIC(20,8),
    stop_price       NUMERIC(20,8),
    status           order_status   NOT NULL DEFAULT 'pending',
    trading_mode     trading_mode   NOT NULL,
    prediction_id    UUID,
    parent_order_id  UUID REFERENCES orders(id),
    algo_type        VARCHAR(20),
    submitted_at     TIMESTAMPTZ,
    filled_at        TIMESTAMPTZ,
    cancelled_at     TIMESTAMPTZ,
    created_at       TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_orders_symbol_status ON orders (symbol, status);
CREATE INDEX idx_orders_created ON orders (created_at DESC);

-- Fills
CREATE TABLE fills (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id         UUID NOT NULL REFERENCES orders(id),
    external_fill_id VARCHAR(100),
    price            NUMERIC(20,8)    NOT NULL,
    quantity         NUMERIC(20,8)    NOT NULL,
    commission       NUMERIC(20,8)    NOT NULL DEFAULT 0,
    slippage_bps     DOUBLE PRECISION,
    filled_at        TIMESTAMPTZ      NOT NULL,
    created_at       TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_fills_order ON fills (order_id);

-- Positions
CREATE TABLE positions (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol            VARCHAR(20)    NOT NULL,
    asset_class       asset_class    NOT NULL,
    side              order_side     NOT NULL,
    quantity          NUMERIC(20,8)  NOT NULL,
    avg_entry_price   NUMERIC(20,8)  NOT NULL,
    current_price     NUMERIC(20,8),
    unrealized_pnl    NUMERIC(20,8),
    realized_pnl      NUMERIC(20,8)  NOT NULL DEFAULT 0,
    stop_loss_price   NUMERIC(20,8),
    trailing_stop_pct DOUBLE PRECISION,
    highest_price     NUMERIC(20,8),
    opened_at         TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    closed_at         TIMESTAMPTZ,
    updated_at        TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_positions_open ON positions (symbol) WHERE closed_at IS NULL;

-- Portfolio snapshots (hypertable)
CREATE TABLE portfolio_snapshots (
    time             TIMESTAMPTZ      NOT NULL,
    total_equity     NUMERIC(20,8)    NOT NULL,
    cash             NUMERIC(20,8)    NOT NULL,
    positions_value  NUMERIC(20,8)    NOT NULL,
    unrealized_pnl   NUMERIC(20,8)    NOT NULL,
    realized_pnl     NUMERIC(20,8)    NOT NULL,
    daily_return_pct DOUBLE PRECISION,
    max_drawdown_pct DOUBLE PRECISION,
    sharpe_ratio     DOUBLE PRECISION,
    position_count   INTEGER          NOT NULL,
    allocations      JSONB,
    trading_mode     trading_mode     NOT NULL
);
SELECT create_hypertable('portfolio_snapshots', 'time');

-- Risk metrics (hypertable)
CREATE TABLE risk_metrics (
    time                   TIMESTAMPTZ      NOT NULL,
    var_95                 DOUBLE PRECISION,
    var_99                 DOUBLE PRECISION,
    cvar_95                DOUBLE PRECISION,
    cvar_99                DOUBLE PRECISION,
    max_drawdown           DOUBLE PRECISION,
    current_drawdown       DOUBLE PRECISION,
    beta                   DOUBLE PRECISION,
    correlation_max        DOUBLE PRECISION,
    concentration_max      DOUBLE PRECISION,
    circuit_breaker_active BOOLEAN          NOT NULL DEFAULT FALSE,
    details                JSONB
);
SELECT create_hypertable('risk_metrics', 'time');

-- Model metadata
CREATE TABLE model_metadata (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name          VARCHAR(100)  NOT NULL,
    model_type          VARCHAR(50)   NOT NULL,
    version             INTEGER       NOT NULL,
    hyperparameters     JSONB         NOT NULL,
    training_metrics    JSONB,
    validation_metrics  JSONB,
    feature_importance  JSONB,
    artifact_path       VARCHAR(500)  NOT NULL,
    trained_at          TIMESTAMPTZ   NOT NULL,
    training_data_start TIMESTAMPTZ,
    training_data_end   TIMESTAMPTZ,
    is_active           BOOLEAN       NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE(model_name, version)
);

-- Audit logs (hypertable - composite PK includes timestamp for partitioning)
CREATE TABLE audit_logs (
    id             UUID        NOT NULL DEFAULT gen_random_uuid(),
    timestamp      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    service        VARCHAR(50) NOT NULL,
    action         VARCHAR(100) NOT NULL,
    entity_type    VARCHAR(50),
    entity_id      VARCHAR(100),
    user_id        UUID,
    details        JSONB       NOT NULL,
    decision_trace JSONB,
    ip_address     INET,
    PRIMARY KEY (id, timestamp)
);
SELECT create_hypertable('audit_logs', 'timestamp');
CREATE INDEX idx_audit_service ON audit_logs (service, timestamp DESC);

-- Users & auth
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    role            VARCHAR(20)  NOT NULL DEFAULT 'viewer',
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE api_keys (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID         NOT NULL REFERENCES users(id),
    key_hash     VARCHAR(255) NOT NULL,
    name         VARCHAR(100) NOT NULL,
    permissions  JSONB        NOT NULL DEFAULT '[]',
    expires_at   TIMESTAMPTZ,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ
);

-- Retention policies
SELECT add_retention_policy('ohlcv', INTERVAL '5 years');
SELECT add_retention_policy('features', INTERVAL '2 years');
SELECT add_retention_policy('predictions', INTERVAL '2 years');
SELECT add_retention_policy('risk_metrics', INTERVAL '3 years');
SELECT add_retention_policy('audit_logs', INTERVAL '7 years');
