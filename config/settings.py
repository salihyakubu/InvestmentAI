import logging
from decimal import Decimal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_DEFAULT_JWT_SECRET = "change-this-to-a-random-secret-key"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")

    # Database
    database_url: str = "postgresql+asyncpg://investai:investai@localhost:5432/investai"
    redis_url: str = "redis://localhost:6379/0"

    @property
    def async_database_url(self) -> str:
        """Convert DATABASE_URL to asyncpg format (Railway uses postgresql://)."""
        url = self.database_url
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        return url

    # Alpaca
    alpaca_api_key: SecretStr = SecretStr("")
    alpaca_secret_key: SecretStr = SecretStr("")
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # Binance
    binance_api_key: SecretStr = SecretStr("")
    binance_secret_key: SecretStr = SecretStr("")

    # Trading
    trading_mode: str = "paper"  # backtest, paper, live
    initial_capital: Decimal = Decimal("100.00")
    active_symbols_stocks: list[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    active_symbols_crypto: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]

    # Risk rules (tuned for $100 micro-account)
    max_position_pct: float = 0.10
    max_sector_pct: float = 0.30
    max_asset_class_pct: float = 0.70
    max_single_order_pct: float = 0.05
    max_daily_drawdown_pct: float = 0.05
    max_total_drawdown_pct: float = 0.15
    max_pairwise_correlation: float = 0.85
    max_portfolio_positions: int = 10
    max_portfolio_var_95: float = 0.03
    circuit_breaker_loss_pct: float = 0.07
    circuit_breaker_cooldown_minutes: int = 30

    # ML
    model_artifact_path: str = "model_artifacts"
    retrain_interval_hours: int = 168
    # How many days of 1m bars the AutoRetrainer loads for a retraining run.
    # 30 days assumes scripts/backfill_history.py has seeded deep history;
    # the rolling feature replay over ~216k crypto bars runs in a worker
    # thread (minutes), so the event loop is unaffected.
    retrain_lookback_days: int = 30
    # Serving-side abstain: a non-flat prediction whose max-class probability
    # is below this floor is emitted as flat. With calibrated 3-class outputs
    # (base rates ~31/38/31) absolute probability mostly reflects the class
    # prior, so this is a loose sanity floor, not the trade gate -- production
    # overrides it via MIN_PREDICTION_CONFIDENCE (0.40 as of 2026-07-21).
    min_prediction_confidence: float = 0.6
    # Trade trigger: a long prediction qualifies for rebalancing only when
    # p(long) - p(short) >= this margin. Unlike an absolute confidence bar,
    # the margin measures directional edge independent of class priors.
    min_edge_margin: float = 0.10
    ensemble_min_agreement: int = 3
    # Exploration paper-trading: the learning period must also exercise the
    # trade path (entries, fills, exits, real P&L attribution), so when no
    # signal clears the full trade gate, the single BEST fresh signal above the
    # lower exploration margin may open a small, auto-expiring position.
    # Exploration trades are tagged ("explore-" order ids) and their P&L is
    # LEARNING data, never edge evidence. Enforced paper-mode-only in code.
    exploration_enabled: bool = True
    exploration_edge_margin: float = 0.03
    exploration_weight: float = 0.03
    exploration_hold_minutes: int = 15

    # API
    jwt_secret: SecretStr = SecretStr(_DEFAULT_JWT_SECRET)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440
    api_rate_limit: int = 100
    # Slack incoming-webhook for operational alerts (circuit breaker, health).
    # Empty = alerts go to logs only. Set ALERT_WEBHOOK_URL in the deployment.
    alert_webhook_url: str = ""
    # CORS: explicit allowlist instead of a wildcard. Override via
    # CORS_ALLOW_ORIGINS (JSON list) per environment / deployed frontend origin.
    cors_allow_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
    ]

    # Monitoring
    prometheus_enabled: bool = True
    # Worker log verbosity. DEBUG emits per-event lines (~12/min steady state,
    # bursts far higher) which trips Railway's 500 logs/sec limit and DROPS
    # messages -- exactly the lines needed during an incident. INFO in prod.
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _validate_security(self) -> "Settings":
        """Refuse the default JWT secret in live trading; warn otherwise.

        With HS256 the default secret is published in source, so any token can
        be forged. Fail closed before real money is at risk; surface a loud
        warning in paper/backtest so it is fixed before going live.
        """
        secret = self.jwt_secret.get_secret_value()
        is_default = secret == _DEFAULT_JWT_SECRET
        if self.trading_mode == "live":
            if is_default or len(secret) < 32:
                raise ValueError(
                    "JWT_SECRET must be a strong (>=32 char) non-default value "
                    "when TRADING_MODE=live. Refusing to start."
                )
        elif is_default:
            logger.warning(
                "JWT_SECRET is the insecure published default; tokens are "
                "forgeable. Set a strong JWT_SECRET before exposing the API."
            )
        return self


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
