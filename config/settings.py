from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from decimal import Decimal


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
    min_prediction_confidence: float = 0.6
    ensemble_min_agreement: int = 3

    # API
    jwt_secret: SecretStr = SecretStr("change-this-to-a-random-secret-key")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440
    api_rate_limit: int = 100

    # Monitoring
    prometheus_enabled: bool = True


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
