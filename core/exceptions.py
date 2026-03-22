"""Custom exception hierarchy for the InvestAI platform."""

from __future__ import annotations

from typing import Any


class InvestAIError(Exception):
    """Base exception for all InvestAI platform errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

class DataIngestionError(InvestAIError):
    """Raised when data ingestion from an external source fails."""


class DataValidationError(InvestAIError):
    """Raised when ingested or computed data fails validation checks."""


# ---------------------------------------------------------------------------
# ML / prediction layer
# ---------------------------------------------------------------------------

class PredictionError(InvestAIError):
    """Raised when a model prediction call fails."""


class ModelNotFoundError(InvestAIError):
    """Raised when a requested model artifact cannot be located."""


class ModelTrainingError(InvestAIError):
    """Raised when model training encounters an unrecoverable error."""


# ---------------------------------------------------------------------------
# Risk layer
# ---------------------------------------------------------------------------

class RiskLimitExceededError(InvestAIError):
    """Raised when a proposed action would breach a risk limit."""


class CircuitBreakerActiveError(InvestAIError):
    """Raised when the circuit breaker is active and blocks trading."""


# ---------------------------------------------------------------------------
# Order / execution layer
# ---------------------------------------------------------------------------

class OrderError(InvestAIError):
    """Generic order-related error."""


class OrderRejectedError(OrderError):
    """Raised when an order is rejected by the broker or risk system."""


class InsufficientFundsError(OrderError):
    """Raised when account balance is too low to place an order."""


# ---------------------------------------------------------------------------
# Broker connectivity
# ---------------------------------------------------------------------------

class BrokerConnectionError(InvestAIError):
    """Raised when connection to a broker fails."""


class BrokerAPIError(InvestAIError):
    """Raised when a broker API call returns an error response."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConfigurationError(InvestAIError):
    """Raised when application configuration is missing or invalid."""
