"""Shared Pydantic schemas used across the API."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic wrapper for paginated list responses."""

    items: list[T]
    total: int = Field(description="Total number of items matching the query")
    page: int = Field(ge=1, description="Current page number")
    page_size: int = Field(ge=1, description="Number of items per page")
    pages: int = Field(ge=0, description="Total number of pages")


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str
    error_code: str | None = None


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = Field(description="Overall system status: ok | degraded | down")
    services: dict[str, str] = Field(
        default_factory=dict,
        description="Per-service health status map",
    )


class SuccessResponse(BaseModel):
    """Generic success acknowledgement."""

    message: str
