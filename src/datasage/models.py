"""Traceable diagnostic outputs shared across agents."""
from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high", "critical"]
Confidence = Literal["low", "medium", "high"]


class Evidence(BaseModel):
    method: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    sample_size: int | None = Field(default=None, ge=0)


class Finding(BaseModel):
    finding_id: str
    agent: str
    category: str
    column: str | None = None
    title: str
    description: str
    evidence: Evidence
    severity: Severity
    confidence: Confidence
    limitations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class DatasetMetadata(BaseModel):
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    schema_: dict[str, str] = Field(alias="schema")
    fingerprint: str
    model_config = {"populate_by_name": True}
