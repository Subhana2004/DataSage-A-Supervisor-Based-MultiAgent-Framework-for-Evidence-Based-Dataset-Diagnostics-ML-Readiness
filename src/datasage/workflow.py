"""Minimal, deterministic supervisor skeleton for Milestone 1."""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from datasage.ingestion import load_dataset


class InvestigationState(TypedDict, total=False):
    dataset_path: str
    dataset_metadata: dict[str, Any]
    target_column: str | None
    prediction_context: str | None
    investigation_plan: list[str]
    completed_agents: list[str]
    findings: list[dict[str, Any]]
    validation_results: list[dict[str, Any]]
    agent_errors: list[str]
    final_report: dict[str, Any]
    status: str


def ingest(state: InvestigationState) -> InvestigationState:
    _, metadata = load_dataset(state["dataset_path"])
    return {"dataset_metadata": metadata.model_dump(by_alias=True), "status": "ingested"}


def plan(state: InvestigationState) -> InvestigationState:
    # Agent implementations arrive in Milestones 2–4.
    tasks = ["profiling"]
    if state.get("target_column") is not None:
        tasks.append("ml_risk")
    return {"investigation_plan": tasks, "status": "planned"}


def build_workflow():
    graph = StateGraph(InvestigationState)
    graph.add_node("ingest", ingest)
    graph.add_node("supervisor", plan)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "supervisor")
    graph.add_edge("supervisor", END)
    return graph.compile()
