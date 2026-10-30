"""LangGraph orchestration of read-only diagnostics and evidence checks."""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from datasage.anomaly import anomaly_findings
from datasage.distribution import distribution_findings
from datasage.ingestion import load_dataset
from datasage.missingness import missingness_findings
from datasage.models import Finding
from datasage.profiling import profile_dataset, profile_findings
from datasage.validation import validate_findings


class InvestigationState(TypedDict, total=False):
    dataset_path: str
    dataset_metadata: dict[str, Any]
    target_column: str | None
    prediction_context: str | None
    investigation_plan: list[str]
    completed_agents: list[str]
    profiling_result: dict[str, Any]
    findings: list[dict[str, Any]]
    validation_results: list[dict[str, Any]]
    agent_errors: list[str]
    final_report: dict[str, Any]
    status: str


def ingest(state: InvestigationState) -> InvestigationState:
    _, metadata = load_dataset(state["dataset_path"])
    return {"dataset_metadata": metadata.model_dump(by_alias=True), "status": "ingested"}


def plan(state: InvestigationState) -> InvestigationState:
    tasks = ["profiling", "missingness", "anomaly", "distribution", "validation"]
    if state.get("target_column") is not None:
        tasks.append("ml_risk")  # planned, not yet implemented
    return {"investigation_plan": tasks, "status": "planned"}


def profile(state: InvestigationState) -> InvestigationState:
    frame, _ = load_dataset(state["dataset_path"])
    summary = profile_dataset(frame)
    findings = [item.model_dump(mode="json") for item in profile_findings(summary)]
    return {
        "profiling_result": summary, "findings": findings,
        "completed_agents": ["profiling"], "status": "profiling_complete_unvalidated",
    }


def missingness(state: InvestigationState) -> InvestigationState:
    frame, _ = load_dataset(state["dataset_path"])
    new_findings = [f.model_dump(mode="json") for f in missingness_findings(frame)]
    return {
        "findings": state.get("findings", []) + new_findings,
        "completed_agents": state.get("completed_agents", []) + ["missingness"],
        "status": "missingness_complete_unvalidated",
    }


def anomaly(state: InvestigationState) -> InvestigationState:
    frame, _ = load_dataset(state["dataset_path"])
    new_findings = [f.model_dump(mode="json") for f in anomaly_findings(frame)]
    return {
        "findings": state.get("findings", []) + new_findings,
        "completed_agents": state.get("completed_agents", []) + ["anomaly"],
        "status": "anomaly_complete_unvalidated",
    }


def distribution(state: InvestigationState) -> InvestigationState:
    frame, _ = load_dataset(state["dataset_path"])
    findings = [f.model_dump(mode="json") for f in distribution_findings(frame, state.get("target_column"))]
    return {
        "findings": state.get("findings", []) + findings,
        "completed_agents": state.get("completed_agents", []) + ["distribution"],
        "status": "distribution_complete_unvalidated",
    }


def validate(state: InvestigationState) -> InvestigationState:
    frame, _ = load_dataset(state["dataset_path"])
    findings = [Finding.model_validate(f) for f in state.get("findings", [])]
    return {
        "validation_results": validate_findings(frame, findings),
        "completed_agents": state.get("completed_agents", []) + ["validation"],
        "status": "validation_complete",
    }


def build_workflow():
    graph = StateGraph(InvestigationState)
    graph.add_node("ingest", ingest)
    graph.add_node("supervisor", plan)
    graph.add_node("profiling", profile)
    graph.add_node("missingness", missingness)
    graph.add_node("anomaly", anomaly)
    graph.add_node("distribution", distribution)
    graph.add_node("validation", validate)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "supervisor")
    graph.add_edge("supervisor", "profiling")
    graph.add_edge("profiling", "missingness")
    graph.add_edge("missingness", "anomaly")
    graph.add_edge("anomaly", "distribution")
    graph.add_edge("distribution", "validation")
    graph.add_edge("validation", END)
    return graph.compile()
