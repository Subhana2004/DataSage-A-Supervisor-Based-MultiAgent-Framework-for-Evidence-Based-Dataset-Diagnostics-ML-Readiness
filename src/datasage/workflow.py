"""LangGraph orchestration of read-only diagnostics and evidence checks."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from datasage.anomaly import anomaly_findings
from datasage.distribution import distribution_findings
from datasage.ingestion import load_dataset
from datasage.missingness import missingness_findings
from datasage.ml_risk import ml_risk_findings
from datasage.models import Finding
from datasage.profiling import profile_dataset, profile_findings
from datasage.reporting import reconcile
from datasage.validation import validate_findings


class InvestigationState(TypedDict, total=False):
    dataset_path: str
    dataset_metadata: dict[str, Any]
    target_column: str | None
    prediction_context: str | None
    investigation_plan: list[str]
    completed_agents: Annotated[list[str], operator.add]
    profiling_result: dict[str, Any]
    findings: Annotated[list[dict[str, Any]], operator.add]
    validation_results: list[dict[str, Any]]
    agent_errors: list[str]
    final_report: dict[str, Any]
    status: str


def ingest(state: InvestigationState) -> InvestigationState:
    _, metadata = load_dataset(state["dataset_path"])
    return {"dataset_metadata": metadata.model_dump(by_alias=True), "status": "ingested"}


def plan(state: InvestigationState) -> InvestigationState:
    tasks = ["profiling", "missingness", "anomaly", "distribution"]
    if state.get("target_column") is not None:
        tasks.append("ml_risk")
    tasks.extend(["validation", "reporting"])
    return {"investigation_plan": tasks, "status": "planned"}


def profile(state: InvestigationState) -> InvestigationState:
    frame, _ = load_dataset(state["dataset_path"])
    summary = profile_dataset(frame)
    findings = [item.model_dump(mode="json") for item in profile_findings(summary)]
    return {
        "profiling_result": summary, "findings": findings,
        "completed_agents": ["profiling"],
    }


def missingness(state: InvestigationState) -> InvestigationState:
    frame, _ = load_dataset(state["dataset_path"])
    new_findings = [f.model_dump(mode="json") for f in missingness_findings(frame)]
    return {
        "findings": new_findings,
        "completed_agents": ["missingness"],
    }


def anomaly(state: InvestigationState) -> InvestigationState:
    frame, _ = load_dataset(state["dataset_path"])
    new_findings = [f.model_dump(mode="json") for f in anomaly_findings(frame)]
    return {
        "findings": new_findings,
        "completed_agents": ["anomaly"],
    }


def distribution(state: InvestigationState) -> InvestigationState:
    frame, _ = load_dataset(state["dataset_path"])
    findings = [f.model_dump(mode="json") for f in distribution_findings(frame, state.get("target_column"))]
    return {
        "findings": findings,
        "completed_agents": ["distribution"],
    }


def ml_risk(state: InvestigationState) -> InvestigationState:
    frame, _ = load_dataset(state["dataset_path"])
    findings = [f.model_dump(mode="json") for f in ml_risk_findings(
        frame, state["target_column"], state.get("prediction_context"))]
    return {
        "findings": findings,
        "completed_agents": ["ml_risk"],
    }


def core_diagnostics_complete(state: InvestigationState) -> InvestigationState:
    """Barrier: all independent diagnostic branches must finish before validation."""
    return {"status": "core_diagnostics_complete"}


def route_risk(state: InvestigationState) -> str:
    return "ml_risk" if state.get("target_column") is not None else "validation"


def validate(state: InvestigationState) -> InvestigationState:
    frame, _ = load_dataset(state["dataset_path"])
    findings = [Finding.model_validate(f) for f in state.get("findings", [])]
    return {
        "validation_results": validate_findings(frame, findings),
        "completed_agents": ["validation"],
        "status": "validation_complete",
    }


def report(state: InvestigationState) -> InvestigationState:
    findings = [Finding.model_validate(f) for f in state.get("findings", [])]
    output = reconcile(findings, state.get("validation_results", []),
                       state.get("dataset_metadata"))
    return {
        "final_report": output,
        "completed_agents": ["reporting"],
        "status": "report_complete",
    }


def build_workflow():
    graph = StateGraph(InvestigationState)
    graph.add_node("ingest", ingest)
    graph.add_node("supervisor", plan)
    graph.add_node("profiling", profile)
    graph.add_node("missingness", missingness)
    graph.add_node("anomaly", anomaly)
    graph.add_node("distribution", distribution)
    graph.add_node("ml_risk", ml_risk)
    graph.add_node("core_barrier", core_diagnostics_complete)
    graph.add_node("validation", validate)
    graph.add_node("reporting", report)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "supervisor")
    # All four diagnostic branches are independent and can run concurrently.
    for node in ("profiling", "missingness", "anomaly", "distribution"):
        graph.add_edge("supervisor", node)
    graph.add_edge(["profiling", "missingness", "anomaly", "distribution"], "core_barrier")
    graph.add_conditional_edges("core_barrier", route_risk)
    graph.add_edge("ml_risk", "validation")
    graph.add_edge("validation", "reporting")
    graph.add_edge("reporting", END)
    return graph.compile()
