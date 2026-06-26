from langgraph.graph import StateGraph, END

from agents import pingpingostate
from agents.a2_seo_extraction_agent import seo_extraction_node
from pingpingostate import pingpingostate
from a1_compliance_agent import complaince_node
from a2_seo_extraction_agent import seo_extraction_node

def should_proceed_after_complaience(state) -> str:
    if state.get("is_complete", False):
        return "proceed"
    return "end"

workflow = StateGraph(pingpingostate)
workflow.add_node("compliance_check", complaince_node)
workflow.add_edge("seo_extraction", seo_extraction_node)
workflow.set_entry_point("compliance_check")
workflow.add_contional_edges(
    "compliance_check",
    should_proceed_after_complaience,
    {
        "proceed": "seo_extraction",
        "end": END
    }
)