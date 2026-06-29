from langgraph.graph import StateGraph, END
from pingpingostate import pingpingostate
from a1_compliance_agent import compliance_node
from a2_seo_extraction_agent import seo_extraction_node
from a3_listing_draft import listing_draft_node
from a4_audit import audit_node

def should_proceed_after_compliance(state) -> str:
    if state.get("is_compliance", False):
        return "proceed"
    return "end"
def should_continue_after_audit(state) -> str:
    if state.get("system_feedback") == "Passed":
        return "complete"
    if state.get("retry_count", 0) < 2:
        return "revise"
    return "human_intervention"

workflow = StateGraph(pingpingostate)
workflow.add_node("compliance_check", compliance_node)
workflow.add_node("seo_extraction", seo_extraction_node)
workflow.add_node("listing_draft", listing_draft_node)
workflow.add_node("audit_node", audit_node)

workflow.set_entry_point("compliance_check")
workflow.add_conditional_edges(
    "compliance_check",
    should_proceed_after_compliance,
    {
        "proceed": "seo_extraction",
        "end": END
    }
)
workflow.add_edge("seo_extraction", "listing_draft")
workflow.add_edge("listing_draft", "audit_node")
workflow.add_conditional_edges(
    "audit_node",
    should_continue_after_audit,
    {
        "complete": END,
        "revise": "listing_draft",
        "human_intervention": END
    }
)

app = workflow.compile()
