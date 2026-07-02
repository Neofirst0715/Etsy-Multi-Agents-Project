from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage
from PingPinGoState import PingPinGoState
from a1_compliance_agent import compliance_node
from a2_seo_extraction_agent import seo_extraction_node
from a3_listing_draft import listing_draft_node
from a4_audit_node import audit_node
from a5_final_delivery_node import final_delivery_node

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

initial_state = {
    "messages": [HumanMessage(content="Help me draft a pin for a cat lover.")],
    "sku": "ENAMEL-CAT-001",
    "user_ideas": "A cute enamel pin of a sleeping cat, designed for daily wear.",
    "competitor_data": "High quality cat pin, fast shipping, perfect gift.",
    "retry_count": 0
}
workflow = StateGraph(PingPinGoState)
workflow.add_node("compliance_check", compliance_node)
workflow.add_node("seo_extraction", seo_extraction_node)
workflow.add_node("listing_draft", listing_draft_node)
workflow.add_node("audit", audit_node)
workflow.add_node("delivery", final_delivery_node)

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
workflow.add_edge("listing_draft", "audit")
workflow.add_conditional_edges(
    "audit",
    should_continue_after_audit,
    {
        "complete": "delivery",
        "revise": "listing_draft",
        "human_intervention": END
    }
)
workflow.add_edge("delivery", END)

app = workflow.compile()

if __name__ == "__main__":
    final_state = app.invoke(initial_state)
    print("\n=== Final State ===")
    for key, value in final_state.items():
        if key == "messages":
            continue
        print(f"{key}: {value}")
