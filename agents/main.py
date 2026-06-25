from langgraph.graph import StateGraph
from agents.a1_compliance_agent import compliance_agent
from agents.a2_seo import seo_agent
from state import state

workflow = StateGraph(state)

workflow.add_node("compliance_checker", compliance_agent)
workflow.add_node("seo_optimizer", seo_agent)

workflow.set_entry_point("compliance_checker")
workflow.add_edge("compliance_checker", "seo_optimizer")
workflow.add_edge("seo_optimizer", END)

app = workflow.compile()