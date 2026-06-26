from typing import TypedDict, Annotated, Sequence, List
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from agents.pingpingostate import pingpingostate

load_dotenv()

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0
)

class CompetitorSignal(BaseModel):
    is_valid: bool = Field(description="False if the input is noise or lacks product attributes.")
    keywords: List[str] = Field(description = "Clean, deduplicated keywords.")
    selling_points: List[str] = Field(description = "Unique selling points like material or size.")
    reasoning: str = Field(description = "Brief note on filtering decisions.")

def seo_extraction_node(state: pingpingostate) -> dict:
    structured_llm = llm.with_structured_output(CompetitorSignal)
    raw_input = state.get("competitor_data", "")
    if not raw_input:
        return {"seo_metadata":{"is_valid": False, "reasoning": "No input povided"}}
    result = structured_llm.invoke(f"Analysing the input: {raw_input}")
    return {
        "keyword_list": result.keywords,
        "selling_point": result.selling_points,
        "seo_metadata":{"reasoning": result.reasoning, "is_valid": result.is_valid},
    }