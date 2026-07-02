import os
from typing import TypedDict, Annotated, Sequence, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from PingPinGoState import PingPinGoState

load_dotenv()

llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0
)

class CompetitorSignal(BaseModel):
    is_valid: bool = Field(description="False if the input is noise or lacks product attributes.")
    sku: str= Field(description="SKU of the product.")
    title: str = Field(description="Title of the competitor.")
    keywords: List[str] = Field(description = "Clean, deduplicated keywords.")
    selling_point: List[str] = Field(description = "Unique selling points like material or size.")
    description: str = Field(description = "The product description extracted or generated. If missing from input, return 'MISSING_DESCRIPTION'.")
    reasoning: str = Field(description = "Brief note on filtering decisions.")

def seo_extraction_node(state: PingPinGoState) -> dict:
    structured_llm = llm.with_structured_output(CompetitorSignal)
    own_data = state.get("user_ideas", "No own data provided")
    competitor_data = state.get("competitor_data", "No competitor_data provided")
    combined_input = f"""
    Analyze the following product information and extract structured SEO signals.
    [User's Own Product Information (High Priority - Please prioritize this data)]: 
    {own_data}
    [Competitor Listing Data (For reference)]: 
    {competitor_data}        
    Instructions:
    1. Extract keywords and selling points based on the provided information.
    2. If a section is empty or contains no relevant data, ignore it.
    3. Maintain high precision and avoid marketing fluff.
    """
    if own_data == "No own data provided" and competitor_data == "No competitor_data provided":
        return{"seo_metadata": {"is_valid": False, "reasoning": "No input provided"}}
    result = structured_llm.invoke(combined_input)
    return {
        "keyword_list": result.keywords,
        "selling_point": result.selling_point,
        "extracted_description": result.description,
        "seo_metadata":{"reasoning": result.reasoning, "is_valid": result.is_valid},
    }