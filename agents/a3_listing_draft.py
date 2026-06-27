import re
from http.client import responses
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

def construct_draft_prompt(state: pingpingostate) -> str:
    keywords = state["keyword_list"]
    tone = state.get("tone_preference", "professional and friendly")
    selling_point = state.get("selling_point")
    description = state.get("extracted_description")
    rubric = state.get("rubric_config", {})
    prompt = f"""
    You are an expert Etsy listing copywriter. Your goal is to create high-converting, SEO-optimized listing copy.
    
    ### Task Parameters:
    - Tone: {tone}
    - Keywords to include: {keywords}
    - Unique Selling Points: {selling_point}
    - Reference Description: {description}
    - Constraints/Rubric: {rubric}
    
    ### Output Format:
    Please separate your output clearly using these tags:
    <title> [Write your title here] </title>
    <description> [Write your description here] </description>
    """

    current_retry = state.get("retry_count", 0)
    if current_retry > 0:
        prev_title = state.get("draft_title", "")
        feedback = state.get("system_feedback", "")
        prompt += f"""{prev_title} + {feedback}"""
    return prompt

def listing_draft_node(state: pingpingostate) -> dict:
    prompt = construct_draft_prompt(state)
    response = llm.invoke(prompt)

    content = response.content if hasattr(response, "content") else str(response)
    title_match = re.search(r"<title>(.*)</title>", content, re.DOTALL)
    desc_match = re.search(r"<description>(.*)</description>", content, re.DOTALL)

    final_title = title_match.group(1).strip() if title_match else "Draft Title Failed"
    final_description = desc_match.group(1).strip() if desc_match else "Draft Description Failed"
    return {
        "draft_title": final_title,
        "draft_description": final_description,
    }