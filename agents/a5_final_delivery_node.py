import re
import os
import csv
from datetime import datetime
import audit_config
from typing import TypedDict, List
from PingPinGoState import PingPinGoState
from pydantic import BaseModel, Field
from audit_config import A4_PROMPT_VERSION, A2_EXTRACTION_VERSION, A3_DRAFT_VERSION

timestamp = datetime.now().isoformat()

class final_listing_payload(BaseModel):
    final_title: str = Field(description="Cleaned and formatted final Etsy title")
    final_description: str = Field(description="Cleaned and formatted final Etsy description")
    audit_score: int = Field(description="The final quality score from Agent 4")
    is_archived: bool = Field(description="Flag indicating if the data was successfully logged")

def save_to_archive(data: dict):
    directory = "logs"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, "listing_archive.csv")

    try:
        file_exists = os.path.isfile(filename)

        with open(filename, "a", newline= '', encoding= 'utf-8') as f:
            writer = csv.DictWriter(f, data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        print(f"✅ Data archived successfully to {filename}")
    except PermissionError:
        print(f"❌ Error: Permission denied. Please close the file {filename} if it's open in Excel.")
    except Exception as e:
        print(f"❌ Error saving to archive: {e}")

def final_delivery_node(state: PingPinGoState) -> dict:
    """
        Agent 5: Deliver & Archive Node
        负责清洗文案，记录日志，并交付最终结果。
    """
    sku = state.get("sku", "")
    is_compliance = state.get("is_compliance", False)
    keyword_list = state.get("keyword_list", [])
    final_title = state.get("final_title", "Title Generation Failed")
    final_description = state.get("final_description", "Description Generation Failed")
    audit_result = state.get("audit_result", 0)
    system_feedback = state.get("system_feedback", 0)
    reasoning = state.get("reasoning", "")
    retry_count = state.get("retry_count", 0)
    a2_prompt_version = A2_EXTRACTION_VERSION
    a3_prompt_version = A3_DRAFT_VERSION
    a4_prompt_version = A4_PROMPT_VERSION
    current_timestamp = datetime.now().isoformat()

    log_data = {
        "timestamp": current_timestamp,
        "is_compliance": is_compliance,
        "sku": sku,
        "keyword_list": keyword_list,
        "final_title": final_title,
        "final_description": final_description,
        "audit_result": audit_result,
        "system_feedback": system_feedback,
        "reasoning": reasoning,
        "retry_count": retry_count,
        "a2_prompt_version": a2_prompt_version,
        "a3_prompt_version": a3_prompt_version,
        "a4_prompt_version": a4_prompt_version,
    }

    save_to_archive(log_data)
    return{
        "is_complete": True
    }








