from typing import TypedDict, Annotated, Sequence
import os
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END

class PinpingoState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_ideas:str
    price: str
    tone_preference: str
    competitor_data: str
    keyword_list: list[str]     #A2
    selling_point: list[str]     #A2
    seo_metadata: dict     #A2-A3
    final_title: str       #A3
    final_description: str #A3
    is_compliance: bool    #A1
    rubric_config: dict[str,int]    #A4
    audit_result: dict[str,int]     #A4
    system_feedback: str   #A1/A4