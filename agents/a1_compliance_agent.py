from typing import TypedDict, Annotated, Sequence
import os
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import dashscope
dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_chroma import Chroma
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from PingPinGoState import PingPinGoState

load_dotenv()

llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0
)
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
)

#Try to find the folder and split the documents into pages
folder_path = "/Users/neo/Library/Mobile Documents/com~apple~CloudDocs/20xx我的项目集/Etsy_Enamel_Pin/agents/Etsy_Policy"
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"No such file: {folder_path}")
try:
    loader = PyPDFDirectoryLoader(folder_path)
    docs = loader.load()
    print(f"PDF has been loaded and has {len(docs)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise
#Highlight the category
for doc in docs:
    source_path = doc.metadata.get("source", "")
    if "Seller Policy" in source_path:
        doc.metadata["category"] = "Seller_Standards"
    elif "Production partners" in source_path:
        doc.metadata["category"] = "Production_Partners"
    elif "Shop Policies" in source_path:
        doc.metadata["category"] = "Shop_Policies"
    else:
        doc.metadata["category"] = "General_Help"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
page_split = text_splitter.split_documents(docs)

#Save the split documents
persist_directory = "/Users/neo/Library/Mobile Documents/com~apple~CloudDocs/20xx我的项目集/Etsy_Enamel_Pin/agents/Etsy_Policy/persist_directory"
collection_name = "Neo_research_qwen"  # separate from the old local-embedding collection (different vector dimensions)

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
try:
    vectorstore = Chroma.from_documents(
        documents = page_split,
        embedding = embeddings,
        persist_directory = persist_directory,
        collection_name = collection_name,
    )
    print(f"Created ChromaDB vector store!")
except Exception as e:
    print(f"Error creating chromaDB vector store: {e}")
    raise

from pydantic import BaseModel, Field

class ComplianceVerdict(BaseModel):
    is_compliant: bool = Field(description="True only if the selling concept clearly meets Etsy policy")
    reason: str = Field(description="Which policy clause supports this verdict, or what's missing")

#Search the information that we need
retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 5}
)
@tool
def retriever_tool(query:str) -> str:
    """This tool searches and returns the information from the Etsy seller policies"""
    docs = retriever.invoke(query)
    if not docs:
        return ("No clauses directly matching your idea were found in the current policy database."
                "This just means the search didn't hit anything; it doesn't mean the creative idea is compliant or not."
                "Please go through the checklist item by item based on the other info retrieved."
                "If you don't have enough basis to judge a specific category, clearly state what part of the policy is missing instead of letting it pass by default.")
    formatted_results = []
    for i, doc in enumerate(docs):
        title = f"{i+1}. "
        label = ""
        if doc.metadata:
            cat = doc.metadata.get('category', 'Not Labelled')
            label = f"{cat}"
        piece = title + label + doc.page_content
        formatted_results.append(piece)
    formatted = "\n\n".join(formatted_results)
    return f"Retrieved the following relevant Etsy policy clauses:\n\n{formatted}"

tools = [retriever_tool]
llm = llm.bind_tools(tools)

def call_model(state: PingPinGoState):
    system_prompt = SystemMessage(content = "You are an Etsy specialist")
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)
a1_builder = StateGraph(PingPinGoState)
a1_builder.add_node("call_model", call_model)
a1_builder.add_node("tools", tool_node)
a1_builder.set_entry_point("call_model")
a1_builder.add_conditional_edges("call_model", tools_condition)
a1_builder.add_edge("tools", "call_model")
a1_compliance_app = a1_builder.compile()

def compliance_node(state: PingPinGoState):
    result = a1_compliance_app.invoke(state)
    structured_llm = llm.with_structured_output(ComplianceVerdict)
    verdict = structured_llm.invoke(result["messages"])
    return{
        "is_compliance": verdict.is_compliant,
        "system_feedback": "Passed" if verdict.is_compliant else f"Not compliant: {verdict.reason}"
    }





