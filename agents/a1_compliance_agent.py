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

load_dotenv()

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0
)
embeddings = OllamaEmbeddings(
    model = "nomic-embed-text",
)

#Try to find the folder and split the documents into pages
folder_path = "/Users/neo/Library/Mobile Documents/com~apple~CloudDocs/20xx 我的项目集/Etsy胸针计划/Etsy_Policy"
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"No such file: {folder_path}")
try:
    loader = PyPDFDirectoryLoader(folder_path)
    docs = loader.load()
    print(f"PDF has been loaded and has {len(folder_path)} pages")
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
persist_directory = "/Users/neo/Library/Mobile Documents/com~apple~CloudDocs/20xx 我的项目集/Etsy胸针计划/Etsy_Policy"
collection_name = "Neo_research"

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

#Search the information that we need
retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 5}
)
@tool
def retriever_tool(query:str) -> str:
    """This tool searches and returns the information from the Etsy seller policies"""
    docs = retriever.search(query)
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
        piece = title + label + doc.page.content
        formatted_results.append(piece)
    formatted = "\n\n".join(formatted_results)
    return f"I found some requirements that your idea may not saitisfied with: \n\n{formatted}"

tools = [retriever_tool]
llm = llm.bind(tools)

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




