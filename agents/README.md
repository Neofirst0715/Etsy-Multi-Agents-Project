# Pingpin — Etsy Multi-Agent Listing System

A multi-agent pipeline for generating Etsy listing copy, built as **Condition C** of a controlled experiment testing whether AI assistance improves listing performance for non-native-English Etsy sellers.

> **Status: in progress.** Agent 1 (compliance pre-screening) is implemented below. Agents 2–5 are designed (see diagram) and will be added incrementally.

## Background

Most Etsy copywriting tools assume the seller already understands the platform's rules around what counts as "handmade," "designed by," or "curated" — and assume English fluency. Non-native-English sellers sourcing finished goods (e.g. from wholesale markets) often don't know these distinctions exist until a listing gets flagged.

This project treats that as the first problem to solve, not an edge case: before any copy gets generated, the system checks whether the *selling concept itself* is compliant with Etsy's seller policies — based on Etsy's actual published Creativity Standards, not assumptions.

## Experiment context

This pipeline is one of three conditions in a live A/B/C test run on a real Etsy shop:

| Condition | Approach |
|---|---|
| A | Fully human-written copy (baseline) |
| B | LangChain + Ollama batch generation, fixed prompt template |
| C | **This repo** — stateful LangGraph multi-agent system with compliance pre-screening, market-signal synthesis, and a hybrid (rule-based + LLM) critic loop |

Primary metric: favorites rate, compared across conditions with significance testing (scipy.stats).

## Architecture

Five-agent pipeline, diagrammed before any code was written:
flowchart TD
    %% Start Node
    Start([Start: New SKU for Copy Gen]) --> Decision1{Existing own-shop<br/>conversion data?}

    %% Information Gathering Layer
    Decision1 -- No / Cold start --> InfoGathering
    Decision1 -- Yes --> InfoGathering

    subgraph InfoGathering [Information Gathering]
        direction TB
        A1[Agent 1<br/>Etsy compliance pre-screening<br/>(RAG: retrieve policy clauses<br/>+ structured checklist judgment)]
        A2[Agent 2<br/>Popular comparable listings<br/>keyword / selling-point<br/>extraction<br/>(categories only, no full text)]
    end

    OwnData[(Own SKU data<br/>if shop exists<br/>material / size / real<br/>attributes)]

    %% Connections to Agent 3
    A1 --> A3
    A2 --> A3
    OwnData -. weighted in if present .-> A3

    A3[Agent 3<br/>Generate listing copy<br/>(fuse market signals + own<br/>data,<br/>keep tone consistent)]

    %% Agent 4 Audit
    A3 --> A4{Agent 4<br/>Audit}
    A4 --> Hard[Hard checklist<br/>(rule-based, not LLM)<br/>· word count 100-150<br/>· keyword coverage met<br/>· >=1 use-case scenario<br/>· no prohibited claims<br/>· no fabricated attributes]
    
    Hard --> Decision2{All hard rules pass?}

    %% Audit Logic Branches
    Decision2 -- Yes --> Soft[Soft review<br/>(LLM scoring, flags only, no<br/>reject)<br/>· tone naturalness<br/>· differentiation vs<br/>competitors]
    
    Decision2 -- No --> Decision3{Retry count<br/>< 2?}

    Decision3 -- Yes --> A3
    Decision3 -- No, cap reached --> Human[Human annotation /<br/>intervention<br/>(human-in-the-loop)]

    %% Convergence to Agent 5
    Soft --> A5
    Human --> A5

    A5[Agent 5<br/>Format output<br/>(final copy + soft flags<br/>+ experiment log w/ prompt<br/>version)]

    A5 --> End([Deliver / archive<br/>write to tracking table])

    %% Styling
    classDef agent fill:#f9f9f9,stroke:#333,stroke-width:1px;
    class A1,A2,A3,A4,A5 agent;

```mermaid
flowchart TD
    Start([Start: new SKU needs copy]) --> HasData{Existing own-shop<br/>conversion data?}

    HasData -- No / cold start --> A1
    HasData -- Yes --> A1

    subgraph Information Gathering
    A1["Agent 1<br/>Etsy compliance pre-screening<br/>(RAG: retrieve policy clauses<br/>+ structured checklist judgment)"]
    A2["Agent 2<br/>Popular comparable listings<br/>keyword / selling-point extraction<br/>(categories only, no full text)"]
    end

    A1 --> A3
    A2 --> A3

    OwnData[("Own SKU data<br/>(if shop exists)<br/>material / size / real attributes")]
    OwnData -. weighted in if present .-> A3

    A3["Agent 3<br/>Generate listing copy<br/>(fuse market signals + own data,<br/>keep tone consistent)"]

    A3 --> A4{"Agent 4<br/>Audit"}

    A4 -- "Hard checklist<br/>(rule-based, not LLM)<br/>· word count 100–150<br/>· keyword coverage met<br/>· >=1 use-case scenario<br/>· no prohibited claims<br/>· no fabricated attributes" --> Hard{All hard rules pass?}

    Hard -- No --> Retry{Retry count<br/>< 2?}
    Retry -- Yes --> A3
    Retry -- No, cap reached --> Human["Human annotation / intervention<br/>(human-in-the-loop)"]

    Hard -- Yes --> Soft["Soft review<br/>(LLM scoring, flags only, no reject)<br/>· tone naturalness<br/>· differentiation vs competitors"]

    Soft --> A5
    Human --> A5

    A5["Agent 5<br/>Format output<br/>(final copy + soft flags<br/>+ experiment log w/ prompt version)"]

    A5 --> End([Deliver / archive<br/>write to tracking table])
```

| Agent | Role | Status |
|---|---|---|
| **Agent 1** | Etsy compliance pre-screening (RAG over Etsy seller policy docs + structured checklist judgment) | ✅ Implemented |
| Agent 2 | Extracts keyword/selling-point categories from competitor listings (manually pasted, not scraped — Etsy's ToS prohibits automated collection for AI use) | 🔲 Planned |
| Agent 3 | Synthesizes market signals + own sales data into title/description | 🔲 Planned |
| Agent 4 | Hybrid critic — hard rules in Python (word count, keyword coverage, use-case presence) gate before LLM soft-scoring on tone/naturalness | 🔲 Planned |
| Agent 5 | Formats and archives final output + experiment log | 🔲 Planned |

## Agent 1 — Compliance Pre-Screening

Etsy requires every listing to be filed under one of four categories (Made by a Seller / Designed by a Seller / Sourced by a Seller / Curated set of purchased goods), each with different requirements. Agent 1 checks a seller's product concept against this before any copy is generated.

**Pipeline:**
1. Loads Etsy seller policy documents (PDF) from a local policy folder
2. Tags each chunk with a category (`Seller_Standards`, `Production_Partners`, `Shop_Policies`, `General_Help`) based on source filename
3. Splits and embeds into a Chroma vector store (`nomic-embed-text` via Ollama)
4. Exposes a `retriever_tool` that the agent calls to ground its compliance judgment in retrieved policy text — rather than guessing from the model's own (often outdated or hallucinated) sense of platform rules

**Design choice — why retrieval is split from judgment:** the retriever's only job is to report what it found (or honestly report that it found nothing). The actual compliance verdict is deliberately *not* decided by free-text LLM judgment alone — local models (this runs on `qwen2.5:7b` via Ollama) tend to hedge or contradict themselves on open-ended questions. The verdict is structured as a category classification step followed by a fixed per-category checklist, so the model's output is auditable rather than a vague paragraph.

**Model:** `qwen2.5:7b` (Ollama, local — chosen for reproducibility and zero API cost during development)

## Tech stack

- LangGraph / LangChain — agent orchestration
- Ollama (`qwen2.5:7b`, `nomic-embed-text`) — local inference + embeddings
- ChromaDB — vector store
- Python

## Setup

```bash
pip install langgraph langchain langchain-ollama langchain-community langchain-chroma langchain-text-splitters python-dotenv
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Agent 1 currently expects Etsy policy PDFs in a local folder referenced by `folder_path` in `agents/a1_compliance_agent.py`. **This path is hardcoded to a personal machine right now — update it to a relative `./data/etsy_policy/` path (or an env var) before running elsewhere.**

## Known issues / next steps

- [ ] `folder_path` / `persist_directory` are hardcoded absolute paths — needs to be relative or env-configured for portability
- [ ] `retriever.search(query)` should be `retriever.invoke(query)` — current method doesn't exist on the retriever interface
- [ ] `doc.page.content` should be `doc.page_content`
- [ ] `main.py` references `agents.a2_seo.seo_agent` and `state.state`, which don't exist yet — placeholder for once Agent 2 and the shared state module are wired up
- [ ] `a1_compliance_agent.py` builds the retriever tool but doesn't yet expose a callable `compliance_agent` graph node (LLM call + tool loop + conditional edge) — that's the next piece
- [ ] `state.py` currently duplicates the `PinpingoState` definition in `a1_compliance_agent.py` — should live in one place only

## Project context

Part of a larger experiment comparing human-written, pipeline-generated, and agent-generated Etsy listing copy. Full experimental design, pricing/COGS analysis, and compliance research are documented separately as part of the broader portfolio project.
