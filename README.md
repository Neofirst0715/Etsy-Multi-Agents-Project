# Pingpin — Etsy Multi-Agent Listing System

A multi-agent pipeline for generating Etsy listing copy, built as **Condition C** of a controlled experiment testing whether AI assistance improves listing performance for non-native-English Etsy sellers.

> **Status: in progress.** Agent 1 (compliance pre-screening) and Agent 2 (SEO signal extraction) are implemented below. Agents 3–5 are designed (see diagram) and will be added incrementally.

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

Five-agent pipeline, diagrammed before any code was written.

> **Note:** the mermaid diagram below predates the current architecture and is pending an update. It still shows an early version of Agent 1 and an Agent 2 focused on scraping market trends. The current design (RAG-based compliance pre-screening for A1; manual-paste quality-gate + structured extraction for A2) is reflected in the agent table and per-agent sections beneath it.

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
| **Agent 2** | Quality gate + structured SEO signal extraction from manually-pasted competitor listings (sanitize noise → extract deduplicated keywords / selling points via a Pydantic contract). Manual paste, not scraped — Etsy's ToS prohibits automated collection for AI use. | ✅ Implemented |
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

## Agent 2 — Quality Gate + SEO Signal Extraction

Users paste competitor listing text that is often noisy — marketing fluff, social links, malformed or poorly structured fragments. Passing that raw text to Agent 3 would degrade the generated copy ("garbage in, garbage out"). Agent 2's job is to **sanitize first, then extract**, so only clean, structured signals reach the generation stage.

**Design choice — structured output over manual `if`-gates:** rather than hard-coding rules to strip specific words (unmaintainable, and prone to over-stripping high-value modifiers like "Personalized"), the agent uses an LLM constrained by a Pydantic contract (`CompetitorSignal`) to do semantic categorization. The contract forces clean, typed output instead of free text:

- `is_valid: bool` — quality gate; noise / no product attributes → `False`
- `keywords: List[str]` — deduplicated
- `selling_points: List[str]` — material, size, use-case
- `reasoning: str` — audit trail for why the input passed or failed

**Design choice — no vector store for Agent 2:** competitor input is a single short pasted listing, not a corpus to search, so it goes straight into the LLM's context window. ChromaDB is reserved for Agent 1's actual retrieval use case (hundreds of policy pages).

**Data priority rule:** when both own-shop historical conversion data and competitor data exist, own data is treated as ground truth and takes priority; competitor data is treated as a (speculative) market signal. Agent 3 will consume `keyword_list` / `selling_point` / `seo_metadata` from state with this weighting in mind.

**Model:** `qwen2.5:7b` (Ollama, local)

## Tech stack

- LangGraph / LangChain — agent orchestration
- Ollama (`qwen2.5:7b`, `nomic-embed-text`) — local inference + embeddings
- Pydantic — structured-output contracts for agent I/O
- ChromaDB — vector store (Agent 1 only)
- Python

## Setup

```bash
pip install langgraph langchain langchain-ollama langchain-community langchain-chroma langchain-text-splitters python-dotenv pydantic
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Agent 1 currently expects Etsy policy PDFs in a local folder referenced by `folder_path` in `agents/a1_compliance_agent.py`. **This path is hardcoded to a personal machine right now — update it to a relative `./data/etsy_policy/` path (or an env var) before running elsewhere.**

## Known issues / next steps

**Agent 1**
- [ ] `folder_path` / `persist_directory` are hardcoded placeholder paths — replace with a relative path (`./data/etsy_policy/`) or env var before running elsewhere
- [ ] Implement the `compliance_node` graph node (LLM call + tool loop + state write of `is_compliance` / `system_feedback`) — the retriever and `retriever_tool` exist, but nothing calls them yet. Reuse the `call_llm` / `take_action` / `should_continue` pattern already working in `RAG_Agent.py`.
- [ ] Wrap A1's module-level setup (PDF load, splitting, Chroma build) in a function or `if __name__ == "__main__":` guard so importing it elsewhere doesn't re-run the whole ingestion

**State / shared modules**
- [ ] `PinpingoState` must live in exactly one place (the shared state module) and be imported everywhere — no duplicate definitions across agent files
- [ ] Keep the state class name spelled consistently as `PinpingoState` across all files and imports
- [ ] Decide whether to add `retry_count` to `PinpingoState` now, ahead of building Agent 4's retry loop

**Graph wiring (`main.py`)**
- [ ] Add `workflow.compile()` and an actual invocation entrypoint
- [ ] A1 → A2 must be a **conditional edge** on `is_compliance` (`False` → `END`, `True` → `seo_extraction`), not a flat unconditional edge
- [ ] Add a conditional edge after A2 on `is_valid` (valid → Agent 3; invalid → human intervention / re-prompt)

**Docs**
- [ ] Update the mermaid diagram to match the current A1 (RAG compliance) / A2 (quality-gate extraction) design

## Project context

Part of a larger experiment comparing human-written, pipeline-generated, and agent-generated Etsy listing copy. Full experimental design, pricing/COGS analysis, and compliance research are documented separately as part of the broader portfolio project.