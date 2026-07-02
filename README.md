# Pingpin — Etsy Multi-Agent Listing System

A multi-agent pipeline for generating Etsy listing copy, built as **Condition C** of a controlled experiment testing whether AI assistance improves listing performance for non-native-English Etsy sellers.

> **Status: end-to-end working.** All five agents are implemented and wired into one compiled LangGraph workflow that runs start to finish — a product concept is compliance-screened, market signals are extracted, copy is drafted, audited (with a retry loop), and archived to CSV. As of 2026-07-02 the pipeline runs on Alibaba DashScope cloud Qwen (`qwen-plus`) rather than local Ollama.

## Background

Most Etsy copywriting tools assume the seller already understands the platform's rules around what counts as "handmade," "designed by," or "curated" — and assume English fluency. Non-native-English sellers sourcing finished goods (e.g. from wholesale markets) often don't know these distinctions exist until a listing gets flagged.

This project treats that as the first problem to solve, not an edge case: before any copy gets generated, the system checks whether the *selling concept itself* is compliant with Etsy's seller policies — based on Etsy's actual published Creativity Standards, not assumptions.

## Experiment context

This pipeline is one of three conditions in a live A/B/C test run on a real Etsy shop:

| Condition | Approach |
|---|---|
| A | Fully human-written copy (baseline) |
| B | Batch generation, fixed prompt template |
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
| **Agent 1** | Etsy compliance pre-screening — RAG sub-graph (`call_model ⇄ tool_node` loop) + structured `ComplianceVerdict` | ✅ Implemented |
| **Agent 2** | Quality gate + structured SEO signal extraction from manually-pasted competitor listings (sanitize noise → extract deduplicated keywords / selling points via a Pydantic contract). Manual paste, not scraped — Etsy's ToS prohibits automated collection for AI use. | ✅ Implemented |
| **Agent 3** | Drafting agent — fuses A2 market signals + own sales data + tone into a title/description, with a feedback-aware retry loop driven by A4's audit results | ✅ Implemented |
| **Agent 4** | Two-layer critic — Python hard-gate (pass/fail) before LLM soft-scoring (0–20) on tone, selling points, naturalness, differentiation | ✅ Implemented |
| **Agent 5** | Deliver & archive — format, log to CSV (with prompt versions), terminate | ✅ Implemented |

## Agent 1 — Compliance Pre-Screening

Etsy requires every listing to be filed under one of four categories (Made by a Seller / Designed by a Seller / Sourced by a Seller / Curated set of purchased goods), each with different requirements. Agent 1 checks a seller's product concept against this before any copy is generated.

**Pipeline:**
1. Loads Etsy seller policy documents (PDF) from a local policy folder
2. Tags each chunk with a category (`Seller_Standards`, `Production_Partners`, `Shop_Policies`, `General_Help`) based on source filename
3. Splits and embeds into a Chroma vector store (`text-embedding-v3` via DashScope)
4. Exposes a `retriever_tool` that the agent calls to ground its compliance judgment in retrieved policy text — rather than guessing from the model's own (often outdated or hallucinated) sense of platform rules

**Design choice — why retrieval is split from judgment:** the retriever's only job is to report what it found (or honestly report that it found nothing). The compliance verdict is not decided by free-text LLM judgment alone; after the retrieval loop, a `with_structured_output(ComplianceVerdict)` step forces a typed `{is_compliant: bool, reason: str}` output, so the verdict is auditable rather than a vague paragraph.

**Implementation:** A1 is built as its own compiled sub-graph — a `call_model ⇄ tool_node` loop wired with `tools_condition` — and invoked from the main graph's `compliance_check` node. This encapsulates the tool loop and keeps it decoupled from the linear A2–A5 flow. Tool invocation is verified (local models tend to *narrate* a tool call while `tool_calls` comes back empty; a print inside `retriever_tool` confirms real calls).

## Agent 2 — Quality Gate + SEO Signal Extraction

Users paste competitor listing text that is often noisy — marketing fluff, social links, malformed or poorly structured fragments. Passing that raw text to Agent 3 would degrade the generated copy ("garbage in, garbage out"). Agent 2's job is to **sanitize first, then extract**, so only clean, structured signals reach the generation stage.

**Design choice — structured output over manual `if`-gates:** rather than hard-coding rules to strip specific words (unmaintainable, and prone to over-stripping high-value modifiers like "Personalized"), the agent uses an LLM constrained by a Pydantic contract (`CompetitorSignal`) to do semantic categorization:

- `is_valid: bool` — quality gate; noise / no product attributes → `False`
- `keywords: List[str]` — deduplicated
- `selling_points: List[str]` — material, size, use-case
- `reasoning: str` — audit trail for why the input passed or failed

**Design choice — no vector store for Agent 2:** competitor input is a single short pasted listing, not a corpus to search, so it goes straight into the LLM's context window. ChromaDB is reserved for Agent 1's retrieval use case (hundreds of policy pages).

**Data priority rule:** when both own-shop historical conversion data and competitor data exist, own data is treated as ground truth and takes priority; competitor data is a (speculative) market signal.

## Agent 3 — Drafting Agent (with feedback loop)

Agent 3 is where every upstream signal converges into an actual `title` + `description`. It reads A2's cleaned keywords/selling points, the user's own sales data (if any), the tone preference, and the Etsy constraints, then synthesizes the copy.

**Design choice — one base prompt + conditional append, not two prompts.** The standard the copy must meet is *identical* on the first pass and on a retry — what changes is the material on hand, not the strictness. So `construct_draft_prompt` maintains one base prompt (rules + tone + signals, written once) and, only when `retry_count >= 1`, appends a feedback block: the previous draft plus A4's `system_feedback`, with an instruction to revise the flagged parts. Single source of truth for the Etsy-rules block.

**Design choice — state is the memory, so A1/A2 never re-run.** When A4 rejects a draft and routes back to A3, A3 re-reads `keyword_list` / `selling_point` / etc. straight from shared state. The retry edge is `A4 → A3`, not `A4 → A1`, so a revision costs one A3 call, not a full pipeline re-execution.

**Design choice — structured output + regex split.** A Pydantic `BaseModel` plus `re` cleanly separates the title and description, and XML-tagged prompt structure guides the model toward parseable output.

**Tone handling:** if the user supplied a `tone_preference`, it's folded into the prompt; if blank, the model self-determines a market-appropriate tone. The branch is decided in Python, not by the model.

## Agent 4 — Two-Layer Audit (hard gate + soft scoring)

Agent 4 decides whether A3's draft ships or gets sent back to revise. It runs in two layers, cheapest and most reliable first.

**Layer 1 — hard gate (pure Python, pass/fail, no LLM).** `check_hard_rules()` checks the objective things: description word count (100–150), a banned-word blacklist match, and presence of at least one use-case signal. Any failure rejects the draft immediately — it never reaches the LLM.

**Layer 2 — soft scoring (LLM, only after the hard gate passes).** An `AuditResult` Pydantic contract carries the subjective dimensions — `tone_match`, `selling_points`, `naturalness`, `differentiation` (each 0–5) and `feedback_points`.

**Design choice — the score is computed in Python, not trusted from the model.** The final score is the **sum of the four sub-scores**, computed in Python. (The model's self-reported total was found to be unreliable — it would report a total that didn't match its own sub-scores, e.g. sub-scores summing to 18 while the model reported 4, falsely rejecting good copy.) Whether that summed score clears the threshold is also a plain Python `if` — deterministic arithmetic and threshold decisions belong in Python, not the LLM.

**Design choice — rules live in `audit_config.py`, not in the function.** `BANNED_WORDS`, `USE_CASE_SIGNALS`, and the word-count bounds are config constants, separated from logic. The blacklist is kept deliberately narrow — context-dependent words like "best"/"authentic" are left for the LLM layer to judge, to avoid false positives on legitimate copy.

**On reject**, `audit_node` increments `retry_count` and writes the reason into `system_feedback` for A3's next pass. The retry-cap → human-intervention decision lives in the graph router, not inside the node.

## Agent 5 — Deliver & Archive (clean terminus)

Agent 5 is a pure terminus: format, archive, log, then END. No content is changed here (the copy was finalized by A4) and there is deliberately **no human-edit loop** — reintroducing open-ended manual edits would inject a non-standardizable variable and break Condition C's reproducibility. (That's noted as a post-experiment productization idea, not built.)

**Design choice — A5 imports nothing but the state type + its own tools (`csv`, `os`, `datetime`).** It reads everything it needs (`is_compliance`, `keyword_list`, `audit_result`, etc.) straight from shared state — no importing other agents, since their results are already there.

**Logging:** one row per run appended to `logs/listing_archive.csv` (header auto-written on first run via `DictWriter`). Fields include the final copy, SKU id, compliance flag, keywords/selling points, audit scores, `retry_count`, timestamp, and **prompt versions** (`a2_prompt_version` / `a3_prompt_version` / `a4_prompt_version`) — the reproducibility anchor, so any archived listing can be traced to the exact prompt versions that produced and scored it. CSV is chosen so the data loads straight into pandas/scipy for the August experiment analysis.

## Graph wiring (`main.py`)

All five agents are compiled into one LangGraph workflow that runs end-to-end:

- **Entry:** `compliance_check` (A1 sub-graph)
- `compliance_check` → conditional edge on `is_compliance` (`True` → `seo_extraction`; `False` → `END`)
- `seo_extraction` → `listing_draft`
- `listing_draft` → `audit_node`
- `audit_node` → conditional edge (`Passed` → `archive_node`; `retry_count < 2` → back to `listing_draft`; else → human intervention)
- `archive_node` → `END`

## Tech stack

- LangGraph / LangChain — agent orchestration
- **DashScope cloud Qwen** (`qwen-plus` via the OpenAI-compatible endpoint) — inference; `text-embedding-v3` — embeddings
- Pydantic — structured-output contracts for agent I/O
- ChromaDB — vector store (Agent 1 only)
- Python (`re` for parsing, `csv` for archiving)

> The pipeline was originally built on local Ollama (`qwen2.5:7b` + `nomic-embed-text`) for zero-cost reproducible development, then migrated to DashScope cloud Qwen on 2026-07-02 for higher output quality and for the Qwen Hackathon deployment. The code still follows the same structure; only the client and embeddings changed.

## Setup

```bash
pip install langgraph langchain langchain-openai langchain-community langchain-chroma langchain-text-splitters dashscope python-dotenv pydantic
```

Set your DashScope API key in `agents/.env` (git-ignored). Note: if you use an **international** DashScope account, the `dashscope` SDK defaults to the domestic China endpoint and will 401 on embeddings — point it at the intl endpoint explicitly with `dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"`.

Agent 1 expects Etsy policy PDFs in a local folder referenced by `folder_path` in `agents/a1_compliance_agent.py` (currently a hardcoded absolute path — make it relative before running elsewhere).

## Known issues / next steps

**Performance / hardening**
- [ ] Chroma rebuilds the vector store on every run — load the persisted store instead of rebuilding (functional but wasteful)
- [ ] `folder_path` / `persist_directory` are hardcoded absolute paths — make relative or env-configured
- [ ] Move `SOFT_THRESHOLD` into `audit_config.py` with the other rule constants
- [ ] Add keyword-coverage check to `check_hard_rules()` as a threshold (e.g. ≥80%), not all-or-nothing
- [ ] Add attribute-consistency check (draft vs SKU data) to the hard gate

**State / shared modules**
- [ ] Lock the state class to one spelling everywhere (`PinpingoState`)

**Docs**
- [ ] Update the mermaid diagram to match the current architecture (A1 RAG compliance, A2 quality-gate extraction)

## Project context

Part of a larger experiment comparing human-written, pipeline-generated, and agent-generated Etsy listing copy. Full experimental design, pricing/COGS analysis, and compliance research are documented separately as part of the broader portfolio project.
