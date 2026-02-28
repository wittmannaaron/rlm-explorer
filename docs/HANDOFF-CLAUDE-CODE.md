# RLM Explorer — Handoff Document for Claude Code

**Date:** 2026-02-28
**Repo:** https://github.com/wittmannaaron/rlm-explorer
**Project Root:** `~/Projects/rlm-explorer`
**Status:** PoC with critical agent-loop bug preventing final answer synthesis

---

## 1. What This Project Is

RLM Explorer is a **Recursive Language Model** workbench for exploring ~718 German legal documents (~4.8MB, ~1.2M tokens) for a family law case. It is based on the RLM paper (arXiv:2512.24601) and inspired by the [fast-rlm](https://github.com/sidharthrajaram/fast-rlm) reference implementation.

The user asks questions in a chat UI. The system uses an LLM (GPT-5.2) to programmatically search and analyze the documents, then returns a comprehensive answer.

---

## 2. Architecture — DO NOT CHANGE THE CORE PATTERN

### The REPL Architecture (sacred, do not replace)

The core innovation of RLM is that **the context is NOT stuffed into the LLM prompt**. Instead:

1. The 4.8MB document corpus is loaded as a **Python variable** (`context`) in a sandboxed REPL
2. The LLM writes **Python code** (in ` ```repl ``` ` blocks) to explore, search, slice, and analyze the context
3. The REPL executes the code and returns stdout output to the LLM
4. This loop repeats until the LLM calls `FINAL(answer)` to return its answer
5. The LLM can spawn **sub-agents** via `await llm_query(prompt)` to process chunks in parallel

**Why this matters:** The LLM never sees the full 4.8MB in its context window. It uses code to navigate and extract only the relevant parts. This is fundamentally different from RAG or simple prompt-stuffing.

### System Components

```
poc/
├── rlm_engine.py      # Core engine: REPL sandbox + LLM loop + sub-agent orchestration
├── server.py           # Flask API server with session management
├── start.sh            # Startup script with env var configuration
├── concat_context.py   # Concatenates 718 .txt files with [DOCUMENT: filename] headers
├── static/index.html   # Chat UI (dark theme, session management, steps viewer)
├── logs/               # Timestamped log files (essential for debugging)
└── full_context.txt    # Generated: concatenated 4.8MB context (gitignored)

context/
├── PROMPT.md           # Domain briefing: role, case details, response guidelines (German)
└── *.txt               # 718 legal document files (gitignored)
```

### Data Flow

```
User Question
    ↓
server.py (Flask) — session management, history, findings
    ↓
rlm_engine.py: RLMEngine.run()
    ↓
    ├── Assemble full context: [BRIEFING] + documents + [CONVERSATION HISTORY] + [USER QUERY]
    ├── Create REPLSandbox with: context, FINAL, findings, add_finding, llm_query, asyncio
    ├── Run initial inspection code (document index, briefing, findings)
    └── Enter REPL loop:
         ├── LLM generates ```repl code
         ├── Sandbox executes code, returns stdout
         ├── If FINAL() called → return result
         ├── If llm_query() called → spawn sub-agent in ThreadPool
         └── Loop continues until FINAL or step limit
```

### Agent Hierarchy

```
Root Agent (depth=0)
├── Gets: SYSTEM_PROMPT, full 4.8MB context, llm_query function
├── Job: Chunk the context, spawn parallel sub-agents, synthesize results
├── Budget: 25 REPL rounds, 5 reserved LLM calls for synthesis
│
└── Sub-Agents (depth=1, up to 12 parallel)
    ├── Gets: LEAF_SYSTEM_PROMPT, one chunk (~400K chars) as context
    ├── Job: Search chunk for query-relevant information, return findings
    ├── Budget: 8 REPL rounds each
    └── NO llm_query function (cannot spawn further sub-agents)
```

---

## 3. The Current Problem

### Symptom
The root agent spawns ~10-12 sub-agents correctly, they search their chunks and return results. But **the root agent fails to produce a proper synthesized answer**. Instead, it either:

- Returns a raw dump of all sub-agent outputs (~360K chars unprocessed) as the "answer"
- Or hits a call/token limit and returns an error string as the answer

### Root Cause Analysis

The root agent makes **1 LLM call** which generates a single large REPL code block. This code:
1. Chunks the context into ~10-12 segments
2. Spawns sub-agents via `asyncio.gather()`
3. Collects results
4. **Attempts synthesis** — but this is where it fails

The problem is that the **synthesis step happens inside the same REPL execution as the gather()**. The LLM writes ALL the code in one shot — chunking, spawning, and synthesis — in a single `repl` block. When the gather() takes 90+ seconds and sub-agents consume most of the LLM call budget, the synthesis portion of the code either:

- **Calls `llm_query()` again for synthesis** (which fails because budget is exhausted)
- **Just concatenates results and calls FINAL()** (dumping raw output without real synthesis)
- **Never gets to the FINAL() call** because an error occurs in the synthesis sub-agent

The fundamental issue: **the LLM does not get a second turn after gather() returns** to intelligently process results. Everything happens in one REPL execution, and the LLM must predict the synthesis code BEFORE seeing the sub-agent results.

### What Does NOT Work (Already Tried)

1. **Telling the LLM "do NOT use llm_query for synthesis"** — The LLM then just concatenates raw results and calls FINAL(concatenated_dump). No real synthesis happens; the user gets 360K chars of raw agent output.

2. **Reserving LLM calls for root** — We reserved 5 calls for root, but the root still writes all its code in ONE step. The reserved calls go unused because the root doesn't need another LLM call — it already called FINAL() with garbage.

3. **Increasing/decreasing budgets** — More budget = more sub-agent loops but same synthesis failure. Less budget = sub-agents cut short AND synthesis fails.

### What Might Work (Ideas for Next Iteration)

**Option A: Two-Phase Root Agent**
Split the root agent's work into explicit phases:
- Phase 1: Chunk + gather (one REPL block, executed, results returned)
- Phase 2: The root gets ANOTHER LLM call where it SEES the sub-agent results and writes synthesis code
- This could be done by NOT calling FINAL() inside the gather code, so the loop continues and the LLM gets another turn with the results visible

**Option B: Engine-Level Post-Processing**
After the root agent returns (even with raw results), have the engine make one more LLM call specifically for synthesis. This would be a separate step outside the REPL loop.

**Option C: Smarter SYSTEM_PROMPT**
Restructure the prompt to tell the agent: "In your FIRST code block, ONLY chunk and gather. Do NOT call FINAL(). Print a summary of results. In your SECOND code block, read the printed results and write the final synthesis."

**Option D: Forced Two-Step via Engine**
After the gather() REPL execution completes (which returns printed output but no FINAL), the engine automatically provides the output back to the LLM. The LLM then writes a second REPL block that does the synthesis. This already works architecturally — the loop continues if FINAL is not called. The problem is that the LLM writes FINAL() prematurely inside the first block.

---

## 4. Configuration Reference

### RLMConfig (rlm_engine.py)

| Field | Default | Purpose |
|-------|---------|---------|
| `primary_model` | gpt-5.2 | Model for root agent |
| `sub_model` | gpt-5.2 | Model for sub-agents |
| `max_depth` | 2 | Agent depth limit (0=root, 1=sub, 2=leaf) |
| `max_calls_per_agent` | 25 | Max REPL rounds for root agent |
| `sub_agent_max_calls` | 8 | Max REPL rounds per sub-agent |
| `max_total_llm_calls` | 50 | Hard global limit across ALL agents |
| `root_reserved_calls` | 5 | Calls reserved for root synthesis (sub-agents stop at 45) |
| `max_concurrent_agents` | 12 | Max parallel sub-agents |
| `truncate_len` | 10000 | REPL output truncation (first half + last half) |
| `max_total_tokens` | 1,500,000 | Token budget (95% pre-check) |
| `timeout` | 180 | OpenAI API timeout |

### Environment Variables (start.sh / server.py)

All config fields can be set via environment variables: `RLM_PRIMARY_MODEL`, `RLM_MAX_DEPTH`, `RLM_MAX_CALLS`, `RLM_MAX_TOKENS`, `RLM_MAX_CONCURRENT`, `RLM_MAX_LLM_CALLS`, `RLM_SUB_AGENT_CALLS`, `RLM_ROOT_RESERVED`, `RLM_TIMEOUT`, `RLM_TRUNCATE_LEN`.

---

## 5. Key Files in Detail

### rlm_engine.py — The Core

- **SYSTEM_PROMPT** (line 81-208): Instructions for root agent. Contains REPL usage guide, parallelization examples, mandatory research strategy. THIS IS WHAT THE LLM READS.
- **LEAF_SYSTEM_PROMPT** (line 210-228): Simpler instructions for sub-agents (no llm_query available).
- **REPLSandbox** (line 235-313): Persistent Python execution environment. Handles sync code and async code (wraps in `async def` + `loop.run_until_complete`). Key: `_final_set` flag indicates FINAL() was called.
- **RLMEngine._run_agent()** (line 427-638): The main agent loop. Key points:
  - `llm_query` only injected at depth=0 (line 460)
  - Initial inspection code auto-runs (line 493-533)
  - Pre-checks before each LLM call: token budget (95% margin) + global call limit
  - Sub-agents see tighter limit (max_total - reserved) than root
  - Loop exits when `sandbox.is_done` (FINAL called) or step limit reached

### server.py — Flask Backend

- Session management: conversation history (last 10 turns) + findings buffer
- `load_context()`: Loads concatenated 4.8MB context + PROMPT.md
- `PROMPT.md` date placeholder: `{date}` → current date
- Endpoints: `/api/chat` (POST), `/api/info`, `/api/session`, `/api/reload-context`

### context/PROMPT.md — Domain Briefing

This file is injected as `[BRIEFING START]...[BRIEFING END]` at the beginning of the context variable. It tells the agent:
- Its role ("Anuk" — legal research assistant)
- The case details (Ihring vs. Ihring, family law)
- Involved parties and their relationships
- How to format responses (German, with document references)

---

## 6. How Logging Works

Logs are in `poc/logs/rlm_YYYYMMDD_HHMMSS.log`. Format: `%(asctime)s [%(levelname)-7s] %(name)s: %(message)s`

Key log patterns to search for:
- `[depth=0 step=X]` — root agent steps
- `[depth=1 step=X]` — sub-agent steps
- `Spawning sub-agent #N` — sub-agent creation
- `Sub-agent #N returned in Xs` — sub-agent completion
- `REPL done in Xs | output=N chars | final=True/False` — REPL execution results
- `Token budget exceeded` / `Global LLM call limit reached` — budget limits
- `Agent DONE` — successful completion with FINAL()
- `Agent hit step limit` — ran out of REPL rounds

---

## 7. What NOT to Do

1. **DO NOT replace the REPL architecture** with RAG, embeddings, or vector search. The REPL approach is the core innovation and must be preserved.

2. **DO NOT stuff the full context into the LLM prompt.** The context is 4.8MB (~1.2M tokens). It must stay as a Python variable that the LLM accesses via code.

3. **DO NOT remove sub-agent parallelism.** Sub-agents are essential for searching the large corpus efficiently. The issue is synthesis, not search.

4. **DO NOT fundamentally restructure the codebase.** This is a PoC. Fix the synthesis problem incrementally. The REPL loop, sandbox, and sub-agent spawning all work correctly.

5. **DO NOT change the Flask server architecture.** It works fine. The problem is entirely in how the root agent synthesizes sub-agent results.

---

## 8. How to Test

```bash
cd ~/Projects/rlm-explorer/poc
./start.sh
# Open http://localhost:5055
# Ask: "Was ist der aktuelle Status des Falls Ihring vs. Ihring?"
# Check logs in poc/logs/ for the latest file
```

### What Success Looks Like

- Root agent spawns 10-12 sub-agents (visible in logs)
- Sub-agents return findings from their chunks
- Root agent produces a **coherent, synthesized German-language answer** (not raw dumps)
- Answer references specific documents by `[DOCUMENT: ...]` names
- Total time: 60-120 seconds
- Total LLM calls: 20-50
- Total tokens: < 1,500,000

### What Failure Looks Like

- Answer is a raw concatenation of sub-agent outputs (360K+ chars)
- Answer is an error string like "[CALL LIMIT REACHED...]"
- Answer is very short (< 100 chars) — likely an error message passed through
- Root agent makes only 1 LLM call and never gets a second chance to synthesize

---

## 9. Technical Constraints

- **Model:** GPT-5.2 via OpenAI API (supports ~500K context per call)
- **OpenAI Prompt Caching:** Already achieving ~81% cache hit rate — context stays in cache across calls
- **Python:** 3.x, no special dependencies beyond flask, flask-cors, openai
- **Async pattern:** `llm_query` is async (for asyncio.gather), but internally dispatches to ThreadPoolExecutor for synchronous `_run_agent` calls
- **REPL sandbox:** Uses `exec()` with shared globals dict — variables persist across code executions within the same agent

---

## 10. Immediate Priority

**Fix the synthesis step so the root agent produces a coherent answer from sub-agent results.**

The search/chunking/sub-agent phase works. The problem is only in the final synthesis. The root agent needs to:
1. See the sub-agent results (it does — they're printed in the REPL output)
2. Get another LLM call to write synthesis code (it doesn't — it calls FINAL too early)
3. Call FINAL() with a properly synthesized answer

The most promising approach: ensure the root agent does NOT call FINAL() in the same code block as asyncio.gather(). Force a two-step pattern where gather results are printed first, then a second LLM turn generates the synthesis.
