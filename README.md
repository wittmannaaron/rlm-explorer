# rlm-explorer

An experimental workbench for exploring large document contexts using Recursive Language Models (RLMs).

## What is this?

rlm-explorer is a pure Python re-implementation of the RLM concept — an inference technique where an LLM interacts with arbitrarily long prompts through a Python REPL. Instead of stuffing millions of tokens into a single prompt, the LLM writes code to explore, chunk, and analyze the context programmatically. It can spawn parallel sub-agents to process different sections simultaneously.

This project is a research playground, not a production tool. It was built to test whether RLM can outperform classical RAG for exploring large document collections (~1.2M tokens) in terms of answer quality and comprehensiveness.

## How it works

```
User Question
     │
     ▼
┌─────────────┐
│  RLM Engine  │  ← System prompt with research strategy
│              │
│  context var │  ← 4.8MB of documents loaded as Python string
│  llm_query() │  ← Spawns sub-agents for parallel processing
│  findings[]  │  ← Persistent memory across queries
│  FINAL()     │  ← Returns the answer
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│          Python REPL Sandbox          │
│                                       │
│  Agent writes code to:                │
│  • Slice & search the context         │
│  • Chunk into segments                │
│  • Launch parallel sub-agents         │
│  • Collect & synthesize results       │
│  • Store findings for follow-up       │
└──────────────────────────────────────┘
```

The key insight from the RLM paper: the LLM never sees the full context directly. It only sees truncated previews and writes Python code to navigate the data. Sub-agent results are returned as variables, not loaded into the parent's context window.

## Quick start

```bash
# Clone
git clone https://github.com/wittmannaaron/rlm-explorer.git
cd rlm-explorer

# Set your API key
export OPENAI_API_KEY=sk-...

# Put your documents in context/ as .txt files
mkdir -p context
cp /path/to/your/documents/*.txt context/

# Optionally add a domain-specific briefing
# (see poc/README.md for PROMPT.md format)

# Start
cd poc
./start.sh
```

Open `http://localhost:5055` in your browser.

## Project structure

```
rlm-explorer/
├── poc/                      # The PoC implementation
│   ├── rlm_engine.py         # Core RLM engine (REPL sandbox, agent loop, sub-agents)
│   ├── server.py             # Flask API server with session management
│   ├── concat_context.py     # Concatenates .txt files into single context
│   ├── start.sh              # Start script with env defaults
│   ├── requirements.txt      # Python dependencies
│   ├── logs/                 # Timestamped log files (gitignored)
│   └── static/
│       └── index.html        # Chat UI (dark theme, steps viewer, findings panel)
└── context/                  # Your documents go here (gitignored)
    └── PROMPT.md             # Optional: domain-specific agent briefing
```

## Configuration

Environment variables (set before running or in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `RLM_PRIMARY_MODEL` | `gpt-5.2` | Model for the root agent |
| `RLM_SUB_MODEL` | `gpt-5.2` | Model for sub-agents |
| `RLM_MAX_DEPTH` | `2` | Agent depth limit (0=root, 1=sub, 2=leaf) |
| `RLM_MAX_CALLS` | `25` | Max REPL steps for root agent |
| `RLM_SUB_AGENT_CALLS` | `8` | Max REPL steps per sub-agent |
| `RLM_MAX_LLM_CALLS` | `50` | Hard global limit across all agents |
| `RLM_ROOT_RESERVED` | `5` | LLM calls reserved for root synthesis |
| `RLM_MAX_CONCURRENT` | `12` | Max parallel sub-agents |
| `RLM_MAX_TOKENS` | `1500000` | Token budget per query |
| `RLM_TRUNCATE_LEN` | `10000` | REPL output truncation (chars) |
| `RLM_TIMEOUT` | `180` | API call timeout (seconds) |
| `RLM_PORT` | `5055` | Server port |

## Features

- **Parallel sub-agents**: The root agent chunks the context and spawns up to 12 sub-agents via `asyncio.gather()` to search different sections simultaneously
- **Two-phase execution**: The engine enforces a two-phase pattern — Phase 1 (search via sub-agents) and Phase 2 (synthesis with actual results visible). If the LLM calls `FINAL()` prematurely in the same step as `gather()`, the engine suppresses it and forces a second turn
- **Fallback synthesis**: If the REPL loop exhausts its call budget without producing an answer, a dedicated LLM call outside the loop synthesizes sub-agent results into a coherent response
- **Session memory**: Conversation history (last 10 turns) and a persistent findings buffer that carries key facts across queries
- **Briefing system**: A `[BRIEFING]` block at the start of the context defines the agent's role, domain knowledge, and response guidelines — read automatically on each query
- **Smart initial inspection**: The agent starts each query with a structural overview: briefing content, document index, previous findings, and the user's question
- **Step viewer**: The chat UI shows every REPL step (code + output) at each depth level for full transparency
- **Budget controls**: Hard limits on LLM calls, tokens, concurrent agents, and recursion depth prevent runaway costs

## Domain briefing (PROMPT.md)

Place a `PROMPT.md` file in `context/` to give the agent domain-specific instructions. This file is wrapped in `[BRIEFING START/END]` markers and read by the agent at the beginning of every query. It can contain:

- Role definition (who the agent is, what it specializes in)
- Case/project details (parties, key facts, dates)
- Search strategy hints (what to look for, how to prioritize)
- Response format requirements (citations, structure, language)

Without a PROMPT.md, the agent operates as a general-purpose document explorer.

## Inspiration & references

This project is inspired by and builds upon:

- **RLM Paper**: Nishant Aklecha et al., "Recursive LLMs: A New Inference Paradigm" ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601)), MIT
- **fast-rlm**: Reference implementation by Arjun Viswanathan ([github.com/avbiswas/fast-rlm](https://github.com/avbiswas/fast-rlm))

## License

MIT
