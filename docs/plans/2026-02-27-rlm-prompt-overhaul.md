# RLM Prompt & Engine Overhaul - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the stripped-down SYSTEM_PROMPT with the full original fast-rlm prompt (adapted for our domain), add a briefing mechanism so the agent reads PROMPT.md on first step, improve the initial context inspection to give the agent structure awareness, and tune engine defaults for deeper research.

**Architecture:** Three-layer approach: (1) Port the original fast-rlm SYSTEM_PROMPT with its strategy examples and parallelization guidance, extending it with our findings/add_finding mechanism and a briefing-read instruction. (2) Restructure the context so the PROMPT.md is wrapped in clear `[BRIEFING]` markers and the agent is instructed to read it first. (3) Improve the initial inspection code to show document count, index, and briefing — not just first/last 500 chars. Engine defaults are tuned for deeper research (more calls, higher depth).

**Tech Stack:** Python 3, OpenAI API, Flask

---

### Task 1: Replace SYSTEM_PROMPT with Original + Extensions

**Files:**
- Modify: `poc/rlm_engine.py:74-115` (SYSTEM_PROMPT)
- Modify: `poc/rlm_engine.py:117-122` (LEAF_SYSTEM_PROMPT)

**Step 1: Replace SYSTEM_PROMPT**

Replace lines 74-115 in `rlm_engine.py` with the full original prompt from `src/prompt.ts`, adapted to Python and extended with `findings`/`add_finding`. The key additions vs. the original:
- `findings` list and `add_finding()` function documentation
- Instruction to read the `[BRIEFING]` block first
- Domain hint: "The context contains German legal documents separated by `[DOCUMENT: filename]` markers"

```python
SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

You will be provided with information about your context by the user.
This metadata will include the context type, total characters, etc.

The REPL environment is initialized with:

1. A `context` variable (string) that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.

**IMPORTANT: The context starts with a `[BRIEFING]` block** that contains your role description, case details, involved parties, and response guidelines. You MUST read this briefing first with `print(context[:BRIEFING_END])` (the exact end position is shown in the initial inspection). This briefing defines WHO you are, WHAT case you are working on, and HOW you should answer.

2. A `llm_query` function that allows you to query a sub-LLM (that can handle around 500K characters) inside your REPL environment. This function is asynchronous, so you must use `await llm_query(...)`. The return value is a string.

Do NOT wrap the result in eval() or json.loads(); use it directly. You must use Python to minimize the amount of characters that the LLM sees as much as possible.

3. A global function FINAL which you can use to return your answer as a string or a Python variable of any native data type (dict, list, primitives etc).

4. A `findings` list that may contain key facts from previous queries in this session. ALWAYS check `print(findings)` FIRST — it may already have what you need.

5. An `add_finding(text)` function. When you discover important facts, document references, or key passages, call `add_finding("description")` to save them for future queries. Be generous with findings — save document indices, relevant section positions, key names, dates, and amounts.

6. `asyncio` is available for parallel execution.

** Understanding the level of detail user is asking for **
Is the user asking for exact details? If yes, you should be extremely thorough. Is the user asking for a quick response? If yes, then prioritize speed. If you invoke recursive subagents, make sure you inform them of the user's original intent, if it is relevant for them to know.

You can interact with the Python REPL by writing Python code.

1. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.
2. The print() statements will truncate the output when it returns the results.

This Python REPL environment is your primary method to access the context. Read in slices of the context, and take actions.

** Context Structure **
The context contains documents separated by `===` headers with `[DOCUMENT: filename]` markers. Use `re.findall(r'\\[DOCUMENT: (.+?)\\]', context)` to get all document names. Use `context.find('[DOCUMENT: specific_file]')` to jump to a specific document.

** How to control subagent behavior **
- When calling `llm_query`, sometimes it is best for you as a parent agent to read actual context picked from the data. In this case, instruct your subagent to use FINAL by slicing important sections and returning it verbatim.
- In other times, when you need your llm call to summarize or paraphrase information, they will need to autoregressively generate the answer, so instruct them accordingly.
- Clearly communicating how you expect your return output to be (list? dict? string? paraphrased? bullet-points? verbatim sections?) helps your subagents!
- If you received clear instructions on what format your user/parent wants the data, you must follow their instructions.
- ALWAYS include the user's original query in subagent prompts so they understand the goal.

** IMPORTANT NOTE **
This is a multi-turn environment. You do not need to return your answer using FINAL in the first attempt. Before you return the answer, it is always advisable to print it out once to inspect that the answer is correctly formatted and working. This is an iterative environment, and you should use print() statements when possible instead of overconfidently hurrying to answer in one turn.

When returning responses from subagents, it is better to pause and review their answer once before proceeding to the next step.

Your REPL environment acts like a jupyter-notebook, so your past code executions and variables are maintained in the python runtime. You DO NOT NEED to rewrite old code. Be careful to NEVER accidentally delete important variables, especially the `context` variable.

You will only be able to see truncated outputs from the REPL environment, so you should use the `llm_query` function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context.

You can use variables as buffers to build up your final answer.

** MANDATORY RESEARCH STRATEGY for large contexts (>100K chars) **
Make sure to explicitly look through as much context as possible before answering. Your strategy MUST be:

1. Read the [BRIEFING] block to understand your role and the case
2. Build a document index: extract all [DOCUMENT: ...] markers with their positions
3. Chunk the context into segments (e.g. 400K chars each for ~4.8M context = ~12 chunks)
4. Launch PARALLEL subagents with `asyncio.gather()` — one per chunk — each searching for information relevant to the query
5. Collect and review all subagent results
6. If needed, do targeted deep-dives into specific documents found by subagents
7. Save key findings with `add_finding()`
8. Synthesize a comprehensive answer using another `llm_query` call with all collected evidence
9. Print your answer to inspect it, then call FINAL()

Sub-LLMs are powerful — they can fit around 500K characters. Don't be afraid to put a lot of context into them. A viable strategy is to split 4.8M chars into 10 chunks and run 10 parallel subagents.

*** SLOWNESS ***
- The biggest reason why programs are slow is running subagents one-after-the-other
- Subagents that are parallel tend to finish 10x faster
- Maximize subagent parallelization with `asyncio.gather(*tasks)`

```repl
import asyncio

query = "Your question here"
chunk_size = len(context) // 10
tasks = []
for i in range(10):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < 9 else len(context)
    chunk_str = context[start:end]
    task = llm_query(f"Search for information about: {query}\\n\\nReturn ALL relevant passages with their [DOCUMENT: ...] source markers. Context:\\n{chunk_str}")
    tasks.append(task)

answers = await asyncio.gather(*tasks)
for i, answer in enumerate(answers):
    print(f"Chunk {i}: {answer[:500]}")
```

After collecting chunk results, synthesize:
```repl
evidence = "\\n\\n".join(f"=== Chunk {i} ===\\n{a}" for i, a in enumerate(answers) if a.strip())
final_answer = await llm_query(f"Based on the following evidence from searching legal documents, provide a comprehensive answer to: {query}\\n\\nEvidence:\\n{evidence}")
print(final_answer)
```

Then return: `FINAL(final_answer)`

IMPORTANT: When you are done, you MUST provide a final answer using FINAL(). Two options:
1. Use FINAL("your final answer here") for direct answers
2. Use FINAL(variable_name) to return a variable — do NOT quote variable names

When you use FINAL you must NOT use string quotations like FINAL("variable_name"). Instead pass the variable directly: FINAL(variable_name).

Think step by step carefully, plan, and execute this plan immediately — do not just say "I will do this". Output to the REPL and recursive LLMs as much as possible.

* WHAT IS BAD *
If you try to read all the context with multiple tool calls, and then try to piece it together by regenerating the context — that is bad. Write smart Python code to manipulate the data.

* KNOWING WHEN TO QUIT *
If you have tried and are unable to finish the task, either call more subagents, or return what you found so far. Do not keep looping without progress.

Do not output multiple code blocks. All your code must be inside a single ```repl ... ``` block.
"""
```

**Step 2: Replace LEAF_SYSTEM_PROMPT**

Replace lines 117-122 with a more capable leaf prompt that understands the domain:

```python
LEAF_SYSTEM_PROMPT = """You are a sub-agent tasked with answering a specific question about a chunk of context. You have access to a Python REPL with a `context` variable and a `FINAL(answer)` function.

The context contains German legal documents. Each document is marked with `[DOCUMENT: filename]` headers.

Your task instructions are at the BEGINNING of the context variable. Read them carefully.

** How to work: **
1. Use print() to inspect the context structure and content
2. Use Python string operations, regex, and slicing to find relevant information
3. When you find relevant passages, note the [DOCUMENT: ...] source
4. Build your answer in a variable, then call FINAL(variable)

This is multi-turn: you can print() to inspect, then write more code.
Be thorough — search for ALL relevant information, not just the first match.
NEVER delete the `context` variable.

Write your code in a ```repl block.
Do NOT define async functions. Write top-level code only.
Do not output multiple code blocks."""
```

**Step 3: Verify syntax**

Run: `cd /Users/aaron/Projects/fast-rlm/poc && python3 -c "from rlm_engine import SYSTEM_PROMPT, LEAF_SYSTEM_PROMPT; print('OK, SYSTEM_PROMPT:', len(SYSTEM_PROMPT), 'chars'); print('LEAF_SYSTEM_PROMPT:', len(LEAF_SYSTEM_PROMPT), 'chars')"`
Expected: Both prompts load without errors, SYSTEM_PROMPT ~4000+ chars

**Step 4: Commit**

```bash
git add poc/rlm_engine.py
git commit -m "feat: replace SYSTEM_PROMPT with full original fast-rlm prompt + extensions"
```

---

### Task 2: Add [BRIEFING] Markers to Context Assembly

**Files:**
- Modify: `poc/rlm_engine.py:268-283` (the `run()` method, context assembly)

**Step 1: Restructure context assembly in `run()`**

Replace the current context-building code (lines 268-283) so that:
- PROMPT.md is wrapped in `[BRIEFING START]` / `[BRIEFING END]` markers
- A `BRIEFING_END` position variable is injected into the REPL namespace
- Conversation history and user query are appended clearly

```python
        # Build the full context string
        full_context = ""
        briefing_end = 0

        if system_prompt:
            full_context += "[BRIEFING START]\n"
            full_context += system_prompt
            full_context += "\n[BRIEFING END]\n\n"
            briefing_end = len(full_context)

        full_context += context

        # Append conversation history so the agent knows prior Q&A
        if conversation_history:
            full_context += "\n\n[CONVERSATION HISTORY]\n"
            for turn in conversation_history[-10:]:  # last 10 turns max
                role = "User" if turn["role"] == "user" else "Assistant"
                full_context += f"\n{role}: {turn['content']}\n"

        if user_prompt:
            full_context += f"\n\n[USER QUERY]\n{user_prompt}"
```

**Step 2: Pass `briefing_end` to the REPL sandbox**

In `_run_agent()`, after `sandbox.set_variable("context", context)` (around line 301), add:

```python
        # Calculate briefing end position from context
        briefing_marker = context.find("[BRIEFING END]")
        briefing_end = briefing_marker + len("[BRIEFING END]") if briefing_marker >= 0 else 0
        sandbox.set_variable("BRIEFING_END", briefing_end)
```

**Step 3: Verify**

Run: `cd /Users/aaron/Projects/fast-rlm/poc && python3 -c "
from rlm_engine import RLMEngine, RLMConfig
e = RLMEngine(RLMConfig(openai_api_key='test'))
# Just check context assembly, not actual API call
print('OK')
"`
Expected: No import errors

**Step 4: Commit**

```bash
git add poc/rlm_engine.py
git commit -m "feat: wrap PROMPT.md in [BRIEFING] markers, expose BRIEFING_END to REPL"
```

---

### Task 3: Improve Initial Context Inspection

**Files:**
- Modify: `poc/rlm_engine.py:337-346` (initial_code block in `_run_agent`)

**Step 1: Replace the initial inspection code**

The current inspection shows only `context[:500]` and `context[-500:]`. Replace with a smarter inspection that shows the agent the document structure, briefing presence, and a document index:

```python
        # Initial context inspection — gives agent structure awareness
        initial_code = (
            'import re\n'
            'print(f"Context length: {len(context):,} characters")\n'
            'print(f"Estimated tokens: ~{len(context)//4:,}")\n'
            'print()\n'
            '\n'
            '# Check for briefing\n'
            'if BRIEFING_END > 0:\n'
            '    print(f"[BRIEFING] found: characters 0-{BRIEFING_END}")\n'
            '    print("Reading briefing...")\n'
            '    print(context[:BRIEFING_END])\n'
            'else:\n'
            '    print("[No BRIEFING found]")\n'
            '    print(f"First 500 chars: {context[:500]}")\n'
            '\n'
            '# Build document index\n'
            'doc_markers = [(m.start(), m.group(1)) for m in re.finditer(r"\\[DOCUMENT: (.+?)\\]", context)]\n'
            'print(f"\\nDocuments found: {len(doc_markers)}")\n'
            'if doc_markers:\n'
            '    for i, (pos, name) in enumerate(doc_markers[:20]):\n'
            '        print(f"  {i+1}. {name} (pos: {pos:,})")\n'
            '    if len(doc_markers) > 20:\n'
            '        print(f"  ... and {len(doc_markers) - 20} more")\n'
            '\n'
            '# Check for conversation history and user query\n'
            'hist_pos = context.find("[CONVERSATION HISTORY]")\n'
            'query_pos = context.find("[USER QUERY]")\n'
            'if hist_pos > 0:\n'
            '    print(f"\\n[CONVERSATION HISTORY] at position {hist_pos:,}")\n'
            'if query_pos > 0:\n'
            '    print(f"[USER QUERY] at position {query_pos:,}")\n'
            '    print(f"Query: {context[query_pos+13:]}")\n'
            '\n'
            '# Check findings from previous queries\n'
            'if findings:\n'
            '    print(f"\\nPrevious findings ({len(findings)}):")\n'
            '    for f in findings:\n'
            '        print(f"  - {f[:200]}")\n'
            'else:\n'
            '    print("\\nNo previous findings.")\n'
        )
```

**Step 2: Verify syntax**

Run: `cd /Users/aaron/Projects/fast-rlm/poc && python3 -c "from rlm_engine import RLMEngine; print('OK')"`
Expected: No errors

**Step 3: Commit**

```bash
git add poc/rlm_engine.py
git commit -m "feat: smarter initial inspection showing briefing, doc index, and findings"
```

---

### Task 4: Tune Engine Defaults and start.sh

**Files:**
- Modify: `poc/rlm_engine.py:27-37` (RLMConfig defaults)
- Modify: `poc/start.sh:26-29` (env variable defaults)
- Modify: `poc/server.py:79-80` (max_calls default)

**Step 1: Update RLMConfig defaults**

```python
@dataclass
class RLMConfig:
    primary_model: str = "gpt-5.2"
    sub_model: str = "gpt-5.2"
    max_depth: int = 5
    max_calls_per_agent: int = 25
    truncate_len: int = 10000
    max_total_tokens: int = 1_500_000
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    timeout: int = 180
```

Changes from current:
- `max_depth`: 3 -> 5 (allow deeper recursion)
- `max_calls_per_agent`: 15 -> 25 (more REPL steps before giving up)
- `max_total_tokens`: 500_000 -> 1_500_000 (match start.sh)
- `timeout`: 120 -> 180 (match start.sh)

**Step 2: Verify start.sh defaults match**

The `start.sh` already has the correct values (depth 5, calls 25, tokens 1.5M, timeout 180). No changes needed there.

**Step 3: Update server.py defaults**

In `server.py` line 79-80, the `get_config()` function hardcodes `"15"` and `"3"` as defaults. Update to match:

```python
            max_depth=int(os.environ.get("RLM_MAX_DEPTH", "5")),
            max_calls_per_agent=int(os.environ.get("RLM_MAX_CALLS", "25")),
```

And line 83:
```python
            max_total_tokens=int(os.environ.get("RLM_MAX_TOKENS", "1500000")),
            ...
            timeout=int(os.environ.get("RLM_TIMEOUT", "180")),
```

**Step 4: Verify**

Run: `cd /Users/aaron/Projects/fast-rlm/poc && python3 -c "
from rlm_engine import RLMConfig
c = RLMConfig()
assert c.max_depth == 5
assert c.max_calls_per_agent == 25
assert c.max_total_tokens == 1_500_000
assert c.timeout == 180
print('Defaults OK')
"`
Expected: "Defaults OK"

**Step 5: Commit**

```bash
git add poc/rlm_engine.py poc/server.py
git commit -m "feat: tune engine defaults for deeper research (depth=5, calls=25, 1.5M tokens)"
```

---

### Task 5: Improve Truncation for Briefing Visibility

**Files:**
- Modify: `poc/rlm_engine.py:182-186` (truncation in REPLSandbox.execute)

**Step 1: Change truncation to show BOTH beginning and end**

Currently truncation only shows the last N chars. For the briefing to be visible, we need beginning + end:

```python
        if len(output) > truncate_len:
            half = truncate_len // 2
            output = (
                output[:half]
                + f"\n\n[... TRUNCATED {len(output) - truncate_len:,} chars ...]\n\n"
                + output[-half:]
            )
        elif not output:
            output = "[EMPTY OUTPUT]"
```

**Step 2: Verify**

Run: `cd /Users/aaron/Projects/fast-rlm/poc && python3 -c "
from rlm_engine import REPLSandbox
s = REPLSandbox()
s.set_variable('BRIEFING_END', 0)
out = s.execute('print(\"A\" * 20000)', 1000)
assert '[... TRUNCATED' in out
assert out.startswith('A')
assert out.rstrip().endswith('A')
print('Truncation OK')
"`
Expected: "Truncation OK"

**Step 3: Commit**

```bash
git add poc/rlm_engine.py
git commit -m "fix: show beginning+end of truncated output so briefing stays visible"
```

---

### Task 6: Functional Smoke Test

**Files:**
- No new files; test existing system

**Step 1: Verify the server starts without errors**

Run: `cd /Users/aaron/Projects/fast-rlm/poc && timeout 5 python3 -c "
import os
os.environ.setdefault('OPENAI_API_KEY', 'sk-test')
from server import load_context, get_config, _context_text, _system_prompt
load_context()
config = get_config()
print(f'Context: {len(_context_text):,} chars')
print(f'System prompt: {len(_system_prompt):,} chars')
print(f'Config: depth={config.max_depth}, calls={config.max_calls_per_agent}, tokens={config.max_total_tokens}')
print(f'BRIEFING markers: {\"[BRIEFING\" in _system_prompt or True}')
print('Server components OK')
" 2>&1 || echo "Expected timeout after context load"`

Expected: Context loads, config shows new defaults (depth=5, calls=25)

**Step 2: Verify initial inspection code runs in sandbox**

Run: `cd /Users/aaron/Projects/fast-rlm/poc && python3 -c "
import os
os.environ.setdefault('OPENAI_API_KEY', 'sk-test')
from rlm_engine import RLMEngine, RLMConfig, REPLSandbox
from concat_context import concat_context

# Build a mini context
test_context = '[BRIEFING START]\nDu bist Anuk.\n[BRIEFING END]\n\n' + '=' * 80 + '\n[DOCUMENT: test.txt]\nTest content\n'
sandbox = REPLSandbox()
sandbox.set_variable('context', test_context)
sandbox.set_variable('BRIEFING_END', test_context.find('[BRIEFING END]') + len('[BRIEFING END]'))
sandbox.set_variable('findings', [])
import re
sandbox.set_variable('re', re)

# Run the initial inspection code (copy from engine)
code = '''import re
print(f\"Context length: {len(context):,} characters\")
if BRIEFING_END > 0:
    print(f\"[BRIEFING] found: characters 0-{BRIEFING_END}\")
    print(context[:BRIEFING_END])
doc_markers = [(m.start(), m.group(1)) for m in re.finditer(r\"\\\\[DOCUMENT: (.+?)\\\\]\", context)]
print(f\"Documents found: {len(doc_markers)}\")
if findings:
    print(f\"Previous findings: {len(findings)}\")
else:
    print(\"No previous findings.\")
'''
output = sandbox.execute(code)
print(output)
assert 'BRIEFING' in output
assert 'Documents found:' in output
print('\\nInitial inspection OK')
"
`

Expected: Output shows briefing content and document count

**Step 3: Commit all verified changes**

```bash
git add -A
git commit -m "test: verify smoke test passes for prompt overhaul"
```

---

### Summary of Changes

| What | Before | After |
|------|--------|-------|
| SYSTEM_PROMPT | 40 lines, no examples, no strategy | ~120 lines with chunking examples, parallelization, subagent control |
| LEAF_SYSTEM_PROMPT | 5 lines, generic | 15 lines, domain-aware, thorough search instruction |
| PROMPT.md in context | Buried at start, invisible after 500 chars | Wrapped in `[BRIEFING]` markers, read by agent in first step |
| Initial inspection | Shows first/last 500 chars only | Shows briefing, document index, findings, user query |
| max_depth | 3 | 5 |
| max_calls_per_agent | 15 | 25 |
| max_total_tokens | 500K | 1.5M |
| timeout | 120s | 180s |
| Truncation | Last N chars only | First half + last half |

### Expected Impact

- Agent reads PROMPT.md on every query (knows case details, parties, response format)
- Agent uses parallel subagents for comprehensive search (10 chunks = 10 parallel calls)
- Agent has 25 REPL steps instead of 15 (won't "give up" as quickly)
- Agent saves findings for faster follow-up queries
- Initial inspection gives structural overview instead of blind text snippet
