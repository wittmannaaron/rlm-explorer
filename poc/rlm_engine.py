"""
Pure Python implementation of the Recursive Language Model (RLM) engine.
Based on the RLM paper (arXiv:2512.24601) and the fast-rlm reference implementation.

Core idea: Instead of stuffing the entire context into the LLM prompt,
the context is loaded as a variable in a Python REPL. The LLM writes
Python code to explore, search, chunk, and analyze the context programmatically.
Sub-agents can be spawned to process chunks in parallel.
"""

import asyncio
import io
import contextlib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI

logger = logging.getLogger("rlm_engine")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RLMConfig:
    primary_model: str = "gpt-5.2"
    sub_model: str = "gpt-5.2"
    max_depth: int = 2              # 0=root, 1=sub-agents, 2=leaf (keep shallow!)
    max_calls_per_agent: int = 25
    truncate_len: int = 10000
    max_total_tokens: int = 1_500_000
    max_concurrent_agents: int = 12  # Global limit on parallel sub-agents
    max_total_llm_calls: int = 50    # Hard global limit across ALL agents
    sub_agent_max_calls: int = 8     # Max REPL rounds per sub-agent (depth>=1)
    root_reserved_calls: int = 5     # Calls reserved for root agent synthesis
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    timeout: int = 180


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    llm_calls: int = 0
    sub_agent_calls: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        return self.end_time - self.start_time if self.end_time else time.time() - self.start_time

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "llm_calls": self.llm_calls,
            "sub_agent_calls": self.sub_agent_calls,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


# ---------------------------------------------------------------------------
# RLM System Prompt (adapted from fast-rlm)
# ---------------------------------------------------------------------------

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
6. Save key findings with `add_finding()`
7. Synthesize the answer YOURSELF in Python — do NOT use llm_query for synthesis! Just combine the subagent results with string operations. The subagents already did the analysis; you just need to format and merge their findings.
8. Print your answer to inspect it, then call FINAL()

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

After collecting chunk results, synthesize DIRECTLY in Python (do NOT call llm_query again!):
```repl
# Combine all non-empty results
parts = []
for i, answer in enumerate(answers):
    text = answer.strip() if answer else ""
    if text and "[CALL LIMIT" not in text and "[TOKEN BUDGET" not in text:
        parts.append(f"### Ergebnisse aus Abschnitt {i+1}\\n{text}")

# Save findings for future queries
for part in parts:
    add_finding(part[:500])

# Build final answer
final_answer = f"## {query}\\n\\n" + "\\n\\n".join(parts)
print(f"Answer length: {len(final_answer):,} chars, {len(parts)} sections")
FINAL(final_answer)
```

CRITICAL: Do NOT use `llm_query` for the final synthesis step — you will run out of budget! Just combine results with Python string operations and call FINAL() directly.

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


# ---------------------------------------------------------------------------
# REPL Sandbox
# ---------------------------------------------------------------------------

class REPLSandbox:
    """A persistent Python execution environment for the RLM agent.

    Uses synchronous execution by default. For code containing `await`,
    wraps it in an async function and runs it in a dedicated event loop
    on the current thread. Sub-agent calls (llm_query) use a thread pool
    internally so they don't block the event loop.
    """

    def __init__(self):
        self._globals = {"__builtins__": __builtins__}
        self._final_result = None
        self._final_set = False

        def final_fn(x):
            self._final_result = x
            self._final_set = True

        self._globals["FINAL"] = final_fn

    def set_variable(self, name: str, value):
        self._globals[name] = value

    def execute(self, code: str, truncate_len: int = 10000) -> str:
        """Execute Python code synchronously and return captured stdout."""
        stdout_capture = io.StringIO()
        self._final_set = False
        code_preview = code[:120].replace('\n', '\\n')
        logger.debug(f"REPL execute: {code_preview}...")
        exec_start = time.time()

        try:
            if "await " in code:
                # Wrap top-level code in async function and run it
                indented = "\n".join("    " + line for line in code.split("\n"))
                wrapped = (
                    f"async def __rlm_async__():\n"
                    f"{indented}\n"
                )
                exec(compile(wrapped, "<repl>", "exec"), self._globals)

                loop = asyncio.new_event_loop()
                try:
                    with contextlib.redirect_stdout(stdout_capture):
                        loop.run_until_complete(self._globals["__rlm_async__"]())
                finally:
                    # Shut down pending tasks gracefully
                    _shutdown_loop(loop)
            else:
                with contextlib.redirect_stdout(stdout_capture):
                    exec(compile(code, "<repl>", "exec"), self._globals)
        except Exception as e:
            stdout_capture.write(f"\nError: {type(e).__name__}: {e}\n")

        exec_elapsed = time.time() - exec_start
        logger.debug(f"REPL execute finished in {exec_elapsed:.2f}s (final_set={self._final_set})")

        output = stdout_capture.getvalue()

        if len(output) > truncate_len:
            half = truncate_len // 2
            output = (
                output[:half]
                + f"\n\n[... TRUNCATED {len(output) - truncate_len:,} chars ...]\n\n"
                + output[-half:]
            )
        elif not output:
            output = "[EMPTY OUTPUT]"

        return output

    @property
    def final_result(self):
        return self._final_result

    @property
    def is_done(self) -> bool:
        return self._final_set


def _shutdown_loop(loop: asyncio.AbstractEventLoop):
    """Gracefully shut down an event loop, cancelling pending tasks."""
    try:
        # Cancel all pending tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except Exception:
        pass
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()


# ---------------------------------------------------------------------------
# RLM Engine (fully synchronous, uses threads for parallel sub-agents)
# ---------------------------------------------------------------------------

# Shared thread pool for parallel sub-agent execution
_executor = ThreadPoolExecutor(max_workers=12)
# Global semaphore to limit concurrent sub-agents (set per-engine in run())
_agent_semaphore: Optional[asyncio.Semaphore] = None


class RLMEngine:
    """The main RLM engine that orchestrates LLM + REPL interaction.

    All methods are synchronous. The OpenAI client is synchronous.
    llm_query is exposed as an async function to the REPL but internally
    dispatches synchronous _run_agent calls to a thread pool, enabling
    asyncio.gather() parallelism without nested event loops.
    """

    def __init__(self, config: RLMConfig, on_step=None):
        self.config = config
        self.client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            timeout=config.timeout,
        )
        self.usage = UsageStats()
        self.on_step = on_step
        self.steps_log = []

    def run(
        self,
        context: str,
        user_prompt: str = "",
        system_prompt: str = "",
        conversation_history: list = None,
        findings: list = None,
    ) -> dict:
        """Run the RLM agent on the given context with a user question.

        Args:
            context: The full concatenated document context.
            user_prompt: The current user question.
            system_prompt: Domain-specific system prompt (e.g. PROMPT.md).
            conversation_history: List of {"role": "user"/"assistant", "content": ...}
                from previous turns in this session.
            findings: List of strings - key findings from previous queries that
                the agent stored. These are loaded into a `findings` REPL variable.
        """
        self.usage = UsageStats(start_time=time.time())
        self.steps_log = []
        self._new_findings = []
        self._max_concurrent = self.config.max_concurrent_agents
        logger.info(f"=== RLM run() started | prompt={user_prompt[:100]!r}... | context={len(context):,} chars | max_depth={self.config.max_depth} max_concurrent={self._max_concurrent} ===")

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

        logger.info(f"Full context assembled: {len(full_context):,} chars | history_turns={len(conversation_history) if conversation_history else 0} | findings={len(findings) if findings else 0}")
        result = self._run_agent(full_context, depth=0, findings=findings or [])

        self.usage.end_time = time.time()
        logger.info(
            f"=== RLM run() finished in {self.usage.elapsed_seconds:.1f}s | "
            f"llm_calls={self.usage.llm_calls} sub_agents={self.usage.sub_agent_calls} "
            f"total_tokens={self.usage.total_tokens:,} ==="
        )
        return {
            "answer": str(result) if result is not None else "No answer produced.",
            "usage": self.usage.to_dict(),
            "steps": self.steps_log,
            "new_findings": self._new_findings,
        }

    def _run_agent(self, context: str, depth: int, findings: list = None) -> Optional[str]:
        """Run a single RLM agent at a given depth. Fully synchronous."""
        model = self.config.primary_model if depth == 0 else self.config.sub_model
        is_leaf = depth >= self.config.max_depth
        system = LEAF_SYSTEM_PROMPT if is_leaf else SYSTEM_PROMPT
        agent_start = time.time()
        logger.info(f"[depth={depth}] Agent started | model={model} leaf={is_leaf} context_len={len(context):,}")

        sandbox = REPLSandbox()
        sandbox.set_variable("context", context)
        sandbox.set_variable("asyncio", asyncio)

        # Calculate briefing end position from context
        briefing_marker = context.find("[BRIEFING END]")
        briefing_end = briefing_marker + len("[BRIEFING END]") if briefing_marker >= 0 else 0
        sandbox.set_variable("BRIEFING_END", briefing_end)

        # Findings buffer: persistent across session, readable and writable
        findings_list = list(findings) if findings else []
        sandbox.set_variable("findings", findings_list)

        engine_ref = self

        def add_finding(text: str):
            """Store a key finding for future queries in this session."""
            findings_list.append(text)
            engine_ref._new_findings.append(text)

        sandbox.set_variable("add_finding", add_finding)

        # Set up llm_query ONLY for root agent (depth=0).
        # Sub-agents at depth>=1 are leaf-like: they analyze their chunk directly
        # without spawning further sub-agents. This prevents exponential explosion.
        if depth == 0:
            engine_ref = self
            max_concurrent = self._max_concurrent

            async def llm_query(prompt: str) -> str:
                # Check total sub-agent count before spawning
                if engine_ref.usage.sub_agent_calls >= max_concurrent:
                    logger.warning(f"[depth={depth}] Sub-agent limit reached ({engine_ref.usage.sub_agent_calls} >= {max_concurrent}), skipping")
                    return "[SUB-AGENT LIMIT REACHED - synthesize results directly with Python instead of calling llm_query again.]"
                # Check if call budget is nearly exhausted
                sub_limit = engine_ref.config.max_total_llm_calls - engine_ref.config.root_reserved_calls
                if engine_ref.usage.llm_calls >= sub_limit:
                    logger.warning(f"[depth={depth}] Call budget exhausted for sub-agents ({engine_ref.usage.llm_calls} >= {sub_limit}), skipping")
                    return "[CALL BUDGET EXHAUSTED - synthesize results directly with Python instead of calling llm_query again.]"
                sub_id = engine_ref.usage.sub_agent_calls + 1
                engine_ref.usage.sub_agent_calls += 1
                logger.info(f"[depth={depth}] Spawning sub-agent #{sub_id} at depth={depth+1} | prompt_len={len(prompt):,}")
                sub_start = time.time()
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    _executor,
                    engine_ref._run_agent,
                    prompt,
                    depth + 1,
                )
                sub_elapsed = time.time() - sub_start
                result_str = str(result) if result is not None else ""
                logger.info(f"[depth={depth}] Sub-agent #{sub_id} returned in {sub_elapsed:.1f}s | result_len={len(result_str):,}")
                return result_str

            sandbox.set_variable("llm_query", llm_query)

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

        initial_output = sandbox.execute(initial_code, self.config.truncate_len)

        messages = [
            {"role": "user", "content": (
                f"Outputs are truncated to last {self.config.truncate_len} characters.\n"
                f"code:\n```repl\n{initial_code}\n```\n"
                f"Output:\n{initial_output}"
            )}
        ]

        self._log_step(depth, 0, initial_code, initial_output)

        # Sub-agents get fewer REPL rounds than the root agent
        max_steps = self.config.max_calls_per_agent if depth == 0 else self.config.sub_agent_max_calls

        for step in range(max_steps):
            # --- PRE-CHECKS before each LLM call ---
            # 1. Token budget (with 5% safety margin for parallel agents)
            token_limit = int(self.config.max_total_tokens * 0.95)
            if self.usage.total_tokens > token_limit:
                logger.warning(f"[depth={depth}] Token budget exceeded: {self.usage.total_tokens:,} > {token_limit:,} (95% of {self.config.max_total_tokens:,})")
                if sandbox.is_done:
                    return sandbox.final_result
                return f"[TOKEN BUDGET EXCEEDED after {step} steps — returning partial results]"

            # 2. Global LLM call limit (sub-agents see a tighter limit to reserve calls for root synthesis)
            effective_limit = self.config.max_total_llm_calls
            if depth > 0:
                effective_limit = self.config.max_total_llm_calls - self.config.root_reserved_calls
            if self.usage.llm_calls >= effective_limit:
                logger.warning(f"[depth={depth}] Global LLM call limit reached: {self.usage.llm_calls} >= {effective_limit} (reserved={self.config.root_reserved_calls if depth > 0 else 0} for root)")
                if sandbox.is_done:
                    return sandbox.final_result
                return f"[CALL LIMIT REACHED after {self.usage.llm_calls} calls — returning partial results]"

            # Call LLM (synchronous)
            self.usage.llm_calls += 1
            llm_start = time.time()
            msg_count = len(messages)
            msg_chars = sum(len(m.get("content", "")) for m in messages)
            logger.info(f"[depth={depth} step={step+1}] LLM call #{self.usage.llm_calls} | model={model} msgs={msg_count} msg_chars={msg_chars:,}")
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system}] + messages,
                )
            except Exception as e:
                logger.error(f"[depth={depth} step={step+1}] LLM API Error after {time.time()-llm_start:.1f}s: {e}")
                return f"LLM API Error: {e}"

            llm_elapsed = time.time() - llm_start
            # Track usage
            if response.usage:
                self.usage.prompt_tokens += response.usage.prompt_tokens or 0
                self.usage.completion_tokens += response.usage.completion_tokens or 0
                self.usage.total_tokens += response.usage.total_tokens or 0
                if hasattr(response.usage, "prompt_tokens_details") and response.usage.prompt_tokens_details:
                    self.usage.cached_tokens += getattr(
                        response.usage.prompt_tokens_details, "cached_tokens", 0
                    ) or 0
            logger.info(
                f"[depth={depth} step={step+1}] LLM response in {llm_elapsed:.1f}s | "
                f"tokens: prompt={response.usage.prompt_tokens if response.usage else '?'} "
                f"completion={response.usage.completion_tokens if response.usage else '?'} "
                f"total_so_far={self.usage.total_tokens:,}"
            )

            content = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": content})

            # Extract code from ```repl blocks
            repl_blocks = re.findall(r"```repl(.*?)```", content, re.DOTALL)
            code = "\n".join(block.strip() for block in repl_blocks)

            if not code:
                logger.warning(f"[depth={depth} step={step+1}] No repl block in LLM response ({len(content)} chars)")
                messages.append({
                    "role": "user",
                    "content": "Error: No ```repl code block found. Please write Python code in a ```repl block."
                })
                self._log_step(depth, step + 1, "[no code]", "Waiting for repl block...")
                continue

            # Execute code (synchronous, async-await handled inside sandbox)
            logger.info(f"[depth={depth} step={step+1}] Executing REPL code ({len(code)} chars)")
            exec_start = time.time()
            output = sandbox.execute(code, self.config.truncate_len)
            exec_elapsed = time.time() - exec_start
            logger.info(f"[depth={depth} step={step+1}] REPL done in {exec_elapsed:.1f}s | output={len(output)} chars | final={sandbox.is_done}")
            self._log_step(depth, step + 1, code, output)

            if sandbox.is_done:
                total_elapsed = time.time() - agent_start
                logger.info(f"[depth={depth}] Agent DONE in {total_elapsed:.1f}s after {step+1} steps")
                return sandbox.final_result

            messages.append({
                "role": "user",
                "content": f"Output:\n{output}"
            })

        total_elapsed = time.time() - agent_start
        logger.warning(f"[depth={depth}] Agent hit step limit ({max_steps}) after {total_elapsed:.1f}s")
        return "Agent did not produce a final answer within the step limit."

    def _log_step(self, depth: int, step: int, code: str, output: str):
        entry = {
            "depth": depth,
            "step": step,
            "code": code[:2000],
            "output": output[:5000],
        }
        self.steps_log.append(entry)
        if self.on_step:
            self.on_step(entry)
