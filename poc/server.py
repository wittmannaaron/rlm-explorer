#!/usr/bin/env python3
"""
Flask Chat Server for the RLM Proof of Concept.
Serves the HTML chat UI and provides an API endpoint for RLM queries.
Maintains per-session conversation history and findings buffer.
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

from rlm_engine import RLMEngine, RLMConfig
from concat_context import concat_context

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONTEXT_DIR = os.path.join(os.path.dirname(__file__), "..", "context")
PROMPT_FILE = os.path.join(CONTEXT_DIR, "PROMPT.md")
CONTEXT_CACHE = os.path.join(os.path.dirname(__file__), "full_context.txt")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

# Global state
_context_text: str = ""
_system_prompt: str = ""
_config: RLMConfig = None

# Session store: session_id -> { history: [...], findings: [...] }
_sessions: dict = {}


def load_context():
    """Load or generate the concatenated context."""
    global _context_text, _system_prompt

    if not os.path.exists(CONTEXT_CACHE):
        print("Concatenating context files...")
        concat_context(CONTEXT_DIR, CONTEXT_CACHE)

    print(f"Loading context from {CONTEXT_CACHE}...")
    with open(CONTEXT_CACHE, "r", encoding="utf-8") as f:
        _context_text = f.read()
    print(f"Context loaded: {len(_context_text):,} characters (~{len(_context_text)//4:,} tokens)")

    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            _system_prompt = f.read()
        _system_prompt = _system_prompt.replace("{date}", datetime.now().strftime("%d.%m.%Y"))
        print(f"System prompt loaded from {PROMPT_FILE}")
    else:
        _system_prompt = ""
        print("No PROMPT.md found, running without system prompt.")


def get_config() -> RLMConfig:
    """Create RLM config from environment variables."""
    global _config
    if _config is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it before starting the server."
            )
        _config = RLMConfig(
            primary_model=os.environ.get("RLM_PRIMARY_MODEL", "gpt-5.2"),
            sub_model=os.environ.get("RLM_SUB_MODEL", "gpt-5.2"),
            max_depth=int(os.environ.get("RLM_MAX_DEPTH", "5")),
            max_calls_per_agent=int(os.environ.get("RLM_MAX_CALLS", "25")),
            truncate_len=int(os.environ.get("RLM_TRUNCATE_LEN", "10000")),
            max_total_tokens=int(os.environ.get("RLM_MAX_TOKENS", "1500000")),
            openai_api_key=api_key,
            timeout=int(os.environ.get("RLM_TIMEOUT", "180")),
        )
    return _config


def get_session(session_id: str = None) -> tuple:
    """Get or create a session. Returns (session_id, session_data)."""
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]

    new_id = session_id or str(uuid.uuid4())[:8]
    _sessions[new_id] = {
        "history": [],
        "findings": [],
        "created": datetime.now().isoformat(),
        "total_tokens": 0,
    }
    return new_id, _sessions[new_id]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/info")
def info():
    """Return server configuration and context info."""
    config = get_config()
    return jsonify({
        "context_chars": len(_context_text),
        "context_tokens_estimate": len(_context_text) // 4,
        "system_prompt_chars": len(_system_prompt),
        "primary_model": config.primary_model,
        "sub_model": config.sub_model,
        "max_depth": config.max_depth,
        "max_calls_per_agent": config.max_calls_per_agent,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """Process a chat message through the RLM engine with session memory."""
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    user_message = data["message"]
    session_id = data.get("session_id")
    config = get_config()

    # Get or create session
    session_id, session = get_session(session_id)
    engine = RLMEngine(config)

    try:
        result = engine.run(
            context=_context_text,
            user_prompt=user_message,
            system_prompt=_system_prompt,
            conversation_history=session["history"],
            findings=session["findings"],
        )

        # Update session: add this Q&A to history
        session["history"].append({"role": "user", "content": user_message})
        session["history"].append({"role": "assistant", "content": result["answer"]})

        # Merge new findings into session
        new_findings = result.get("new_findings", [])
        if new_findings:
            session["findings"].extend(new_findings)

        # Track cumulative token usage
        session["total_tokens"] += result["usage"].get("total_tokens", 0)

        # Add session info to response
        result["session_id"] = session_id
        result["session_info"] = {
            "turns": len(session["history"]) // 2,
            "findings_count": len(session["findings"]),
            "cumulative_tokens": session["total_tokens"],
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({
            "answer": f"Error: {str(e)}",
            "usage": {},
            "steps": [],
            "session_id": session_id,
        }), 500


@app.route("/api/session", methods=["POST"])
def new_session():
    """Create a new empty session."""
    session_id, session = get_session()
    return jsonify({
        "session_id": session_id,
        "status": "created",
    })


@app.route("/api/session/<session_id>")
def session_info(session_id):
    """Get session details: history, findings, stats."""
    if session_id not in _sessions:
        return jsonify({"error": "Session not found"}), 404

    session = _sessions[session_id]
    return jsonify({
        "session_id": session_id,
        "turns": len(session["history"]) // 2,
        "findings": session["findings"],
        "total_tokens": session["total_tokens"],
        "created": session["created"],
    })


@app.route("/api/reload-context", methods=["POST"])
def reload_context():
    """Force re-concatenation and reload of context."""
    if os.path.exists(CONTEXT_CACHE):
        os.remove(CONTEXT_CACHE)
    load_context()
    return jsonify({
        "status": "ok",
        "context_chars": len(_context_text),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_context()
    port = int(os.environ.get("RLM_PORT", "5055"))
    print(f"\nStarting RLM Chat Server on http://localhost:{port}")
    print(f"Model: {get_config().primary_model}")
    print(f"Open http://localhost:{port} in your browser\n")
    app.run(host="0.0.0.0", port=port, debug=False)
