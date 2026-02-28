#!/bin/bash
# RLM Chat PoC - Start Script
# Usage: ./start.sh

set -e
cd "$(dirname "$0")"

# Source environment if available
[ -f ../.env ] && source ../.env
[ -f ../.envrc ] && source ../.envrc
[ -f .env ] && source .env

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "Run: export OPENAI_API_KEY=sk-..."
    exit 1
fi

# Install dependencies if needed
python3 -m pip install -q flask flask-cors openai 2>/dev/null || python3 -m pip install -q --break-system-packages flask flask-cors openai

# Optional: configure model (defaults to gpt-5.2)
export RLM_PRIMARY_MODEL="${RLM_PRIMARY_MODEL:-gpt-5.2}"
export RLM_SUB_MODEL="${RLM_SUB_MODEL:-gpt-5.2}"
export RLM_MAX_DEPTH="${RLM_MAX_DEPTH:-2}"
export RLM_MAX_CALLS="${RLM_MAX_CALLS:-25}"
export RLM_MAX_TOKENS="${RLM_MAX_TOKENS:-1500000}"
export RLM_MAX_CONCURRENT="${RLM_MAX_CONCURRENT:-12}"
export RLM_MAX_LLM_CALLS="${RLM_MAX_LLM_CALLS:-50}"
export RLM_SUB_AGENT_CALLS="${RLM_SUB_AGENT_CALLS:-8}"
export RLM_ROOT_RESERVED="${RLM_ROOT_RESERVED:-5}"
export RLM_TRUNCATE_LEN="${RLM_TRUNCATE_LEN:-10000}"
export RLM_TIMEOUT="${RLM_TIMEOUT:-180}"

echo ""
echo "=== RLM Chat PoC ==="
echo "Primary Model: $RLM_PRIMARY_MODEL"
echo "Sub Model:     $RLM_SUB_MODEL"
echo "Max Depth:     $RLM_MAX_DEPTH"
echo "Max Concurrent:$RLM_MAX_CONCURRENT"
echo "Max Calls:     $RLM_MAX_CALLS"
echo ""

python3 server.py
