#!/bin/bash
# Launch DPF Test Status Dashboard
# Opens a live visual counter in your browser showing slow test progress.
#
# Usage:
#   ./tools/launch_dashboard.sh          # slow tests only (default)
#   ./tools/launch_dashboard.sh --all    # ALL tests
#   ./tools/launch_dashboard.sh --fast   # non-slow tests only

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Find the Python that has dpf installed
PYTHON="${DPF_PYTHON:-/opt/homebrew/opt/python@3.11/bin/python3.11}"
if ! "$PYTHON" -c "import dpf" 2>/dev/null; then
    # Fallback: search common locations
    for candidate in /opt/homebrew/opt/python@3.11/bin/python3.11 /opt/homebrew/bin/python3.11 python3.11 python3; do
        if "$candidate" -c "import dpf" 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    done
fi
PORT=8877

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
DIM='\033[0;90m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   🔬 DPF Test Status Dashboard       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════╝${NC}"
echo ""

# Kill any stale processes
pkill -f "test_status_runner.py" 2>/dev/null || true
pkill -f "http.server.*${PORT}" 2>/dev/null || true
sleep 0.5

# Clean old status
rm -f "$SCRIPT_DIR/test_status.json" "$SCRIPT_DIR/test_status.tmp"

# Start local HTTP server for the dashboard (serves from tools/ directory)
echo -e "${DIM}Starting dashboard server on port ${PORT}...${NC}"
cd "$SCRIPT_DIR"
$PYTHON -m http.server $PORT --bind 127.0.0.1 > /dev/null 2>&1 &
SERVER_PID=$!

# Wait for server to start
sleep 1

# Open dashboard in browser
DASHBOARD_URL="http://127.0.0.1:${PORT}/test_dashboard.html"
echo -e "${GREEN}Dashboard: ${DASHBOARD_URL}${NC}"
open "$DASHBOARD_URL" 2>/dev/null || echo "Open $DASHBOARD_URL in your browser"

# Start test runner (passes through args like --all, --fast)
echo -e "${DIM}Starting test runner...${NC}"
echo ""

cd "$PROJECT_ROOT"
$PYTHON "$SCRIPT_DIR/test_status_runner.py" "$@"
EXIT_CODE=$?

# Cleanup
echo ""
echo -e "${DIM}Stopping dashboard server...${NC}"
kill $SERVER_PID 2>/dev/null || true

exit $EXIT_CODE
