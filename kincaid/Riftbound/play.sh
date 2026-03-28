#!/usr/bin/env bash
# One-line play: ./play.sh
# Starts the Riftbound simulator backend and opens it in the browser.
# Frontend is pre-built and served by the backend.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
PORT="${PORT:-8000}"

# Determine python command
PYTHON_CMD="python"
PIP_CMD="pip"
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
fi

# Check Python deps
if ! $PYTHON_CMD -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "Installing Python dependencies..."
    $PIP_CMD install -r "$BACKEND_DIR/requirements.txt"
fi

# Rebuild frontend if dist is missing
if [ ! -f "$FRONTEND_DIR/dist/index.html" ]; then
    echo "Building frontend..."
    cd "$FRONTEND_DIR"
    npm install
    npm run build
    cd "$SCRIPT_DIR"
fi

echo ""
echo "==================================="
echo "  Riftbound Simulator"
echo "  http://localhost:$PORT"
echo "==================================="
echo ""
echo "Open two browser tabs to play."
echo "Press Ctrl+C to stop."
echo ""

# Open browser (works on Windows/Mac/Linux)
if command -v start &>/dev/null; then
    start "http://localhost:$PORT" 2>/dev/null &
elif command -v open &>/dev/null; then
    open "http://localhost:$PORT" 2>/dev/null &
elif command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:$PORT" 2>/dev/null &
fi

# Start server
cd "$BACKEND_DIR"
exec $PYTHON_CMD -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
