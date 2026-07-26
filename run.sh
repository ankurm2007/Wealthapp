#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

# Free a stuck previous instance on 8501 (ignore errors).
lsof -tiTCP:8501 -sTCP:LISTEN 2>/dev/null | xargs kill -9 2>/dev/null || true

echo "Starting dashboard at http://localhost:8501"
echo "First load can take 1–3 minutes while Python packages initialize — wait for 'You can now view'."
echo ""

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

exec .venv/bin/streamlit run app.py --server.port 8501 --server.headless true
