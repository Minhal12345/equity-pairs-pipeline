#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Fixing Python SSL certificates ==="
# Install certifi and point pip at proper certs
pip3 install --upgrade certifi pip 2>/dev/null || true

# Fix macOS Python SSL (run the Install Certificates script if it exists)
CERT_SCRIPT="$(python3 -c 'import sys, os; print(os.path.join(os.path.dirname(sys.executable), "..", "Resources", "Install Certificates.command"))' 2>/dev/null)"
if [ -f "$CERT_SCRIPT" ]; then
    bash "$CERT_SCRIPT" 2>/dev/null || true
fi

echo ""
echo "=== Installing dependencies ==="
pip3 install -r requirements.txt

echo ""
echo "=== Running pipeline (yfinance only) ==="
python3 main.py --skip-av

echo ""
echo "=== Committing results ==="
git add -A
git commit -m "Add pipeline results: charts, backtest data, and README

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01K6sFmCQbS5N6Yb3HyV5Nbs"

echo ""
echo "=== Pushing to GitHub ==="
git push -u origin main

echo ""
echo "=== Done! ==="
echo "Repo: https://github.com/Minhal12345/equity-pairs-pipeline"
