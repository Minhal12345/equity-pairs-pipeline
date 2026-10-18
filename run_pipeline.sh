#!/bin/bash
# Run the full pipeline: re-ingest data from Yahoo Finance, then process everything.
# Run this from your own terminal (not through Claude's device shell).
set -e

cd ~/projects/equity-pairs-pipeline

echo "=== Installing dependencies ==="
pip install -r requirements.txt --break-system-packages 2>/dev/null || pip install -r requirements.txt

echo ""
echo "=== Running full pipeline ==="
python main.py --skip-av

echo ""
echo "=== Running tests ==="
python -m pytest tests/ -v

echo ""
echo "=== Done! ==="
