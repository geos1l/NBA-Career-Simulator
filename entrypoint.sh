#!/bin/bash
set -e

# Auto-ingest on first run if database is empty or missing
if [ ! -f /app/data/nba.db ] || [ "$(python -c "
import sqlite3, sys
try:
    c = sqlite3.connect('/app/data/nba.db')
    n = c.execute('SELECT COUNT(*) FROM season_stats').fetchone()[0]
    print(n)
except:
    print(0)
")" = "0" ]; then
    echo "==> No data found. Running initial ingestion (active players only)..."
    python scripts/ingest_data.py --active-only
    echo "==> Initial ingestion complete."
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
