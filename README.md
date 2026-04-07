# NBA Career Simulator

**Live at [nbacareersim.win](https://nbacareersim.win)**

A full-stack NBA career simulator that projects what a player's career would look like without injuries or other disruptions, using Monte Carlo simulation and machine learning.

Search any NBA player, view their full career stats, and simulate their career from any point (year 3+) to see what they could have been. Tmac 💔

<img width="1929" height="1617" alt="Screenshot 2026-04-07 044314" src="https://github.com/user-attachments/assets/75520766-2e13-483b-b437-5cb2f176abba" />

## What it does

- Search 5,000+ NBA players by name
- View full career table (age, per-game, totals, and advanced metrics)
- Simulate from any season starting no earlier than year 3
- Generate realistic projections that ignore injury history and real-world outcomes
- Visualize career trajectories with interactive charts (historical vs. simulated)
- Aggregated P10/P50/P90 projections across 250 Monte Carlo paths

## Architecture

**Frontend:** React, Vite, Recharts

**Backend:** Python, FastAPI, scikit-learn, SQLite

**Data:** nba_api, automated ingestion pipeline (~500 active players, weekly refresh)

**Infrastructure:** Docker, Nginx, Vultr VPS, Cloudflare, Umami

## Local Development

```bash
git clone https://github.com/geos1l/NBA-Career-Simulator.git
cd NBA-Career-Simulator
```

**Backend:**
```bash
python -m venv venv
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Data ingestion:**
```bash
python scripts/ingest_data.py                   # All 5,000+ players (2-4 hours)
python scripts/ingest_data.py --active-only     # Active players only (~10 min)
```

- Frontend: `http://localhost:5173`
- API docs: `http://localhost:8000/docs`
