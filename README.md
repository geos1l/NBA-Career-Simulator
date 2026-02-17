# NBA Player Simulator

A full-stack NBA player career simulator with:

- React UI
- FastAPI backend
<<<<<<< Updated upstream
- Regression machine learning model
- Dynamic, injury-free career projection with retirement logic
- Model retraining and versioned artifacts

It's meant to simulate players careers starting from a certain point using only their previous seasons data, to see what their career would look like (if they didn't injured or xyz). Tmac 💔

## What it does

- Search NBA players by name
- View full career table (age, per-game, totals, and advanced metrics)
- Simulate from any season **starting no earlier than year 3**
- Generate realistic projections that ignore injury history and real-world outcomes
- Show simulation output in a dedicated side panel
- Project retirement age dynamically from multiple factors (age, role, production, longevity)

## Tech Stack


### Backend
- Python
- FastAPI
- nba_api
- pandas / numpy
- scikit-learn
- joblib


### Frontend
- React
- Vite
- Vanilla CSS


## Architecture


### Backend modules
- `app/services/external.py`
  - Player search and NBA data fetching
  - Career stats with per-game, totals, and derived advanced metrics
- `app/services/trainer.py`
  - League-wide training dataset builder (season-to-next-season mapping)
  - Multi-target `HistGradientBoostingRegressor` model training
- `app/services/model_store.py`
  - Versioned model artifact storage in `artifacts/models/simulator/`
  - `latest.json` pointer for active model
- `app/services/sim.py`
  - Monte Carlo simulation engine
  - Realistic age-curve adjustments
  - Dynamic retirement probability model
- `app/main.py`
  - API routes
  - model lifecycle (`startup` load/train + retrain endpoint)


### Frontend
- `frontend/src/App.jsx`
  - search, player selection, career table, simulation controls
  - right-side simulation results panel
- `frontend/src/api.js`
  - backend API client


## API Endpoints


- `GET /api/health`
- `GET /api/models/current`
- `POST /api/models/retrain`
- `GET /api/players/search?query={name}`
- `GET /api/player?name={name}`
- `GET /api/players/{player_id}/career`
- `POST /api/simulate`


### Simulation request body


```json
{
  "player_id": 2544,
  "start_season": 2010,
  "simulations": 250,
  "realism_profile": "realistic"
}
```


## Running Locally


### 1) Backend


```bash
python -m venv venv
```


Windows PowerShell:


```powershell
.\venv\Scripts\Activate.ps1
```


Install backend deps:


```bash
pip install -r requirements.txt
```


Run API:


```bash
uvicorn app.main:app --reload
```


### 2) Frontend


```bash
cd frontend
npm install
npm run dev
```


Open:


- Frontend: `http://localhost:5173`
- API docs: `http://localhost:8000/docs`


### Optional production build


```bash
cd frontend
npm run build
```


If `frontend/dist` exists, backend `/` will serve the built app.


## Notes


- First backend startup may trigger model training if no model exists yet.
- Trained artifacts are saved under:
  - `artifacts/models/simulator/<version>/model.joblib`
  - `artifacts/models/simulator/<version>/metadata.json`
  - `artifacts/models/simulator/latest.json`
- Retraining does not overwrite old versions; it creates a new version and updates `latest.json`.
