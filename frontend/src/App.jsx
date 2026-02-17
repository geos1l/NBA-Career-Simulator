import { useEffect, useMemo, useRef, useState } from "react";
import { getCareer, getModelInfo, searchPlayers, simulateCareer } from "./api";
import {
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

const metricHeaders = [
  { key: "ppg", label: "PPG" },
  { key: "rpg", label: "RPG" },
  { key: "apg", label: "APG" },
  { key: "spg", label: "SPG" },
  { key: "bpg", label: "BPG" },
  { key: "tpg", label: "TPG" },
  { key: "mpg", label: "MPG" },
  { key: "fg_pct", label: "FG%" },
  { key: "fg3_pct", label: "3PT%" },
  { key: "ft_pct", label: "FT%" }
];

const featuredPlayers = [
  { id: 2544, full_name: "LeBron James", is_active: true },
  { id: 201939, full_name: "Stephen Curry", is_active: true },
  { id: 203954, full_name: "Joel Embiid", is_active: true },
  { id: 202681, full_name: "Kyrie Irving", is_active: true },
  { id: 202691, full_name: "Klay Thompson", is_active: true },
  { id: 1629029, full_name: "Luka Doncic", is_active: true },
  { id: 1628369, full_name: "Jayson Tatum", is_active: true },
  { id: 203507, full_name: "Giannis Antetokounmpo", is_active: true },
  { id: 203081, full_name: "Damian Lillard", is_active: true },
  { id: 201565, full_name: "Derrick Rose", is_active: false }
];

const chartMetricOptions = [
  { key: "ppg", label: "Points Per Game" },
  { key: "apg", label: "Assists Per Game" },
  { key: "rpg", label: "Rebounds Per Game" },
  { key: "mpg", label: "Minutes Per Game" },
  { key: "fg_pct", label: "FG%" },
  { key: "ts_pct", label: "TS%" },
  { key: "pts_tot", label: "Total Points" }
];

function pct(value) {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

function num(value) {
  return typeof value === "number" ? value.toFixed(1) : "-";
}

function App() {
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState(featuredPlayers);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selected, setSelected] = useState(null);
  const [career, setCareer] = useState(null);
  const [simStartSeason, setSimStartSeason] = useState("");
  const [simResult, setSimResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [busyLabel, setBusyLabel] = useState("");
  const [simProgress, setSimProgress] = useState(0);
  const [modelInfo, setModelInfo] = useState(null);
  const [selectedChartMetric, setSelectedChartMetric] = useState("ppg");
  const searchRef = useRef(null);
  const progressTimerRef = useRef(null);

  const chartData = useMemo(() => {
    if (!career) {
      return [];
    }

    if (!simResult) {
      return career.seasons.map((season) => ({
        seasonKey: season.season_start,
        label: season.season_label,
        historical: getMetricValueFromSeason(season, selectedChartMetric),
        reality: null,
        projected: null
      }));
    }

    const startYear = Number(simResult.start_season);
    const byYear = new Map();

    career.seasons.forEach((season) => {
      const year = Number(season.season_start);
      byYear.set(year, {
        seasonKey: year,
        label: season.season_label,
        historical: year <= startYear ? getMetricValueFromSeason(season, selectedChartMetric) : null,
        reality: year >= startYear ? getMetricValueFromSeason(season, selectedChartMetric) : null,
        projected: null
      });
    });

    simResult.aggregated_projection.forEach((row) => {
      const year = startYear + row.season_offset;
      const existing = byYear.get(year);
      if (existing) {
        existing.projected = getMetricValueFromProjection(row, selectedChartMetric);
      } else {
        byYear.set(year, {
          seasonKey: year,
          label: `${year}-${String((year + 1) % 100).padStart(2, "0")}`,
          historical: null,
          reality: null,
          projected: getMetricValueFromProjection(row, selectedChartMetric)
        });
      }
    });

    return Array.from(byYear.values()).sort((a, b) => a.seasonKey - b.seasonKey);
  }, [career, selectedChartMetric, simResult]);

  useEffect(() => {
    return () => {
      if (progressTimerRef.current) {
        clearInterval(progressTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchRef.current && !searchRef.current.contains(event.target)) {
        setShowDropdown(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    const trimmed = query.trim();
    if (!trimmed) {
      setMatches(featuredPlayers);
      return undefined;
    }

    const timeout = setTimeout(async () => {
      try {
        const data = await searchPlayers(trimmed);
        setMatches(data);
      } catch {
        setMatches([]);
      }
    }, 220);

    return () => clearTimeout(timeout);
  }, [query]);

  function startProgress(labelText) {
    setBusy(true);
    setBusyLabel(labelText);
    setSimProgress(7);
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
    }
    progressTimerRef.current = setInterval(() => {
      setSimProgress((prev) => (prev >= 92 ? prev : prev + 3));
    }, 300);
  }

  function completeProgress() {
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
    }
    setSimProgress(100);
    setTimeout(() => {
      setBusy(false);
      setBusyLabel("");
      setSimProgress(0);
    }, 220);
  }

  async function handleSelect(player) {
    setError("");
    startProgress("Loading career...");
    setSelected(player);
    setSimResult(null);
    setShowDropdown(false);
    try {
      const [careerData, modelData] = await Promise.all([getCareer(player.id), getModelInfo()]);
      setCareer(careerData);
      setModelInfo(modelData);
      const defaultStart = careerData.seasons[2]?.season_start;
      setSimStartSeason(defaultStart ? String(defaultStart) : "");
    } catch (err) {
      setError(err.message);
    } finally {
      completeProgress();
    }
  }

  async function handleSearch(e) {
    e.preventDefault();
    const top = matches[0];
    if (top) {
      await handleSelect(top);
      return;
    }
    setShowDropdown(true);
  }

  async function handleSimulate() {
    if (!selected || !simStartSeason) {
      return;
    }
    startProgress("Simulating career...");
    setError("");
    try {
      const result = await simulateCareer({
        player_id: selected.id,
        start_season: Number(simStartSeason),
        simulations: 250,
        realism_profile: "arcade"
      });
      setSimResult(result);
    } catch (err) {
      setError(err.message);
    } finally {
      completeProgress();
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>NBA Player Simulator</h1>
        <p>Injury-free projection engine with dynamic retirement and arcade-style variance.</p>
      </header>

      <section className="searchBar" ref={searchRef}>
        <form onSubmit={handleSearch}>
          <input
            value={query}
            onFocus={() => setShowDropdown(true)}
            onChange={(e) => {
              setQuery(e.target.value);
              setShowDropdown(true);
            }}
            placeholder="Search player (e.g., LeBron James)"
          />
          <button type="submit" disabled={busy}>
            Open
          </button>
        </form>
        {showDropdown && (
          <div className="searchDropdown">
            <div className="dropdownTitle">{query.trim() ? "Suggestions" : "Featured Players"}</div>
            {matches.length === 0 ? (
              <div className="dropdownEmpty">No players found.</div>
            ) : (
              matches.map((p) => (
                <button key={p.id} className="dropdownItem" onClick={() => handleSelect(p)}>
                  <span>{p.full_name}</span>
                  <small>{p.is_active ? "Active" : "Inactive"}</small>
                </button>
              ))
            )}
          </div>
        )}
      </section>

      {!career && (
        <section className="featuredPanel">
          <h3>Quick Start</h3>
          <p>Pick any featured player or type to search with autocomplete.</p>
          <div className="featuredGrid">
            {featuredPlayers.map((player) => (
              <button key={player.id} className="featuredBtn" onClick={() => handleSelect(player)}>
                {player.full_name}
              </button>
            ))}
          </div>
        </section>
      )}

      {busy && (
        <section className="loadingWrap">
          <div className="loadingText">{busyLabel || "Working..."}</div>
          <div className="loadingBar">
            <div className="loadingBarFill" style={{ width: `${simProgress}%` }} />
          </div>
        </section>
      )}

      {error && <div className="error">{error}</div>}

      {career && (
        <section className="workspace">
          <div className="leftPanel">
            <div className="panelTitle">
              <h2>
                {career.name} - {career.position || "N/A"}
              </h2>
              <span>{career.seasons_played} seasons</span>
            </div>

            <div className="simControls">
              <label>
                Selected start season:{" "}
                <strong>{simStartSeason || "Choose a season row below (year 3+ only)"}</strong>
              </label>
              <button onClick={handleSimulate} disabled={busy || !simStartSeason}>
                Simulate
              </button>
              {modelInfo && <small>Model: {modelInfo.version}</small>}
            </div>

            <div className="tableWrap">
              <table>
                <thead>
                  <tr>
                    <th>Season</th>
                    <th>Age</th>
                    <th>GP</th>
                    {metricHeaders.map((h) => (
                      <th key={h.key}>{h.label}</th>
                    ))}
                    <th>PTS Tot</th>
                    <th>REB Tot</th>
                    <th>AST Tot</th>
                    <th>TS%</th>
                    <th>eFG%</th>
                  </tr>
                </thead>
                <tbody>
                  {career.seasons.map((s, index) => {
                    const canStart = index >= 2;
                    const isSelected = Number(simStartSeason) === Number(s.season_start);
                    return (
                      <tr
                        key={s.season_label}
                        className={`${canStart ? "clickableRow" : "disabledRow"} ${isSelected ? "selectedRow" : ""}`}
                        onClick={() => {
                          if (canStart) {
                            setSimStartSeason(String(s.season_start));
                          }
                        }}
                        title={canStart ? "Click to start simulation from this season" : "Need at least 3 seasons"}
                      >
                      <td className="seasonCell">
                        {s.season_label}
                        {!canStart && <small className="seasonHint">Locked</small>}
                      </td>
                      <td>{s.age ?? "-"}</td>
                      <td>{s.gp}</td>
                      <td>{num(s.per_game.ppg)}</td>
                      <td>{num(s.per_game.rpg)}</td>
                      <td>{num(s.per_game.apg)}</td>
                      <td>{num(s.per_game.spg)}</td>
                      <td>{num(s.per_game.bpg)}</td>
                      <td>{num(s.per_game.tpg)}</td>
                      <td>{num(s.per_game.mpg)}</td>
                      <td>{pct(s.per_game.fg_pct)}</td>
                      <td>{pct(s.per_game.fg3_pct)}</td>
                      <td>{pct(s.per_game.ft_pct)}</td>
                      <td>{num(s.totals.pts)}</td>
                      <td>{num(s.totals.reb)}</td>
                      <td>{num(s.totals.ast)}</td>
                      <td>{pct(s.advanced.ts_pct)}</td>
                      <td>{pct(s.advanced.efg_pct)}</td>
                    </tr>
                  );
                })}
                </tbody>
              </table>
            </div>

            <div className="chartPanel">
              <div className="chartHeader">
                <h3>Career + Simulation Graph</h3>
                <select value={selectedChartMetric} onChange={(e) => setSelectedChartMetric(e.target.value)}>
                  {chartMetricOptions.map((option) => (
                    <option key={option.key} value={option.key}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="chartCanvas">
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={chartData}>
                    <XAxis dataKey="label" interval="preserveStartEnd" minTickGap={24} />
                    <YAxis domain={["auto", "auto"]} />
                    <Tooltip />
                    <Line type="monotone" dataKey="historical" name="History (Pre-Sim)" stroke="#2e69ff" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="reality" name="Reality (Actual)" stroke="#34a853" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="projected" name="Simulation (P50)" stroke="#ff6a3d" strokeWidth={2} dot={false} />
                    {simResult && (
                      <ReferenceLine
                        x={`${simResult.start_season}-${String((simResult.start_season + 1) % 100).padStart(2, "0")}`}
                        stroke="#111"
                        strokeDasharray="4 4"
                        label="Simulation Start"
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <aside className="rightPanel">
            <h3>Simulation Panel</h3>
            {!simResult && <p>Run a simulation to open projected seasons here.</p>}
            {simResult && (
              <div className="simResults">
                <p>
                  Start: {simResult.start_season} | Projected retirement age:{" "}
                  {simResult.projected_retirement_age}
                </p>
                <p>Paths: {simResult.simulations}</p>

                <h4>Aggregated projection (P10/P50/P90)</h4>
                <table>
                  <thead>
                    <tr>
                      <th>Year+</th>
                      <th>Age</th>
                      <th>PPG</th>
                      <th>RPG</th>
                      <th>APG</th>
                      <th>MPG</th>
                    </tr>
                  </thead>
                  <tbody>
                    {simResult.aggregated_projection.map((row) => (
                      <tr key={row.season_offset}>
                        <td>{row.season_offset}</td>
                        <td>{row.age_median}</td>
                        <td>
                          {row.metrics.ppg.p10}/{row.metrics.ppg.p50}/{row.metrics.ppg.p90}
                        </td>
                        <td>
                          {row.metrics.rpg.p10}/{row.metrics.rpg.p50}/{row.metrics.rpg.p90}
                        </td>
                        <td>
                          {row.metrics.apg.p10}/{row.metrics.apg.p50}/{row.metrics.apg.p90}
                        </td>
                        <td>
                          {row.metrics.mpg.p10}/{row.metrics.mpg.p50}/{row.metrics.mpg.p90}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                <h4>Sample trajectory</h4>
                <table>
                  <thead>
                    <tr>
                      <th>Year+</th>
                      <th>Age</th>
                      <th>GP</th>
                      <th>PPG</th>
                      <th>RPG</th>
                      <th>APG</th>
                      <th>PTS Tot</th>
                      <th>TS%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {simResult.paths_sample.map((row) => (
                      <tr key={row.season_offset}>
                        <td>{row.season_offset}</td>
                        <td>{row.age}</td>
                        <td>{row.gp}</td>
                        <td>{num(row.per_game.ppg)}</td>
                        <td>{num(row.per_game.rpg)}</td>
                        <td>{num(row.per_game.apg)}</td>
                        <td>{num(row.totals.pts)}</td>
                        <td>{pct(row.advanced.ts_pct)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </aside>
        </section>
      )}
    </div>
  );
}

export default App;

function getMetricValueFromSeason(season, metric) {
  if (metric === "ts_pct") {
    return season.advanced.ts_pct;
  }
  if (metric === "pts_tot") {
    return season.totals.pts;
  }
  return season.per_game[metric];
}

function getMetricValueFromProjection(row, metric) {
  if (metric === "ts_pct") {
    return row.sample_projection.advanced.ts_pct;
  }
  if (metric === "pts_tot") {
    return row.sample_projection.totals.pts;
  }
  const metricMap = {
    ppg: "ppg",
    apg: "apg",
    rpg: "rpg",
    mpg: "mpg",
    fg_pct: "fg_pct"
  };
  const key = metricMap[metric];
  return key ? row.sample_projection.per_game[key] : row.metrics.ppg.p50;
}
