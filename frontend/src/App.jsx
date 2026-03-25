import { useEffect, useMemo, useRef, useState } from "react";
import { getCareer, getModelInfo, searchPlayers, simulateCareerStream } from "./api";
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


const chartMetricOptions = [
  { key: "ppg", label: "Points Per Game" },
  { key: "apg", label: "Assists Per Game" },
  { key: "rpg", label: "Rebounds Per Game" },
  { key: "spg", label: "Steals Per Game" },
  { key: "bpg", label: "Blocks Per Game" },
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
  const [matches, setMatches] = useState([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selected, setSelected] = useState(null);
  const [career, setCareer] = useState(null);
  const [simStartSeason, setSimStartSeason] = useState("");
  const [simResult, setSimResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [busyLabel, setBusyLabel] = useState("");
  const [modelInfo, setModelInfo] = useState(null);
  const [selectedChartMetric, setSelectedChartMetric] = useState("ppg");
  const [tableView, setTableView] = useState("real"); // "real" or "simulated"
  /** null = indeterminate bar; number = real % from server (simulate stream). */
  const [loadProgress, setLoadProgress] = useState(null);
  const [loadSubtext, setLoadSubtext] = useState("");
  const searchRef = useRef(null);

  const chartData = useMemo(() => {
    if (!career) {
      return [];
    }

    const firstYear = Number(career.seasons[0]?.season_start ?? 0);

    if (!simResult) {
      return career.seasons.map((season, i) => ({
        seasonKey: season.season_start,
        label: season.season_label,
        yearInLeague: `Year ${i + 1}`,
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
        yearInLeague: `Year ${year - firstYear + 1}`,
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
          yearInLeague: `Year ${year - firstYear + 1}`,
          historical: null,
          reality: null,
          projected: getMetricValueFromProjection(row, selectedChartMetric)
        });
      }
    });

    const sorted = Array.from(byYear.values()).sort((a, b) => a.seasonKey - b.seasonKey);
    const pivot = sorted.find((d) => d.seasonKey === startYear);
    if (pivot && pivot.historical != null) {
      pivot.projected = pivot.historical;
    }

    return sorted;
  }, [career, selectedChartMetric, simResult]);

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
      setMatches([]);
      return undefined;
    }

    const timeout = setTimeout(async () => {
      try {
        const data = await searchPlayers(trimmed);
        setMatches(data);
      } catch {
        setMatches([]);
      }
    }, 100);

    return () => clearTimeout(timeout);
  }, [query]);

  function startProgress(labelText, mode = "indeterminate") {
    setBusy(true);
    setBusyLabel(labelText);
    setLoadSubtext("");
    setLoadProgress(mode === "determinate" ? 0 : null);
  }

  function completeProgress() {
    setBusy(false);
    setBusyLabel("");
    setLoadProgress(null);
    setLoadSubtext("");
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
    startProgress("Simulating career…", "determinate");
    setError("");
    try {
      const result = await simulateCareerStream(
        {
          player_id: selected.id,
          start_season: Number(simStartSeason),
          simulations: 250
        },
        (ev) => {
          if (ev.pct != null) setLoadProgress(ev.pct);
          let sub = "";
          if (ev.phase === "paths" && ev.done > 0 && ev.total != null) {
            sub = `${ev.done}/${ev.total} paths`;
            if (ev.eta_seconds != null) sub += ` · ~${ev.eta_seconds}s left`;
          } else if (ev.message) {
            sub = ev.message;
          }
          if (sub) setLoadSubtext(sub);
        }
      );
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
        <h1 style={{ cursor: "pointer" }} onClick={() => { setSelected(null); setCareer(null); setSimResult(null); setQuery(""); setError(""); }}>NBA Career Simulator</h1>
        <p>Monte Carlo projection engine with dynamic retirement modeling.</p>
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
            <div className="dropdownTitle">{query.trim() ? "Suggestions" : "Start typing to search"}</div>
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


      {busy && (
        <section className="loadingWrap" aria-busy="true" aria-live="polite">
          <div className="loadingText">{busyLabel || "Working..."}</div>
          {loadProgress === null ? (
            <>
              <div
                className="loadingBar loadingBarIndeterminate"
                role="status"
                aria-valuetext={busyLabel || "Loading"}
              >
                <div className="loadingBarShimmer" />
              </div>
              <div className="loadingHint">
                No exact percent (waiting on the server / NBA API). Use Simulate for a real progress bar.
              </div>
            </>
          ) : (
            <>
              <div
                className="loadingBar"
                role="progressbar"
                aria-valuenow={loadProgress}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuetext={`${loadProgress}%`}
              >
                <div className="loadingBarFill" style={{ width: `${loadProgress}%` }} />
              </div>
              <div className="loadingPct">{loadProgress}%</div>
              {loadSubtext ? <div className="loadingSubtext">{loadSubtext}</div> : null}
              <div className="loadingHint">
                Bar tracks completed simulation paths (and a short wrap-up). ETA is an estimate from recent speed.
              </div>
            </>
          )}
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

            {simResult && (
              <div className="tableToggle">
                <button className={tableView === "real" ? "active" : ""} onClick={() => setTableView("real")}>Real Career</button>
                <button className={tableView === "simulated" ? "active" : ""} onClick={() => setTableView("simulated")}>Simulated Career</button>
              </div>
            )}

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
                  {(tableView === "real" || !simResult) &&
                    career.seasons.map((s, index) => {
                      const canStart = index >= 2;
                      const isSelected = Number(simStartSeason) === Number(s.season_start);
                      const isSimStart = simResult && Number(s.season_start) === Number(simResult.start_season);
                      return (
                        <tr
                          key={s.season_label}
                          className={`${canStart ? "clickableRow" : "disabledRow"} ${isSelected ? "selectedRow" : ""} ${isSimStart ? "simStartRow" : ""}`}
                          onClick={() => {
                            if (canStart) {
                              setSimStartSeason(String(s.season_start));
                            }
                          }}
                          title={canStart ? "Click to start simulation from this season" : "Need at least 3 seasons"}
                        >
                        <td className="seasonCell">
                          {isSimStart && <span className="simStartMarker">SIM START</span>}
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
                  {tableView === "simulated" && simResult && (() => {
                    // Pre-sim real seasons + projected seasons
                    const startYear = Number(simResult.start_season);
                    const preSim = career.seasons.filter((s) => Number(s.season_start) <= startYear);
                    return (
                      <>
                        {preSim.map((s) => {
                          const isSimStart = Number(s.season_start) === startYear;
                          return (
                            <tr key={s.season_label} className={isSimStart ? "simStartRow" : ""}>
                              <td className="seasonCell">
                                {isSimStart && <span className="simStartMarker">SIM START</span>}
                                {s.season_label}
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
                        {simResult.paths_sample.map((s) => {
                          const projYear = startYear + s.season_offset;
                          const label = `${projYear}-${String((projYear + 1) % 100).padStart(2, "0")}`;
                          return (
                            <tr key={`proj-${s.season_offset}`} className="projectedRow">
                              <td className="seasonCell">{label}</td>
                              <td>{s.age}</td>
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
                      </>
                    );
                  })()}
                </tbody>
                <tfoot>
                  {(() => {
                    const allSeasons = (tableView === "simulated" && simResult)
                      ? [
                          ...career.seasons.filter((s) => Number(s.season_start) <= Number(simResult.start_season)),
                          ...simResult.paths_sample,
                        ]
                      : [...career.seasons];
                    const n = allSeasons.length;
                    if (n === 0) return null;
                    const avg = (arr, fn) => arr.reduce((s, x) => s + fn(x), 0) / arr.length;
                    const sum = (arr, fn) => arr.reduce((s, x) => s + fn(x), 0);
                    const label = (tableView === "simulated" && simResult) ? "Simulated Career Totals / Avg" : "Career Totals / Avg";
                    return (
                      <tr className="careerAvgRow">
                        <td className="seasonCell">{label}</td>
                        <td>-</td>
                        <td>{Math.round(avg(allSeasons, (s) => s.gp))}</td>
                        <td>{num(avg(allSeasons, (s) => s.per_game.ppg))}</td>
                        <td>{num(avg(allSeasons, (s) => s.per_game.rpg))}</td>
                        <td>{num(avg(allSeasons, (s) => s.per_game.apg))}</td>
                        <td>{num(avg(allSeasons, (s) => s.per_game.spg))}</td>
                        <td>{num(avg(allSeasons, (s) => s.per_game.bpg))}</td>
                        <td>{num(avg(allSeasons, (s) => s.per_game.tpg))}</td>
                        <td>{num(avg(allSeasons, (s) => s.per_game.mpg))}</td>
                        <td>{pct(avg(allSeasons, (s) => s.per_game.fg_pct))}</td>
                        <td>{pct(avg(allSeasons, (s) => s.per_game.fg3_pct))}</td>
                        <td>{pct(avg(allSeasons, (s) => s.per_game.ft_pct))}</td>
                        <td>{Math.round(sum(allSeasons, (s) => s.totals.pts)).toLocaleString()}</td>
                        <td>{Math.round(sum(allSeasons, (s) => s.totals.reb)).toLocaleString()}</td>
                        <td>{Math.round(sum(allSeasons, (s) => s.totals.ast)).toLocaleString()}</td>
                        <td>{pct(avg(allSeasons, (s) => s.advanced.ts_pct))}</td>
                        <td>{pct(avg(allSeasons, (s) => s.advanced.efg_pct))}</td>
                      </tr>
                    );
                  })()}
                </tfoot>
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
                    <Tooltip
                      content={({ payload, active }) => {
                        if (!active || !payload?.length) return null;
                        const d = payload[0]?.payload;
                        if (!d) return null;
                        // Deduplicate: if historical and projected are the same value (pivot point), just show historical
                        const seen = new Map();
                        for (const entry of payload) {
                          if (entry.value == null) continue;
                          const rounded = Number(entry.value).toFixed(3);
                          if (!seen.has(rounded)) {
                            seen.set(rounded, entry);
                          }
                        }
                        return (
                          <div style={{ background: "#fff", border: "1px solid #ccc", padding: "8px 12px", borderRadius: 4, fontSize: 12 }}>
                            <div style={{ fontWeight: 600, marginBottom: 4 }}>{d.yearInLeague} ({d.label})</div>
                            {[...seen.values()].map((entry) => (
                              <div key={entry.dataKey} style={{ color: entry.color }}>
                                {entry.name}: {typeof entry.value === "number" ? entry.value.toFixed(1) : entry.value}
                              </div>
                            ))}
                          </div>
                        );
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="historical"
                      name="History (pre-sim)"
                      stroke="#1e3a8f"
                      strokeWidth={2.5}
                      dot={false}
                      connectNulls
                    />
                    <Line
                      type="monotone"
                      dataKey="reality"
                      name="Actual (post-start)"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                      connectNulls
                    />
                    <Line
                      type="monotone"
                      dataKey="projected"
                      name="Simulation (P50)"
                      stroke="#c8102e"
                      strokeWidth={2.5}
                      dot={false}
                      connectNulls
                    />
                    {simResult && (
                      <ReferenceLine
                        x={`${simResult.start_season}-${String((simResult.start_season + 1) % 100).padStart(2, "0")}`}
                        stroke="#64748b"
                        strokeWidth={1.5}
                        strokeDasharray="5 5"
                        label={{ value: "Sim start", fill: "#64748b", fontSize: 11 }}
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
    spg: "spg",
    bpg: "bpg",
    mpg: "mpg",
    fg_pct: "fg_pct"
  };
  const key = metricMap[metric];
  return key ? row.sample_projection.per_game[key] : row.metrics.ppg.p50;
}
