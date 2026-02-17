const toJson = async (res) => {
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
};

export async function searchPlayers(query) {
  const res = await fetch(`/api/players/search?query=${encodeURIComponent(query)}`);
  return toJson(res);
}

export async function getCareer(playerId) {
  const res = await fetch(`/api/players/${playerId}/career`);
  return toJson(res);
}

export async function simulateCareer(payload) {
  const res = await fetch("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return toJson(res);
}

export async function getModelInfo() {
  const res = await fetch("/api/models/current");
  return toJson(res);
}
