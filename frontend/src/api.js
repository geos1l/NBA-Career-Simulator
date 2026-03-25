const toJson = async (res) => {
  const text = await res.text();
  let data;
  try {
    data = text.length ? JSON.parse(text) : {};
  } catch {
    throw new Error(res.ok ? "Invalid response from server" : `Request failed: ${text.slice(0, 80)}`);
  }
  if (!res.ok) {
    throw new Error(data.detail || data.message || "Request failed");
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

/**
 * Same payload as simulateCareer; streams SSE from /api/simulate/stream.
 * onProgress receives { event, pct, phase, message, done, total, eta_seconds, ... }.
 */
export async function simulateCareerStream(payload, onProgress) {
  const res = await fetch("/api/simulate/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream"
    },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    const text = await res.text();
    let data;
    try {
      data = text.length ? JSON.parse(text) : {};
    } catch {
      throw new Error(text.slice(0, 120) || `Request failed (${res.status})`);
    }
    throw new Error(data.detail || data.message || `Request failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      const rawBlock = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      for (const line of rawBlock.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        let data;
        try {
          data = JSON.parse(line.slice(6));
        } catch {
          continue;
        }
        if (data.event === "progress") onProgress?.(data);
        if (data.event === "complete") {
          onProgress?.({ event: "progress", pct: data.pct ?? 100, message: "Done" });
          return data.result;
        }
        if (data.event === "error") {
          throw new Error(data.detail || "Simulation failed");
        }
      }
    }
  }

  throw new Error("Stream ended before simulation completed");
}

export async function getModelInfo() {
  const res = await fetch("/api/models/current");
  return toJson(res);
}
