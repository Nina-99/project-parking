/** HTTP adapter. The only module that knows the API base path. */
import type { ApiPort } from "../application/ports";
import type {
  ActiveSession,
  DailyStat,
  Occupancy,
  SessionHistory,
  VehicleSearchHit,
  VehicleSearchQuery,
} from "../domain/types";

const BASE = "/api/v1";

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    credentials: "include",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body: unknown = await response.json();
      if (body && typeof body === "object" && "detail" in body) {
        detail = String((body as { detail: unknown }).detail);
      }
    } catch {
      // keep the status line
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export class HttpApi implements ApiPort {
  getActiveSessions(): Promise<ActiveSession[]> {
    return fetchJson("/stats/active");
  }

  getHistory(limit = 50, offset = 0): Promise<SessionHistory[]> {
    return fetchJson(`/stats/history?limit=${limit}&offset=${offset}`);
  }

  getOccupancy(): Promise<Occupancy> {
    return fetchJson("/stats/occupancy");
  }

  getDailyStats(days = 7): Promise<DailyStat[]> {
    return fetchJson(`/stats/daily?days=${days}`);
  }

  searchVehicles(query: VehicleSearchQuery): Promise<VehicleSearchHit[]> {
    const params = new URLSearchParams({ from: query.from, to: query.to });
    if (query.color) params.set("color", query.color);
    if (query.type) params.set("type", query.type);
    if (query.at) params.set("at", query.at);
    return fetchJson(`/vehicles/search?${params.toString()}`);
  }
}
