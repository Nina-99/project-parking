// SPDX-FileCopyrightText: 2026 Nina
// SPDX-License-Identifier: LicenseRef-Proprietary

/** HTTP adapter. The only module that knows the API base path. */
import type { ApiPort } from "../application/ports";
import type {
  ActiveSession,
  CamerasResponse,
  CloseSessionResult,
  DailyStat,
  ModelStats,
  Occupancy,
  SessionHistory,
  VehicleDetail,
  VehicleSearchHit,
  VehicleSearchQuery,
} from "../domain/types";

const BASE = "/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    credentials: "include",
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...(init?.headers ?? {}),
    },
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
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function fetchJson<T>(path: string): Promise<T> {
  return request<T>(path);
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: JSON.stringify(body) });
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

  getVehicle(id: string): Promise<VehicleDetail> {
    return fetchJson(`/vehicles/${id}`);
  }

  getCameras(): Promise<CamerasResponse> {
    return fetchJson("/cameras");
  }

  getModelStats(days = 7): Promise<ModelStats> {
    return fetchJson(`/stats/model?days=${days}`);
  }

  closeSession(id: string, reason: string): Promise<CloseSessionResult> {
    return postJson(`/sessions/${id}/close`, { reason });
  }
}
