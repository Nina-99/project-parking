// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

/** HTTP adapter. The only module that knows the API base path. */
import type { ApiPort } from "../application/ports";
import type {
  ActiveSession,
  AuthorizedVehicle,
  AuthorizedVehicleCreateInput,
  AuthorizedVehicleUpdateInput,
  CameraCreateInput,
  CamerasResponse,
  CameraUpdateInput,
  CloseSessionResult,
  ColourStats,
  DailyStat,
  HourlyStats,
  ManagedUser,
  HistoryQuery,
  PaginatedHistory,
  ModelStats,
  Occupancy,
  PendingValidation,
  QualityStats,
  SessionHistory,
  ValidationDecisionInput,
  ValidationDecisionResult,
  VehicleDetail,
  UserRole,
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

function patchJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "PATCH", body: JSON.stringify(body) });
}

export class HttpApi implements ApiPort {
  getActiveSessions(): Promise<ActiveSession[]> {
    return fetchJson("/stats/active");
  }

  async getHistory(query?: HistoryQuery): Promise<PaginatedHistory> {
    const params = new URLSearchParams();
    if (query?.limit !== undefined) params.set("limit", String(query.limit));
    if (query?.offset !== undefined) params.set("offset", String(query.offset));
    if (query?.gate) params.set("gate", query.gate);
    if (query?.from_date) params.set("from_date", query.from_date);
    if (query?.to_date) params.set("to_date", query.to_date);
    if (query?.search) params.set("search", query.search);
    if (query?.status) params.set("status", query.status);

    const qs = params.toString();
    const path = qs ? `/stats/history?${qs}` : "/stats/history";

    const response = await fetch(`${BASE}${path}`, {
      credentials: "include",
      headers: { Accept: "application/json" },
    });

    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }

    const totalHeader = response.headers.get("X-Total-Count");
    const total = totalHeader ? parseInt(totalHeader, 10) : 0;
    const items = (await response.json()) as SessionHistory[];

    return { items, total };
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

  createCamera(input: CameraCreateInput): Promise<void> {
    return postJson("/cameras", input);
  }

  updateCamera(cameraId: string, input: CameraUpdateInput): Promise<void> {
    return patchJson(`/cameras/${cameraId}`, input);
  }

  deleteCamera(cameraId: string): Promise<void> {
    return request<void>(`/cameras/${cameraId}`, { method: "DELETE" });
  }

  listAuthorizedVehicles(query?: {
    search?: string;
    active_only?: boolean;
  }): Promise<AuthorizedVehicle[]> {
    const params = new URLSearchParams();
    if (query?.search) params.set("search", query.search);
    if (query?.active_only) params.set("active_only", "true");
    const qs = params.toString();
    return fetchJson(`/whitelist${qs ? `?${qs}` : ""}`);
  }

  createAuthorizedVehicle(
    input: AuthorizedVehicleCreateInput,
  ): Promise<AuthorizedVehicle> {
    return postJson("/whitelist", input);
  }

  updateAuthorizedVehicle(
    id: string,
    input: AuthorizedVehicleUpdateInput,
  ): Promise<AuthorizedVehicle> {
    return patchJson(`/whitelist/${id}`, input);
  }

  deleteAuthorizedVehicle(id: string): Promise<void> {
    return request<void>(`/whitelist/${id}`, { method: "DELETE" });
  }

  checkAuthorization(
    licensePlate: string,
  ): Promise<{ authorized: boolean; vehicle: AuthorizedVehicle | null }> {
    return fetchJson(`/whitelist/check/${encodeURIComponent(licensePlate)}`);
  }

  getModelStats(days = 7): Promise<ModelStats> {
    return fetchJson(`/stats/model?days=${days}`);
  }

  getQualityStats(days = 7): Promise<QualityStats> {
    return fetchJson(`/stats/quality?days=${days}`);
  }

  getColourStats(days = 30): Promise<ColourStats> {
    return fetchJson(`/stats/colors?days=${days}`);
  }

  getHourlyStats(days = 7): Promise<HourlyStats> {
    return fetchJson(`/stats/hourly?days=${days}`);
  }

  listUsers(): Promise<ManagedUser[]> {
    return fetchJson("/users");
  }

  createUser(input: {
    username: string;
    password: string;
    role: UserRole;
  }): Promise<ManagedUser> {
    return postJson("/users", input);
  }

  updateUser(
    id: string,
    input: { role?: UserRole; is_active?: boolean; password?: string },
  ): Promise<ManagedUser> {
    return request(`/users/${id}`, { method: "PATCH", body: JSON.stringify(input) });
  }

  deleteUser(id: string): Promise<void> {
    return request(`/users/${id}`, { method: "DELETE" });
  }

  closeSession(id: string, reason: string): Promise<CloseSessionResult> {
    return postJson(`/sessions/${id}/close`, { reason });
  }

  getPendingValidations(hours = 8): Promise<PendingValidation[]> {
    return fetchJson(`/validations/pending?hours=${hours}`);
  }

  submitValidationDecision(
    vehicleId: string,
    input: ValidationDecisionInput,
  ): Promise<ValidationDecisionResult> {
    return postJson(`/validations/${vehicleId}/decision`, input);
  }
}
