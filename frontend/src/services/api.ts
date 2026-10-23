const API_BASE = "/api/v1";

export interface ActiveSession {
  id: string;
  vehicle_id: string;
  entry_time: string;
  entry_gate: string | null;
  license_plate: string | null;
  vehicle_type: string | null;
  color: string | null;
}

export interface SessionHistory {
  id: string;
  vehicle_id: string;
  entry_gate: string | null;
  exit_gate: string | null;
  entry_time: string;
  exit_time: string | null;
  duration_seconds: number | null;
  status: string;
  license_plate: string | null;
  vehicle_type: string | null;
  color: string | null;
}

export interface Occupancy {
  active_sessions: number;
  timestamp: string;
}

export interface DailyStat {
  date: string;
  total_vehicles: number;
  avg_duration_minutes: number | null;
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function getActiveSessions(): Promise<ActiveSession[]> {
  return fetchJson("/stats/active");
}

export async function getHistory(
  limit = 50,
  offset = 0
): Promise<SessionHistory[]> {
  return fetchJson(`/stats/history?limit=${limit}&offset=${offset}`);
}

export async function getOccupancy(): Promise<Occupancy> {
  return fetchJson("/stats/occupancy");
}

export async function getDailyStats(days = 7): Promise<DailyStat[]> {
  return fetchJson(`/stats/daily?days=${days}`);
}
