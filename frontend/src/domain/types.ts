/** The shapes the API returns. No React, no fetch, no formatting. */

export interface ActiveSession {
  id: string;
  vehicle_id: string;
  entry_time: string;
  entry_gate: string | null;
  license_plate: string | null;
  vehicle_type: string | null;
  color: string | null;
  stale: boolean;
  age_minutes: number;
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
  close_reason: string | null;
  closed_by: string | null;
}

export interface Occupancy {
  active_sessions: number;
  capacity: number;
  percent: number | null;
  timestamp: string;
}

export interface DailyStat {
  date: string;
  total_vehicles: number;
  avg_duration_minutes: number | null;
}

export interface VehicleDetail {
  id: string;
  license_plate: string | null;
  vehicle_type: string | null;
  vehicle_type_confidence: number | null;
  color: string | null;
  color_confidence: number | null;
  image_url: string | null;
  created_at: string;
  sessions: VehicleSession[];
}

export interface VehicleSession {
  id: string;
  entry_gate: string | null;
  exit_gate: string | null;
  entry_time: string;
  exit_time: string | null;
  duration_seconds: number | null;
  status: string;
  close_reason: string | null;
  closed_by: string | null;
}

export interface CameraReading {
  camera_id: string;
  gate: string | null;
  last_seen: string | null;
  online: boolean;
  dropped_frames: number;
}

export interface CamerasResponse {
  redis_available: boolean;
  cameras: CameraReading[];
}

export interface ModelStat {
  model_name: string;
  events: number;
  avg_confidence: number | null;
  p50_latency_ms: number | null;
  p95_latency_ms: number | null;
}

export interface ModelStats {
  window_days: number;
  models: ModelStat[];
  total_events: number;
}

export interface CloseSessionResult {
  session_id: string;
  duration_seconds: number;
  reason: string;
  operator: string;
}

export type MatchKind = "exact" | "neighbour";

export interface VehicleSearchHit {
  id: string;
  license_plate: string | null;
  vehicle_type: string | null;
  color: string | null;
  color_confidence: number | null;
  image_url: string | null;
  created_at: string;
  match: MatchKind;
}

export interface VehicleSearchQuery {
  /** ISO instant, inclusive. */
  from: string;
  /** ISO instant, inclusive. */
  to: string;
  color?: string;
  type?: string;
  at?: string;
}

export interface GateEventMessage {
  type: "vehicle_entry" | "vehicle_exit" | "vehicle_exit_unmatched";
  camera_id: string;
  vehicle_id: string;
  session_id: string | null;
  vehicle_type: string | null;
  color: string | null;
  timestamp: string;
}

export interface SessionClosedMessage {
  type: "session_closed";
  session_id: string;
  operator: string;
  reason: string;
  timestamp: string;
}

export type WsMessage =
  | GateEventMessage
  | SessionClosedMessage
  | { type: "ping"; timestamp: string };

export const REQUESTABLE_COLOURS = [
  "blanco",
  "plata",
  "gris",
  "negro",
  "azul",
  "rojo",
] as const;
