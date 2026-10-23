/** The shapes the API returns. No React, no fetch, no formatting. */

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

export type WsMessage = GateEventMessage | { type: "ping"; timestamp: string };

export const REQUESTABLE_COLOURS = [
  "blanco",
  "plata",
  "gris",
  "negro",
  "azul",
  "rojo",
] as const;
