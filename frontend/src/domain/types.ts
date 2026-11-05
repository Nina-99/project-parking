// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

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

export interface HistoryQuery {
  limit?: number;
  offset?: number;
  gate?: string;
  from_date?: string;
  to_date?: string;
  search?: string;
  status?: string;
}

export interface PaginatedHistory {
  items: SessionHistory[];
  total: number;
}

export interface AuthorizedVehicle {
  id: string;
  license_plate: string;
  owner_name: string | null;
  vehicle_type: string | null;
  color: string | null;
  notes: string | null;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface AuthorizedVehicleCreateInput {
  license_plate: string;
  owner_name?: string;
  vehicle_type?: string;
  color?: string;
  notes?: string;
  is_active?: boolean;
}

export interface AuthorizedVehicleUpdateInput {
  owner_name?: string;
  vehicle_type?: string;
  color?: string;
  notes?: string;
  is_active?: boolean;
}

export interface CheckAuthorizationResult {
  authorized: boolean;
  vehicle: AuthorizedVehicle | null;
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
  clip_url: string | null;
  plate_characters: PlateCharacter[] | null;
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
  label: string | null;
  last_seen: string | null;
  online: boolean;
  dropped_frames: number;
  registered: boolean;
}

export type CameraGate = "entry" | "exit";

export interface CameraCreateInput {
  camera_id: string;
  gate: CameraGate;
  label?: string;
  rtsp_url: string;
  is_active?: boolean;
}

export interface CameraUpdateInput {
  gate?: CameraGate;
  label?: string;
  rtsp_url?: string;
  is_active?: boolean;
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
  license_plate: string | null;
  plate_confidence: number | null;
  /** null when there was no plate to check: unknown is not "unauthorized". */
  authorized: boolean | null;
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

/** One gate event waiting for an operator decision. */
export interface PendingValidation {
  vehicle_id: string;
  camera_id: string | null;
  direction: "entry" | "exit" | null;
  observed_at: string | null;
  created_at: string;
  license_plate: string | null;
  plate_confidence: number | null;
  vehicle_type: string | null;
  vehicle_type_confidence: number | null;
  color: string | null;
  color_confidence: number | null;
  image_url: string | null;
  clip_url: string | null;
  plate_characters: PlateCharacter[] | null;
  authorized: boolean | null;
  session_id: string | null;
}

export type ValidationDecisionKind =
  | "confirmed"
  | "plate_corrected"
  | "dismissed"
  | "session_closed";

export interface PlateAlternative {
  char: string;
  confidence: number;
}

/** One emitted character, with the model's runners-up for that position. */
export interface PlateCharacter {
  char: string;
  confidence: number;
  alternatives: PlateAlternative[];
}

export interface ValidationDecisionInput {
  decision: ValidationDecisionKind;
  corrected_plate?: string;
  session_id?: string;
  notes?: string;
}

export interface ValidationDecisionResult {
  vehicle_id: string;
  decision: ValidationDecisionKind;
  original_plate: string | null;
  corrected_plate: string | null;
  operator: string;
}

export interface OperatorQuality {
  operator: string;
  decisions: number;
  corrected: number;
}

/**
 * How often the operator had to disagree with the machine, over a window.
 * Rates are null when there is nothing to divide by: an empty window has no
 * read rate, which is not the same as zero percent.
 */
export interface QualityStats {
  window_days: number;
  events_total: number;
  events_with_plate: number;
  plate_read_rate: number | null;
  unmatched_exits: number;
  decisions_total: number;
  by_decision: Record<ValidationDecisionKind, number>;
  correction_rate: number | null;
  operators: OperatorQuality[];
}

export interface ColourCounts {
  blanco: number;
  plata: number;
  rojo: number;
  gris: number;
  negro: number;
  azul: number;
  minority: number;
}

export interface ColourStats {
  window_days: number;
  min_confidence: number;
  reportable_colors: string[];
  by_type: Record<string, ColourCounts>;
  totals: ColourCounts;
  total_vehicles: number;
}

export interface HourlyPoint {
  hour: number;
  vehicles: number;
}

export interface HourlyStats {
  window_days: number;
  timezone: string;
  hours: HourlyPoint[];
}

export const REQUESTABLE_COLOURS = [
  "blanco",
  "plata",
  "gris",
  "negro",
  "azul",
  "rojo",
] as const;

/** Who can enter the system, and with which role. */
export type UserRole = "admin" | "operator" | "owner";

export interface SessionUser {
  username: string;
  role: UserRole;
}

export interface ManagedUser {
  id: string;
  username: string;
  role: UserRole;
  is_active: boolean;
  created_at: string;
  last_login_at: string | null;
}

export const ROLE_LABELS: Record<UserRole, string> = {
  admin: "Administrador",
  operator: "Operador",
  owner: "Dueño",
};

export const ROLE_DESCRIPTIONS: Record<UserRole, string> = {
  admin: "Gestiona usuarios y cámaras",
  operator: "Opera el estacionamiento",
  owner: "Solo lectura: métricas y reportes",
};
