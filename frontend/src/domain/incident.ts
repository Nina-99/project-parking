// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

/** Pure presentation helpers: no React, no fetch. Easy to test. */
import type { GateEventMessage, MatchKind, VehicleSearchHit } from "./types";

const LOCALE = "es-BO";

export function formatDateTime(value: string): string {
  return new Date(value).toLocaleString(LOCALE, {
    dateStyle: "short",
    timeStyle: "short",
  });
}

export function formatTime(value: string): string {
  return new Date(value).toLocaleTimeString(LOCALE);
}

export function formatClock(value: string): string {
  return new Date(value).toLocaleTimeString(LOCALE, {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

export function formatDuration(seconds: number | null): string {
  if (!seconds) return "—";
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return hours > 0 ? `${hours}h ${rest}m` : `${rest}m`;
}

export function formatConfidence(value: number | null): string {
  return value === null ? "—" : `${Math.round(value * 100)}%`;
}

export function describeMatch(hit: VehicleSearchHit, requestedColour?: string): string {
  if (!requestedColour) {
    return "Candidato por ventana horaria.";
  }
  if (hit.match === "exact") {
    return `Coincidencia exacta de ${requestedColour}.`;
  }
  return `Vecino medido: el clasificador confunde ${requestedColour} con ${hit.color}.`;
}

export function eventLabel(event: GateEventMessage): string {
  switch (event.type) {
    case "vehicle_entry":
      return "Ingreso";
    case "vehicle_exit":
      return "Salida";
    default:
      return "Salida sin match";
  }
}

export function matchBadgeClass(match: MatchKind): string {
  return match === "exact"
    ? "bg-green-100 text-green-800"
    : "bg-amber-100 text-amber-800";
}
