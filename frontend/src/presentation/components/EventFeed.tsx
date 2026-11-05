// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

import { useState } from "react";
import { Activity, Eye } from "lucide-react";
import { useLiveEvents } from "../../application/liveEvents";
import { formatClock } from "../../domain/incident";
import type { WsMessage } from "../../domain/types";
import { VehiclePreviewModal } from "./VehiclePreviewModal";

function title(event: Exclude<WsMessage, { type: "ping" }>): string {
  switch (event.type) {
    case "vehicle_entry":
      return "Ingreso";
    case "vehicle_exit":
      return "Salida";
    case "vehicle_exit_unmatched":
      return "Salida sin match";
    default:
      return "Cierre manual";
  }
}

function detail(event: Exclude<WsMessage, { type: "ping" }>): string {
  if (event.type === "session_closed") {
    return `${event.operator} · ${event.reason}`;
  }
  const plate = event.license_plate ?? "sin placa";
  const authorised =
    event.authorized === false
      ? " · NO AUTORIZADO"
      : event.authorized === true
        ? " · en padrón"
        : "";
  return `${plate} · ${event.vehicle_type ?? "vehículo"} · ${event.camera_id}${authorised}`;
}

function tone(type: WsMessage["type"]): string {
  switch (type) {
    case "vehicle_entry":
      return "text-live";
    case "vehicle_exit_unmatched":
      return "text-warn";
    case "session_closed":
      return "text-accent";
    default:
      return "text-ink";
  }
}

export function EventFeed() {
  const [selectedVehicleId, setSelectedVehicleId] = useState<string | null>(null);
  const events = useLiveEvents().filter(
    (event): event is Exclude<WsMessage, { type: "ping" }> => event.type !== "ping",
  );

  return (
    <>
      <section className="card flex max-h-[26rem] flex-col p-0">
        <header className="flex items-center justify-between border-b border-border px-5 py-4">
          <h3 className="text-sm font-semibold text-ink">En vivo</h3>
          <span className="inline-flex items-center gap-2 text-[11px] font-medium uppercase tracking-[0.14em] text-subtle">
            <span className="live-dot" />
            streaming
          </span>
        </header>

        <div className="flex-1 overflow-y-auto">
          {events.length === 0 ? (
            <div className="flex h-40 flex-col items-center justify-center gap-2 text-center">
              <Activity className="h-5 w-5 text-subtle" />
              <p className="text-sm text-muted">Esperando el próximo vehículo</p>
              <p className="text-xs text-subtle">Los eventos aparecen acá sin refrescar.</p>
            </div>
          ) : (
            <ul>
              {events.map((event) => {
                const vehicleId =
                  event.type !== "session_closed" ? event.vehicle_id : null;

                return (
                  <li
                    key={`${event.type}-${event.timestamp}-${event.type === "session_closed" ? event.session_id : event.vehicle_id}`}
                    className={`feed-item flex items-center justify-between gap-3 border-b border-border px-5 py-3 last:border-0 ${
                      vehicleId ? "cursor-pointer hover:bg-elevated/60" : ""
                    }`}
                    onClick={() => {
                      if (vehicleId) setSelectedVehicleId(vehicleId);
                    }}
                  >
                    <div className="flex items-start gap-3 min-w-0">
                      <span className="w-12 shrink-0 pt-0.5 font-mono text-[11px] tabular-nums text-subtle">
                        {formatClock(event.timestamp)}
                      </span>
                      <div className="min-w-0">
                        <p className={`text-sm font-medium ${tone(event.type)}`}>
                          {title(event)}
                        </p>
                        <p className="truncate text-xs text-muted">{detail(event)}</p>
                      </div>
                    </div>
                    {vehicleId && (
                      <span
                        title="Ver evidencia"
                        className="rounded p-1 text-subtle hover:text-ink shrink-0"
                      >
                        <Eye className="h-4 w-4" />
                      </span>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </section>

      {selectedVehicleId && (
        <VehiclePreviewModal
          vehicleId={selectedVehicleId}
          onDismiss={() => setSelectedVehicleId(null)}
        />
      )}
    </>
  );
}
