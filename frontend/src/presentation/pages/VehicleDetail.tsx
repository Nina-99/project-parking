// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Car, ImageOff } from "lucide-react";
import { useVehicle } from "../../application/queries";
import {
  formatConfidence,
  formatDateTime,
  formatDuration,
} from "../../domain/incident";
import type { VehicleSession } from "../../domain/types";

function closureLabel(session: VehicleSession): string {
  if (session.closed_by) {
    return `${session.closed_by}${session.close_reason ? `: ${session.close_reason}` : ""}`;
  }
  if (session.close_reason === "plate_match") return "Automático (patente)";
  return "—";
}

export function VehicleDetail() {
  const { id } = useParams<{ id: string }>();
  const vehicle = useVehicle(id);

  if (vehicle.isLoading) {
    return <p className="text-sm text-subtle py-12 text-center">Cargando…</p>;
  }
  if (vehicle.isError || !vehicle.data) {
    return (
      <div className="space-y-4">
        <Link to="/incidents" className="inline-flex items-center gap-1 text-sm text-muted hover:text-ink">
          <ArrowLeft className="w-4 h-4" /> Volver a incidentes
        </Link>
        <p className="text-sm text-danger">No se encontró el vehículo.</p>
      </div>
    );
  }

  const data = vehicle.data;

  return (
    <div className="space-y-6">
      <Link
        to="/incidents"
        className="inline-flex items-center gap-1 text-sm text-muted hover:text-ink"
      >
        <ArrowLeft className="w-4 h-4" /> Volver a incidentes
      </Link>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card p-0 overflow-hidden">
          {data.image_url ? (
            <img
              src={data.image_url}
              alt={`Vehículo ${data.license_plate ?? "sin placa"}`}
              className="w-full h-80 object-cover bg-elevated"
            />
          ) : (
            <div className="h-80 bg-elevated flex items-center justify-center text-subtle">
              <ImageOff className="w-10 h-10" />
            </div>
          )}
        </div>

        <div className="card space-y-4">
          <div>
            <p className="font-mono text-2xl font-semibold text-ink">
              {data.license_plate ?? "Sin placa"}
            </p>
            <p className="text-sm text-muted">
              Registrado el {formatDateTime(data.created_at)}
            </p>
          </div>

          <dl className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <dt className="text-muted">Tipo</dt>
              <dd className="text-ink">
                {data.vehicle_type ?? "—"} · {formatConfidence(data.vehicle_type_confidence)}
              </dd>
            </div>
            <div>
              <dt className="text-muted">Color</dt>
              <dd className="text-ink">
                {data.color ?? "—"} · {formatConfidence(data.color_confidence)}
              </dd>
            </div>
            <div>
              <dt className="text-muted">Sesiones</dt>
              <dd className="text-ink">{data.sessions.length}</dd>
            </div>
            <div>
              <dt className="text-muted">Id</dt>
              <dd className="text-muted font-mono text-xs truncate">{data.id}</dd>
            </div>
          </dl>

          <p className="text-xs text-subtle flex items-center gap-1">
            <Car className="w-3 h-3" />
            El thumbnail corresponde a la vista más grande del vehículo.
          </p>
        </div>
      </div>

      <div className="card overflow-hidden p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-elevated">
                <th className="text-left py-3 px-4 font-medium text-muted">Ingreso</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Salida</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Duración</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Estado</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Cierre</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {data.sessions.length > 0 ? (
                data.sessions.map((session) => (
                  <tr key={session.id} className="hover:bg-elevated">
                    <td className="py-3 px-4 text-muted">
                      {formatDateTime(session.entry_time)}
                    </td>
                    <td className="py-3 px-4 text-muted">
                      {session.exit_time ? formatDateTime(session.exit_time) : "—"}
                    </td>
                    <td className="py-3 px-4 text-muted">
                      {formatDuration(session.duration_seconds)}
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={
                          session.status === "active" ? "badge-active" : "badge-completed"
                        }
                      >
                        {session.status === "active" ? "Activo" : "Completado"}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-muted">
                      {closureLabel(session)}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="py-12 text-center text-subtle">
                    Sin sesiones registradas
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
