// SPDX-FileCopyrightText: 2026 Nina
// SPDX-License-Identifier: LicenseRef-Proprietary

import { useState } from "react";
import { ArrowUpDown, Search } from "lucide-react";
import { useHistory } from "../../application/queries";
import { formatDateTime, formatDuration } from "../../domain/incident";

export function History() {
  const [search, setSearch] = useState("");
  const history = useHistory(200);

  const filtered = (history.data ?? []).filter((session) => {
    if (!search.trim()) return true;
    const query = search.toLowerCase();
    return (
      (session.license_plate?.toLowerCase() ?? "").includes(query) ||
      (session.vehicle_type?.toLowerCase() ?? "").includes(query) ||
      (session.color?.toLowerCase() ?? "").includes(query)
    );
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="font-display text-2xl font-bold text-ink">Historial</h2>

        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-subtle" />
          <input
            type="text"
            placeholder="Buscar placa, tipo, color..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            className="pl-9 pr-4 py-2 border border-border rounded-lg text-sm focus:ring-2 focus:ring-accent/30 focus:border-accent outline-none w-72"
          />
        </div>
      </div>

      <div className="card overflow-hidden p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-elevated">
                <th className="text-left py-3 px-4 font-medium text-muted">
                  <span className="flex items-center gap-1">
                    Placa <ArrowUpDown className="w-3 h-3" />
                  </span>
                </th>
                <th className="text-left py-3 px-4 font-medium text-muted">Tipo</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Color</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Ingreso</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Salida</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Duración</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Estado</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Cierre</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {history.isLoading ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-subtle">
                    Cargando…
                  </td>
                </tr>
              ) : history.isError ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-danger">
                    No se pudo cargar el historial
                  </td>
                </tr>
              ) : filtered.length > 0 ? (
                filtered.map((session) => (
                  <tr key={session.id} className="hover:bg-elevated">
                    <td className="py-3 px-4 font-medium text-ink">
                      {session.license_plate || "—"}
                    </td>
                    <td className="py-3 px-4 text-muted">
                      {session.vehicle_type || "—"}
                    </td>
                    <td className="py-3 px-4 text-muted">{session.color || "—"}</td>
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
                    <td className="py-3 px-4 text-muted text-xs">
                      {session.closed_by
                        ? `${session.closed_by}${session.close_reason ? `: ${session.close_reason}` : ""}`
                        : session.close_reason === "plate_match"
                          ? "Automático (patente)"
                          : "—"}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-subtle">
                    {search ? "No se encontraron resultados" : "Sin registros de ingreso"}
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
