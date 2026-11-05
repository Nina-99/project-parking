// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

import { useState } from "react";
import { ArrowUpDown, ChevronLeft, ChevronRight, Download, FilterX, Search } from "lucide-react";
import { useCameras, useHistory } from "../../application/queries";
import { formatDateTime, formatDuration } from "../../domain/incident";
import type { SessionHistory } from "../../domain/types";

function exportToCsv(data: SessionHistory[]) {
  const headers = [
    "Placa",
    "Tipo",
    "Color",
    "Porton Ingreso",
    "Porton Salida",
    "Ingreso",
    "Salida",
    "Duracion (seg)",
    "Estado",
    "Cierre",
  ];

  const rows = data.map((s) => [
    `"${s.license_plate || ""}"`,
    `"${s.vehicle_type || ""}"`,
    `"${s.color || ""}"`,
    `"${s.entry_gate || ""}"`,
    `"${s.exit_gate || ""}"`,
    `"${s.entry_time || ""}"`,
    `"${s.exit_time || ""}"`,
    s.duration_seconds ?? "",
    `"${s.status || ""}"`,
    `"${s.closed_by ? `${s.closed_by}: ${s.close_reason || ""}` : s.close_reason || ""}"`,
  ]);

  const csvContent =
    "data:text/csv;charset=utf-8,\uFEFF" +
    [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute(
    "download",
    `historial_parqueadero_${new Date().toISOString().slice(0, 10)}.csv`,
  );
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

export function History() {
  const [search, setSearch] = useState("");
  const [gateFilter, setGateFilter] = useState("all");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [page, setPage] = useState(1);
  const pageSize = 20;

  const history = useHistory({
    limit: pageSize,
    offset: (page - 1) * pageSize,
    search: search.trim() || undefined,
    gate: gateFilter !== "all" ? gateFilter : undefined,
    from_date: fromDate ? `${fromDate}T00:00:00Z` : undefined,
    to_date: toDate ? `${toDate}T23:59:59Z` : undefined,
    status: statusFilter !== "all" ? statusFilter : undefined,
  });

  const cameras = useCameras();

  const cameraGates = Array.from(
    new Set(
      (cameras.data?.cameras ?? [])
        .map((c) => c.gate)
        .filter((g): g is string => Boolean(g)),
    ),
  );

  const total = history.data?.total ?? 0;
  const items = history.data?.items ?? [];
  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  const hasActiveFilters =
    search.trim() !== "" ||
    gateFilter !== "all" ||
    fromDate !== "" ||
    toDate !== "" ||
    statusFilter !== "all";

  const clearFilters = () => {
    setSearch("");
    setGateFilter("all");
    setFromDate("");
    setToDate("");
    setStatusFilter("all");
    setPage(1);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-subtle">
            Auditoría y consultas
          </p>
          <h2 className="font-display text-2xl font-bold tracking-tight text-ink">
            Historial de eventos
          </h2>
        </div>

        <button
          type="button"
          onClick={() => exportToCsv(items)}
          disabled={items.length === 0}
          className="btn-secondary inline-flex items-center gap-2 self-start sm:self-auto"
        >
          <Download className="h-4 w-4" />
          Exportar página ({items.length})
        </button>
      </div>

      {/* Filter Toolbar as specified in thesis wireframe Fig 10b */}
      <div className="card grid gap-3 sm:grid-cols-2 lg:grid-cols-12">
        <div className="relative sm:col-span-2 lg:col-span-3">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-subtle" />
          <input
            type="text"
            placeholder="Buscar por placa, tipo o color..."
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              setPage(1);
            }}
            className="w-full rounded-lg border border-border bg-canvas pl-9 pr-4 py-2 text-sm text-ink outline-none focus:border-accent focus:ring-2 focus:ring-accent/30"
          />
        </div>

        <div className="flex items-center gap-2 lg:col-span-2">
          <label htmlFor="gate-filter" className="text-xs text-muted shrink-0">
            Portón:
          </label>
          <select
            id="gate-filter"
            value={gateFilter}
            onChange={(e) => {
              setGateFilter(e.target.value);
              setPage(1);
            }}
            className="w-full rounded-lg border border-border bg-canvas px-2.5 py-1.5 text-xs text-ink outline-none focus:border-accent"
          >
            <option value="all">Todos</option>
            {cameraGates.map((gate) => (
              <option key={gate} value={gate}>
                {gate}
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2 lg:col-span-2">
          <label htmlFor="status-filter" className="text-xs text-muted shrink-0">
            Estado:
          </label>
          <select
            id="status-filter"
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value);
              setPage(1);
            }}
            className="w-full rounded-lg border border-border bg-canvas px-2.5 py-1.5 text-xs text-ink outline-none focus:border-accent"
          >
            <option value="all">Todos</option>
            <option value="active">Activo</option>
            <option value="completed">Completado</option>
          </select>
        </div>

        <div className="flex items-center gap-2 lg:col-span-2">
          <label htmlFor="from-date" className="text-xs text-muted shrink-0">
            Desde:
          </label>
          <input
            id="from-date"
            type="date"
            value={fromDate}
            onChange={(e) => {
              setFromDate(e.target.value);
              setPage(1);
            }}
            className="w-full rounded-lg border border-border bg-canvas px-2 py-1.5 text-xs text-ink outline-none focus:border-accent"
          />
        </div>

        <div className="flex items-center gap-2 lg:col-span-2">
          <label htmlFor="to-date" className="text-xs text-muted shrink-0">
            Hasta:
          </label>
          <input
            id="to-date"
            type="date"
            value={toDate}
            onChange={(e) => {
              setToDate(e.target.value);
              setPage(1);
            }}
            className="w-full rounded-lg border border-border bg-canvas px-2 py-1.5 text-xs text-ink outline-none focus:border-accent"
          />
        </div>

        {hasActiveFilters && (
          <div className="flex items-center justify-end lg:col-span-1">
            <button
              type="button"
              onClick={clearFilters}
              title="Limpiar filtros"
              className="btn-ghost p-2 text-xs text-muted hover:text-ink"
            >
              <FilterX className="h-4 w-4" />
            </button>
          </div>
        )}
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
                <th className="text-left py-3 px-4 font-medium text-muted">Portón</th>
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
                  <td colSpan={9} className="py-12 text-center text-subtle">
                    Cargando…
                  </td>
                </tr>
              ) : history.isError ? (
                <tr>
                  <td colSpan={9} className="py-12 text-center text-danger">
                    No se pudo cargar el historial
                  </td>
                </tr>
              ) : items.length > 0 ? (
                items.map((session) => (
                  <tr key={session.id} className="hover:bg-elevated">
                    <td className="py-3 px-4 font-medium font-mono text-ink">
                      {session.license_plate || "—"}
                    </td>
                    <td className="py-3 px-4 text-muted">
                      {session.vehicle_type || "—"}
                    </td>
                    <td className="py-3 px-4 text-muted">{session.color || "—"}</td>
                    <td className="py-3 px-4 text-muted text-xs">
                      {session.entry_gate ?? "—"}
                      {session.exit_gate && session.exit_gate !== session.entry_gate
                        ? ` → ${session.exit_gate}`
                        : ""}
                    </td>
                    <td className="py-3 px-4 text-muted">
                      {formatDateTime(session.entry_time)}
                    </td>
                    <td className="py-3 px-4 text-muted">
                      {session.exit_time ? formatDateTime(session.exit_time) : "—"}
                    </td>
                    <td className="py-3 px-4 text-muted font-mono text-xs">
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
                  <td colSpan={9} className="py-12 text-center text-subtle">
                    {hasActiveFilters
                      ? "No se encontraron resultados para los filtros seleccionados"
                      : "Sin registros de ingreso"}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination wireframe: < 1 2 3 > */}
        <footer className="flex items-center justify-between border-t border-border px-4 py-3 bg-canvas/30 text-xs">
          <span className="text-muted">
            Mostrando {items.length} de {total} registros
          </span>

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="btn-ghost p-1.5 text-muted hover:text-ink disabled:opacity-40"
              aria-label="Página anterior"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span className="px-2 font-mono text-ink">
              {page} / {totalPages}
            </span>
            <button
              type="button"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              className="btn-ghost p-1.5 text-muted hover:text-ink disabled:opacity-40"
              aria-label="Página siguiente"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}
