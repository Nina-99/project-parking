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
        <h2 className="text-2xl font-bold text-gray-900">Historial</h2>

        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Buscar placa, tipo, color..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            className="pl-9 pr-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none w-72"
          />
        </div>
      </div>

      <div className="card overflow-hidden p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 bg-gray-50">
                <th className="text-left py-3 px-4 font-medium text-gray-500">
                  <span className="flex items-center gap-1">
                    Placa <ArrowUpDown className="w-3 h-3" />
                  </span>
                </th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">Tipo</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">Color</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">Ingreso</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">Salida</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">Duración</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">Estado</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">Cierre</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {history.isLoading ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-gray-400">
                    Cargando…
                  </td>
                </tr>
              ) : history.isError ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-red-500">
                    No se pudo cargar el historial
                  </td>
                </tr>
              ) : filtered.length > 0 ? (
                filtered.map((session) => (
                  <tr key={session.id} className="hover:bg-gray-50">
                    <td className="py-3 px-4 font-medium text-gray-900">
                      {session.license_plate || "—"}
                    </td>
                    <td className="py-3 px-4 text-gray-600">
                      {session.vehicle_type || "—"}
                    </td>
                    <td className="py-3 px-4 text-gray-600">{session.color || "—"}</td>
                    <td className="py-3 px-4 text-gray-600">
                      {formatDateTime(session.entry_time)}
                    </td>
                    <td className="py-3 px-4 text-gray-600">
                      {session.exit_time ? formatDateTime(session.exit_time) : "—"}
                    </td>
                    <td className="py-3 px-4 text-gray-600">
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
                    <td className="py-3 px-4 text-gray-600 text-xs">
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
                  <td colSpan={8} className="py-12 text-center text-gray-400">
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
