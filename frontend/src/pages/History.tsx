import { useQuery } from "@tanstack/react-query";
import { Search, ArrowUpDown } from "lucide-react";
import { getHistory, type SessionHistory } from "../services/api";
import { useState } from "react";

function formatDuration(seconds: number | null): string {
  if (!seconds) return "—";
  const m = Math.floor(seconds / 60);
  const h = Math.floor(m / 60);
  const mins = m % 60;
  return h > 0 ? `${h}h ${mins}m` : `${mins}m`;
}

export function History() {
  const [search, setSearch] = useState("");
  const { data: sessions } = useQuery({
    queryKey: ["history"],
    queryFn: () => getHistory(100),
  });

  const filtered = (sessions ?? []).filter((s) => {
    if (!search.trim()) return true;
    const q = search.toLowerCase();
    return (
      (s.license_plate?.toLowerCase() ?? "").includes(q) ||
      (s.vehicle_type?.toLowerCase() ?? "").includes(q) ||
      (s.color?.toLowerCase() ?? "").includes(q)
    );
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Historial</h2>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Buscar placa, tipo, color..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9 pr-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none w-72"
          />
        </div>
      </div>

      {/* Table */}
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
                <th className="text-left py-3 px-4 font-medium text-gray-500">
                  Tipo
                </th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">
                  Color
                </th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">
                  Ingreso
                </th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">
                  Salida
                </th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">
                  Duración
                </th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">
                  Estado
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {filtered.length > 0 ? (
                filtered.map((s) => (
                  <tr key={s.id} className="hover:bg-gray-50">
                    <td className="py-3 px-4 font-medium text-gray-900">
                      {s.license_plate || "—"}
                    </td>
                    <td className="py-3 px-4 text-gray-600">
                      {s.vehicle_type || "—"}
                    </td>
                    <td className="py-3 px-4 text-gray-600">
                      {s.color || "—"}
                    </td>
                    <td className="py-3 px-4 text-gray-600">
                      {new Date(s.entry_time).toLocaleString("es-BO")}
                    </td>
                    <td className="py-3 px-4 text-gray-600">
                      {s.exit_time
                        ? new Date(s.exit_time).toLocaleString("es-BO")
                        : "—"}
                    </td>
                    <td className="py-3 px-4 text-gray-600">
                      {formatDuration(s.duration_seconds)}
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={
                          s.status === "active"
                            ? "badge-active"
                            : "badge-completed"
                        }
                      >
                        {s.status === "active" ? "Activo" : "Completado"}
                      </span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td
                    colSpan={7}
                    className="py-12 text-center text-gray-400"
                  >
                    {search
                      ? "No se encontraron resultados"
                      : "Sin registros de ingreso"}
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
