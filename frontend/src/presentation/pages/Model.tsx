import { Activity, Cpu, Timer } from "lucide-react";
import { useModelStats } from "../../application/queries";
import { formatConfidence } from "../../domain/incident";
import { StatCard } from "../components/StatCard";

function milliseconds(value: number | null): string {
  return value === null ? "—" : `${Math.round(value)} ms`;
}

export function Model() {
  const stats = useModelStats(7);
  const models = stats.data?.models ?? [];
  const maxP95 = models.reduce(
    (highest, model) => Math.max(highest, model.p95_latency_ms ?? 0),
    0,
  );

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Modelo</h2>
        <p className="text-sm text-gray-500 mt-1">
          Telemetría de la inferencia real: cada evento confirmado deja una fila
          con su latencia y su confianza.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <StatCard
          icon={Activity}
          label="Eventos (7 días)"
          value={stats.data?.total_events ?? "—"}
          sub="vehículos confirmados"
        />
        <StatCard
          icon={Cpu}
          label="Modelos activos"
          value={models.length}
          sub="versiones que reportaron"
        />
        <StatCard
          icon={Timer}
          label="Peor p95"
          value={maxP95 > 0 ? milliseconds(maxP95) : "—"}
          sub="latencia por frame"
        />
      </div>

      <div className="card overflow-hidden p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 bg-gray-50">
                <th className="text-left py-3 px-4 font-medium text-gray-500">Modelo</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">Eventos</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">Confianza media</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">p50</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">p95</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {stats.isLoading ? (
                <tr>
                  <td colSpan={5} className="py-12 text-center text-gray-400">
                    Cargando…
                  </td>
                </tr>
              ) : stats.isError ? (
                <tr>
                  <td colSpan={5} className="py-12 text-center text-red-500">
                    No se pudo consultar la telemetría
                  </td>
                </tr>
              ) : models.length > 0 ? (
                models.map((model) => (
                  <tr key={model.model_name} className="hover:bg-gray-50">
                    <td className="py-3 px-4 font-medium text-gray-900">
                      {model.model_name}
                    </td>
                    <td className="py-3 px-4 text-gray-600">{model.events}</td>
                    <td className="py-3 px-4 text-gray-600">
                      {formatConfidence(model.avg_confidence)}
                    </td>
                    <td className="py-3 px-4 text-gray-600">
                      {milliseconds(model.p50_latency_ms)}
                    </td>
                    <td className="py-3 px-4 text-gray-600">
                      {milliseconds(model.p95_latency_ms)}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="py-12 text-center text-gray-400">
                    Todavía no hay eventos confirmados en la ventana.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <p className="text-xs text-gray-400">
        La ventana es de 7 días. La telemetría se conserva según la política de
        retención del sistema.
      </p>
    </div>
  );
}
