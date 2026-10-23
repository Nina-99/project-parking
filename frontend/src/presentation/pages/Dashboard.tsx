import { Car, Clock, TrendingUp } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  useActiveSessions,
  useDailyStats,
  useOccupancy,
} from "../../application/queries";
import { formatTime } from "../../domain/incident";
import type { ActiveSession } from "../../domain/types";
import { StatCard } from "../components/StatCard";

function ActiveRow({ session }: { session: ActiveSession }) {
  return (
    <div className="flex items-center justify-between py-3 px-4 hover:bg-gray-50 rounded-lg transition-colors">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-primary-100 rounded-full flex items-center justify-center">
          <Car className="w-5 h-5 text-primary-600" />
        </div>
        <div>
          <p className="text-sm font-medium text-gray-900">
            {session.license_plate || "Sin placa"}
          </p>
          <p className="text-xs text-gray-500">
            {session.vehicle_type || "?"} · {session.color || "?"}
          </p>
        </div>
      </div>
      <div className="text-right">
        <p className="text-xs text-gray-500">{formatTime(session.entry_time)}</p>
        <p className="text-xs text-gray-400">{session.entry_gate || "N/A"}</p>
      </div>
    </div>
  );
}

export function Dashboard() {
  const occupancy = useOccupancy();
  const active = useActiveSessions();
  const daily = useDailyStats(7);

  const chartData = (daily.data ?? []).map((day) => ({
    date: new Date(day.date).toLocaleDateString("es-BO", {
      weekday: "short",
      day: "numeric",
    }),
    vehiculos: day.total_vehicles,
  }));

  const lastDay = daily.data?.[daily.data.length - 1];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Dashboard</h2>
        {occupancy.isError && (
          <span className="text-sm text-red-600">
            No se pudo consultar la ocupación
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <StatCard
          icon={Car}
          label="Vehículos en parqueadero"
          value={occupancy.data?.active_sessions ?? "—"}
          sub="ocupados actualmente"
        />
        <StatCard
          icon={TrendingUp}
          label="Ingresos del último día"
          value={lastDay?.total_vehicles ?? "—"}
          sub="según el último registro"
        />
        <StatCard
          icon={Clock}
          label="Promedio de estadía"
          value={
            lastDay?.avg_duration_minutes != null
              ? `${Math.round(lastDay.avg_duration_minutes)} min`
              : "—"
          }
          sub="último día"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card lg:col-span-1">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Vehículos activos
          </h3>
          <div className="space-y-1 max-h-96 overflow-y-auto">
            {active.isLoading ? (
              <p className="text-sm text-gray-400 text-center py-8">Cargando…</p>
            ) : active.data && active.data.length > 0 ? (
              active.data.map((session) => (
                <ActiveRow key={session.id} session={session} />
              ))
            ) : (
              <p className="text-sm text-gray-400 text-center py-8">
                No hay vehículos en el parqueadero
              </p>
            )}
          </div>
        </div>

        <div className="card lg:col-span-2">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Ingresos últimos 7 días
          </h3>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="vehiculos" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-gray-400 text-center py-12">
              {daily.isLoading ? "Cargando…" : "Sin datos para mostrar"}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
