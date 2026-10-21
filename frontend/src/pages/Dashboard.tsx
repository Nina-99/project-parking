import { useQuery } from "@tanstack/react-query";
import {
  Car,
  Clock,
  TrendingUp,
  AlertTriangle,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  getActiveSessions,
  getOccupancy,
  getDailyStats,
  type ActiveSession,
} from "../services/api";

function StatCard({
  icon: Icon,
  label,
  value,
  sub,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string | number;
  sub?: string;
}) {
  return (
    <div className="stat-card">
      <div className="flex items-center justify-between">
        <span className="stat-label">{label}</span>
        <Icon className="w-5 h-5 text-gray-400" />
      </div>
      <span className="stat-value">{value}</span>
      {sub && <span className="text-xs text-gray-400">{sub}</span>}
    </div>
  );
}

function ActiveVehicleRow({
  session,
}: {
  session: ActiveSession;
}) {
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
        <p className="text-xs text-gray-500">
          {new Date(session.entry_time).toLocaleTimeString("es-BO")}
        </p>
        <p className="text-xs text-gray-400">
          {session.entry_gate || "N/A"}
        </p>
      </div>
    </div>
  );
}

export function Dashboard() {
  const { data: occupancy } = useQuery({
    queryKey: ["occupancy"],
    queryFn: getOccupancy,
  });

  const { data: activeSessions } = useQuery({
    queryKey: ["active-sessions"],
    queryFn: getActiveSessions,
  });

  const { data: dailyStats } = useQuery({
    queryKey: ["daily-stats"],
    queryFn: () => getDailyStats(7),
  });

  const chartData = (dailyStats ?? []).map((d) => ({
    date: new Date(d.date).toLocaleDateString("es-BO", {
      weekday: "short",
      day: "numeric",
    }),
    vehiculos: d.total_vehicles,
  }));

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Dashboard</h2>

      {/* Stats grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={Car}
          label="Vehiculos en parqueadero"
          value={occupancy?.active_sessions ?? 0}
          sub="ocupados actualmente"
        />
        <StatCard
          icon={TrendingUp}
          label="Ingresos hoy"
          value={
            dailyStats?.length
              ? dailyStats[dailyStats.length - 1].total_vehicles
              : 0
          }
          sub="último día"
        />
        <StatCard
          icon={Clock}
          label="Promedio estadía"
          value={
            dailyStats?.length
              ? `${(dailyStats[dailyStats.length - 1].avg_duration_minutes ?? 0).toFixed(0)} min`
              : "—"
          }
        />
        <StatCard
          icon={AlertTriangle}
          label="Capacidad"
          value="—"
          sub="sin configurar"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Active vehicles */}
        <div className="card lg:col-span-1">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Vehículos activos
          </h3>
          <div className="space-y-1 max-h-96 overflow-y-auto">
            {activeSessions && activeSessions.length > 0 ? (
              activeSessions.map((s) => (
                <ActiveVehicleRow key={s.id} session={s} />
              ))
            ) : (
              <p className="text-sm text-gray-400 text-center py-8">
                No hay vehículos en el parqueadero
              </p>
            )}
          </div>
        </div>

        {/* Chart */}
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
                <Bar
                  dataKey="vehiculos"
                  fill="#3b82f6"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-gray-400 text-center py-12">
              Sin datos para mostrar
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
