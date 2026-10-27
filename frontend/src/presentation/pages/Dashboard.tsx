// SPDX-FileCopyrightText: 2026 Nina
// SPDX-License-Identifier: LicenseRef-Proprietary

import { useState } from "react";
import { Camera, Car, Clock, Timer, TrendingUp } from "lucide-react";
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
  useCameras,
  useDailyStats,
  useModelStats,
  useOccupancy,
} from "../../application/queries";
import { useSession } from "../../application/session";
import { formatClock, formatTime } from "../../domain/incident";
import type { ActiveSession } from "../../domain/types";
import { CameraTile } from "../components/CameraTile";
import { CloseSessionDialog } from "../components/CloseSessionDialog";
import { EventFeed } from "../components/EventFeed";
import { OccupancyGauge } from "../components/OccupancyGauge";
import { StatCard } from "../components/StatCard";

function ActiveRow({
  session,
  onClose,
  canClose,
}: {
  session: ActiveSession;
  onClose: () => void;
  canClose: boolean;
}) {
  return (
    <li className="flex items-center justify-between gap-3 border-b border-border px-5 py-3 last:border-0">
      <div className="flex min-w-0 items-center gap-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent-soft text-accent">
          <Car className="h-4 w-4" />
        </div>
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-ink">
            {session.license_plate || "Sin placa"}
          </p>
          <p className="truncate text-xs text-muted">
            {session.vehicle_type || "?"} · {session.color || "?"} ·{" "}
            {session.entry_gate || "sin portón"}
          </p>
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {session.stale && <span className="badge-stale">Huérfana</span>}
        <div className="text-right">
          <p className="font-mono text-[11px] tabular-nums text-muted">
            {formatClock(session.entry_time)}
          </p>
          <p className="text-[11px] text-subtle">{session.age_minutes} min</p>
        </div>
        {canClose && (
          <button
            type="button"
            onClick={onClose}
            className="btn-secondary px-2.5 py-1.5 text-xs"
          >
            Cerrar
          </button>
        )}
      </div>
    </li>
  );
}

export function Dashboard() {
  const occupancy = useOccupancy();
  const active = useActiveSessions();
  const daily = useDailyStats(7);
  const cameras = useCameras();
  const model = useModelStats(1);
  const { user } = useSession();
  const canClose = user?.role === "admin" || user?.role === "operator";
  const [closing, setClosing] = useState<ActiveSession | null>(null);

  const chartData = (daily.data ?? []).map((day) => ({
    date: new Date(day.date).toLocaleDateString("es-BO", {
      weekday: "short",
      day: "numeric",
    }),
    vehiculos: day.total_vehicles,
  }));

  const lastDay = daily.data?.[daily.data.length - 1];
  const occupancyData = occupancy.data;
  const cameraList = cameras.data?.cameras ?? [];
  const online = cameraList.filter((camera) => camera.online).length;
  const p95 = (model.data?.models ?? []).reduce(
    (highest, item) => Math.max(highest, item.p95_latency_ms ?? 0),
    0,
  );

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-subtle">
            Operación
          </p>
          <h2 className="font-display text-2xl font-bold tracking-tight text-ink">
            Dashboard
          </h2>
        </div>
        <p className="text-sm text-muted">
          Ocupación, cámaras y evidencia en tiempo real
        </p>
      </header>

      <div className="grid gap-4 lg:grid-cols-12">
        <section className="card flex flex-col items-center justify-center gap-4 lg:col-span-4">
          <OccupancyGauge
            active={occupancyData?.active_sessions ?? 0}
            capacity={occupancyData?.capacity ?? 0}
            percent={occupancyData?.percent ?? null}
          />
          <div className="text-center">
            <p className="text-sm font-medium text-ink">
              Vehículos dentro del recinto
            </p>
            <p className="mt-1 inline-flex items-center gap-2 text-xs text-subtle">
              <span className="live-dot" />
              {occupancyData
                ? `Actualizado ${formatTime(occupancyData.timestamp)}`
                : "Consultando…"}
            </p>
          </div>
        </section>

        <div className="grid gap-4 sm:grid-cols-2 lg:col-span-8">
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
          <StatCard
            icon={Timer}
            label="Latencia p95"
            value={p95 > 0 ? `${Math.round(p95)} ms` : "—"}
            sub="inferencia por frame"
          />
          <StatCard
            icon={Camera}
            label="Cámaras en línea"
            value={cameras.isLoading ? "…" : `${online}/${cameraList.length}`}
            sub="heartbeat de captura"
          />
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {cameraList.map((camera) => (
          <CameraTile key={camera.camera_id} camera={camera} />
        ))}
        {!cameras.isLoading && cameraList.length === 0 && (
          <p className="text-sm text-subtle sm:col-span-2">
            Ninguna cámara reportó todavía. El heartbeat se publica cada 5 segundos
            desde cada capturador.
          </p>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <section className="card flex max-h-[26rem] flex-col p-0 lg:col-span-2">
          <header className="flex items-center justify-between border-b border-border px-5 py-4">
            <h3 className="text-sm font-semibold text-ink">Vehículos activos</h3>
            <span className="font-mono text-xs tabular-nums text-subtle">
              {active.data?.length ?? 0} dentro
            </span>
          </header>
          <div className="flex-1 overflow-y-auto">
            {active.isLoading ? (
              <p className="py-10 text-center text-sm text-subtle">Cargando…</p>
            ) : active.data && active.data.length > 0 ? (
              <ul>
                {active.data.map((session) => (
                  <ActiveRow
                    key={session.id}
                    session={session}
                    onClose={() => setClosing(session)}
                    canClose={canClose}
                  />
                ))}
              </ul>
            ) : (
              <div className="flex h-40 flex-col items-center justify-center gap-1 text-center">
                <Car className="h-5 w-5 text-subtle" />
                <p className="text-sm text-muted">No hay vehículos dentro</p>
                <p className="text-xs text-subtle">
                  Las entradas abren una sesión; las salidas la cierran.
                </p>
              </div>
            )}
          </div>
        </section>

        <EventFeed />
      </div>

      <section className="card">
        <div className="mb-4 flex items-end justify-between">
          <div>
            <h3 className="text-sm font-semibold text-ink">
              Ingresos últimos 7 días
            </h3>
            <p className="text-xs text-subtle">Vehículos confirmados por día</p>
          </div>
        </div>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 12, fill: "var(--color-muted)" }}
                axisLine={{ stroke: "var(--color-border)" }}
                tickLine={false}
              />
              <YAxis
                allowDecimals={false}
                tick={{ fontSize: 12, fill: "var(--color-muted)" }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--color-surface)",
                  border: "1px solid var(--color-border)",
                  borderRadius: 8,
                  color: "var(--color-ink)",
                }}
                labelStyle={{ color: "var(--color-muted)" }}
                itemStyle={{ color: "var(--color-ink)" }}
              />
              <Bar
                dataKey="vehiculos"
                fill="var(--color-accent)"
                radius={[6, 6, 0, 0]}
                maxBarSize={42}
              />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="py-12 text-center text-sm text-subtle">
            {daily.isLoading ? "Cargando…" : "Sin datos para mostrar"}
          </p>
        )}
      </section>

      {closing && (
        <CloseSessionDialog
          session={closing}
          onDismiss={() => setClosing(null)}
        />
      )}
    </div>
  );
}
