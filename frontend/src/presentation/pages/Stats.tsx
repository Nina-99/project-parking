// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

/** Reports for the roles that read the lot instead of operating it.

 * The dashboard answers "what is happening now". This page answers "what
 * happened in the window": traffic, stay, hourly peaks, colour distribution and
 * the quality of the plate read. It is read-only and does not expose inference
 * telemetry, which stays admin-only.
 */
import { useState } from "react";
import { Activity, Car, Clock, ScanText } from "lucide-react";
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
  useColourStats,
  useDailyStats,
  useHourlyStats,
  useQualityStats,
} from "../../application/queries";
import { QualityCard } from "../components/QualityCard";
import { StatCard } from "../components/StatCard";

const RANGES = [7, 14, 30] as const;

const COLOUR_LABELS: Record<string, string> = {
  blanco: "Blanco",
  plata: "Plata",
  rojo: "Rojo",
  gris: "Gris",
  negro: "Negro",
  azul: "Azul",
  minority: "Otros",
};

const COLOUR_SWATCH: Record<string, string> = {
  blanco: "#e2e8f0",
  plata: "#94a3b8",
  rojo: "#b91c1c",
  gris: "#64748b",
  negro: "#0f172a",
  azul: "#1d4ed8",
  minority: "#a855f7",
};

function percent(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

const chartTooltip = {
  background: "var(--color-surface)",
  border: "1px solid var(--color-border)",
  borderRadius: 8,
  color: "var(--color-ink)",
};

export function Stats() {
  const [days, setDays] = useState<number>(7);
  const daily = useDailyStats(days);
  const hourly = useHourlyStats(days);
  const colours = useColourStats(days);
  const quality = useQualityStats(days);

  const dailyRows = daily.data ?? [];
  const totalVehicles = dailyRows.reduce(
    (sum, day) => sum + day.total_vehicles,
    0,
  );
  const weightedDays = dailyRows.filter(
    (day) => day.avg_duration_minutes !== null,
  );
  const weightedTotal = weightedDays.reduce(
    (sum, day) => sum + day.total_vehicles,
    0,
  );
  const averageStay =
    weightedTotal > 0
      ? weightedDays.reduce(
          (sum, day) =>
            sum + (day.avg_duration_minutes ?? 0) * day.total_vehicles,
          0,
        ) / weightedTotal
      : null;

  const peak = (hourly.data?.hours ?? []).reduce(
    (best, point) => (point.vehicles > best.vehicles ? point : best),
    { hour: 0, vehicles: -1 },
  );
  const peakHour =
    peak.vehicles > 0 ? `${String(peak.hour).padStart(2, "0")}:00` : "—";

  const dailyChart = dailyRows.map((day) => ({
    label: new Date(`${day.date}T00:00:00`).toLocaleDateString("es-BO", {
      day: "numeric",
      month: "short",
    }),
    vehiculos: day.total_vehicles,
  }));

  const hourlyChart = (hourly.data?.hours ?? []).map((point) => ({
    label: String(point.hour).padStart(2, "0"),
    vehiculos: point.vehicles,
  }));

  const colourTotals = colours.data?.totals;
  const colourKeys = [
    ...(colours.data?.reportable_colors ?? []),
    "minority",
  ];
  const colourTotal = colours.data?.total_vehicles ?? 0;

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-subtle">
            Reportes
          </p>
          <h2 className="font-display text-2xl font-bold tracking-tight text-gradient">
            Estadísticas
          </h2>
          <p className="mt-1 text-sm text-muted">
            Afluencia, permanencia, distribución horaria y calidad de lectura.
          </p>
        </div>

        <div className="inline-flex shrink-0 rounded-lg border border-border bg-elevated p-1">
          {RANGES.map((range) => (
            <button
              key={range}
              type="button"
              onClick={() => setDays(range)}
              className={`rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${
                days === range
                  ? "bg-surface text-ink shadow-sm"
                  : "text-muted hover:text-ink"
              }`}
            >
              {range} días
            </button>
          ))}
        </div>
      </header>

      <div className="stagger grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          icon={Car}
          label="Ingresos"
          value={daily.isLoading ? "…" : totalVehicles}
          sub={`en los últimos ${days} días`}
        />
        <StatCard
          icon={Clock}
          label="Estadía promedio"
          value={
            averageStay !== null ? `${Math.round(averageStay)} min` : "—"
          }
          sub="ponderada por ingresos"
        />
        <StatCard
          icon={Activity}
          label="Hora pico"
          value={peakHour}
          sub="mayor afluencia del día"
        />
        <StatCard
          icon={ScanText}
          label="Placas legibles"
          value={percent(quality.data?.plate_read_rate ?? null)}
          sub="sobre eventos registrados"
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="card">
          <div className="mb-4 flex items-end justify-between">
            <div>
              <h3 className="text-sm font-semibold text-ink">
                Ingresos por día
              </h3>
              <p className="text-xs text-subtle">Vehículos confirmados</p>
            </div>
            <span className="font-mono text-[11px] tabular-nums text-subtle">
              {days} días
            </span>
          </div>
          {dailyChart.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={dailyChart}>
                <defs>
                  <linearGradient id="stats-daily" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--color-accent)" stopOpacity={0.95} />
                    <stop offset="100%" stopColor="var(--color-accent)" stopOpacity={0.26} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis
                  dataKey="label"
                  tick={{ fontSize: 11, fill: "var(--color-muted)" }}
                  axisLine={{ stroke: "var(--color-border)" }}
                  tickLine={false}
                />
                <YAxis
                  allowDecimals={false}
                  tick={{ fontSize: 11, fill: "var(--color-muted)" }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={chartTooltip}
                  labelStyle={{ color: "var(--color-muted)" }}
                  itemStyle={{ color: "var(--color-ink)" }}
                />
                <Bar
                  dataKey="vehiculos"
                  fill="url(#stats-daily)"
                  radius={[6, 6, 0, 0]}
                  maxBarSize={34}
                />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="py-16 text-center text-sm text-subtle">
              {daily.isLoading ? "Cargando…" : "Sin ingresos en la ventana."}
            </p>
          )}
        </section>

        <section className="card">
          <div className="mb-4 flex items-end justify-between">
            <div>
              <h3 className="text-sm font-semibold text-ink">
                Distribución horaria
              </h3>
              <p className="text-xs text-subtle">
                Ingresos por hora del día · hora local
              </p>
            </div>
            <span className="font-mono text-[11px] text-subtle">
              {hourly.data?.timezone ?? "local"}
            </span>
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={hourlyChart}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis
                dataKey="label"
                interval={2}
                tick={{ fontSize: 10, fill: "var(--color-muted)" }}
                axisLine={{ stroke: "var(--color-border)" }}
                tickLine={false}
              />
              <YAxis
                allowDecimals={false}
                tick={{ fontSize: 11, fill: "var(--color-muted)" }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                contentStyle={chartTooltip}
                labelFormatter={(value) => `${value}:00`}
                labelStyle={{ color: "var(--color-muted)" }}
                itemStyle={{ color: "var(--color-ink)" }}
              />
              <Bar
                dataKey="vehiculos"
                fill="var(--color-accent)"
                radius={[4, 4, 0, 0]}
                maxBarSize={14}
              />
            </BarChart>
          </ResponsiveContainer>
        </section>
      </div>

      <section className="card space-y-4">
        <div className="flex items-end justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-ink">
              Distribución por color
            </h3>
            <p className="text-xs text-subtle">
              Solo etiquetas con confianza ≥{" "}
              {colours.data?.min_confidence ?? 0.3}
            </p>
          </div>
          <span className="font-mono text-[11px] tabular-nums text-subtle">
            {colourTotal} vehículos
          </span>
        </div>

        {colourTotal === 0 ? (
          <p className="py-8 text-center text-sm text-subtle">
            {colours.isLoading
              ? "Cargando…"
              : "Sin observaciones de color en la ventana."}
          </p>
        ) : (
          <div className="grid gap-3">
            {colourKeys.map((key) => {
              const value = colourTotals
                ? (colourTotals as unknown as Record<string, number>)[key] ?? 0
                : 0;
              const share = colourTotal > 0 ? (value / colourTotal) * 100 : 0;
              return (
                <div key={key} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="flex items-center gap-2 text-muted">
                      <span
                        className="h-2.5 w-2.5 rounded-full ring-1 ring-border"
                        style={{ background: COLOUR_SWATCH[key] }}
                      />
                      {COLOUR_LABELS[key] ?? key}
                    </span>
                    <span className="font-mono tabular-nums text-ink">
                      {value}{" "}
                      <span className="text-subtle">{share.toFixed(1)}%</span>
                    </span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-elevated">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${share}%`,
                        background: COLOUR_SWATCH[key],
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        )}
        <p className="text-[11px] text-subtle">
          Las etiquetas poco confiables no se adivinan: se excluyen del conteo.
          “Otros” agrupa los colores con muestra insuficiente.
        </p>
      </section>

      <QualityCard days={days} />
    </div>
  );
}
