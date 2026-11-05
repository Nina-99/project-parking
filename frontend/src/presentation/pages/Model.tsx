// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

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
  const primary = models[0];
  const maxP95 = models.reduce(
    (highest, model) => Math.max(highest, model.p95_latency_ms ?? 0),
    0,
  );
  const latencyScale = Math.max(200, primary?.p95_latency_ms ?? 0);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="font-display text-2xl font-bold text-ink">Modelo</h2>
        <p className="text-sm text-muted mt-1">
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
              <tr className="border-b border-border bg-elevated">
                <th className="text-left py-3 px-4 font-medium text-muted">Modelo</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Eventos</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Confianza media</th>
                <th className="text-left py-3 px-4 font-medium text-muted">p50</th>
                <th className="text-left py-3 px-4 font-medium text-muted">p95</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {stats.isLoading ? (
                <tr>
                  <td colSpan={5} className="py-12 text-center text-subtle">
                    Cargando…
                  </td>
                </tr>
              ) : stats.isError ? (
                <tr>
                  <td colSpan={5} className="py-12 text-center text-danger">
                    No se pudo consultar la telemetría
                  </td>
                </tr>
              ) : models.length > 0 ? (
                models.map((model) => (
                  <tr key={model.model_name} className="hover:bg-elevated">
                    <td className="py-3 px-4 font-medium text-ink">
                      {model.model_name}
                    </td>
                    <td className="py-3 px-4 text-muted">{model.events}</td>
                    <td className="py-3 px-4 text-muted">
                      {formatConfidence(model.avg_confidence)}
                    </td>
                    <td className="py-3 px-4 text-muted">
                      {milliseconds(model.p50_latency_ms)}
                    </td>
                    <td className="py-3 px-4 text-muted">
                      {milliseconds(model.p95_latency_ms)}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="py-12 text-center text-subtle">
                    Todavía no hay eventos confirmados en la ventana.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <p className="text-xs text-subtle">
        La ventana es de 7 días. La telemetría se conserva según la política de
        retención del sistema.
      </p>

      {primary && (
        <div className="grid gap-4 lg:grid-cols-2">
          <section className="card space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-ink">Latencia por frame</h3>
              <span className="font-mono text-xs text-subtle">
                {primary.model_name}
              </span>
            </div>
            {[
              { label: "p50", value: primary.p50_latency_ms },
              { label: "p95", value: primary.p95_latency_ms },
            ].map((row) => (
              <div key={row.label} className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted">{row.label}</span>
                  <span className="font-mono tabular-nums text-ink">
                    {milliseconds(row.value)}
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-elevated">
                  <div
                    className="h-2 rounded-full bg-accent"
                    style={{
                      width: `${Math.min(
                        100,
                        ((row.value ?? 0) / latencyScale) * 100,
                      )}%`,
                    }}
                  />
                </div>
              </div>
            ))}
            <p className="text-xs text-subtle">
              YOLO + clasificadores, medido alrededor de cada evento confirmado.
            </p>
          </section>

          <section className="card flex flex-col justify-between gap-4">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-sm font-semibold text-ink">Confianza media</h3>
                <p className="mt-1 text-xs text-subtle">
                  Detección de los eventos confirmados
                </p>
              </div>
              <span className="font-display text-4xl font-semibold tabular-nums text-ink">
                {formatConfidence(primary.avg_confidence)}
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-elevated">
              <div
                className="h-2 rounded-full bg-accent"
                style={{
                  width: `${Math.min(100, (primary.avg_confidence ?? 0) * 100)}%`,
                }}
              />
            </div>
            <p className="text-xs text-subtle">
              Una confianza baja no se descarta: el evento queda registrado y se
              ve en el detalle del vehículo.
            </p>
          </section>
        </div>
      )}
    </div>
  );
}
