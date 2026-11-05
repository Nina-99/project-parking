// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

/** Decision quality: the operational measure of the OCR gap.

 * The frozen-set evaluation says how the model performs on labelled data. This
 * card says how often a person had to correct it on the lot. Both are needed:
 * only the second one describes the system that is running tonight.
 */
import { ClipboardCheck, DoorOpen, PencilLine, ScanText } from "lucide-react";
import { useQualityStats } from "../../application/queries";
import type { QualityStats } from "../../domain/types";

function rate(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function Metric({
  icon: Icon,
  label,
  value,
  sub,
}: {
  icon: typeof ScanText;
  label: string;
  value: string | number;
  sub: string;
}) {
  return (
    <div className="rounded-lg border border-border bg-canvas/40 px-4 py-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted">{label}</span>
        <Icon className="h-4 w-4 text-subtle" />
      </div>
      <p className="mt-1 font-display text-2xl font-semibold tracking-tight text-ink tabular-nums">
        {value}
      </p>
      <p className="text-[11px] text-subtle">{sub}</p>
    </div>
  );
}

function OperatorTable({ data }: { data: QualityStats }) {
  if (data.operators.length === 0) {
    return (
      <p className="text-xs text-subtle">
        Todavía no hay revisiones registradas en la ventana.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left">
            <th className="py-2 pr-4 font-medium text-muted">Operador</th>
            <th className="py-2 pr-4 text-right font-medium text-muted">Revisiones</th>
            <th className="py-2 pr-4 text-right font-medium text-muted">Correcciones</th>
            <th className="py-2 text-right font-medium text-muted">Tasa</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {data.operators.map((operator) => (
            <tr key={operator.operator}>
              <td className="py-2 pr-4 font-mono text-xs text-ink">
                {operator.operator}
              </td>
              <td className="py-2 pr-4 text-right tabular-nums text-muted">
                {operator.decisions}
              </td>
              <td className="py-2 pr-4 text-right tabular-nums text-muted">
                {operator.corrected}
              </td>
              <td className="py-2 text-right tabular-nums text-ink">
                {rate(
                  operator.decisions > 0
                    ? operator.corrected / operator.decisions
                    : null,
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function QualityCard({ days = 7 }: { days?: number }) {
  const quality = useQualityStats(days);

  if (quality.isLoading) {
    return (
      <section className="card">
        <p className="py-6 text-center text-sm text-subtle">Cargando calidad…</p>
      </section>
    );
  }

  if (quality.isError || !quality.data) {
    return (
      <section className="card">
        <p className="py-6 text-center text-sm text-danger">
          No se pudo leer la calidad de decisión.
        </p>
      </section>
    );
  }

  const data = quality.data;

  return (
    <section className="card space-y-4">
      <div className="flex items-end justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-ink">Calidad de decisión</h3>
          <p className="text-xs text-subtle">
            Cómo se comporta el sistema en operación, no en el conjunto congelado
          </p>
        </div>
        <span className="font-mono text-[11px] tabular-nums text-subtle">
          últimos {data.window_days} días
        </span>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Metric
          icon={ScanText}
          label="Placas legibles"
          value={rate(data.plate_read_rate)}
          sub={`${data.events_with_plate} de ${data.events_total} eventos`}
        />
        <Metric
          icon={PencilLine}
          label="Correcciones"
          value={rate(data.correction_rate)}
          sub="sobre eventos revisados"
        />
        <Metric
          icon={DoorOpen}
          label="Salidas sin match"
          value={data.unmatched_exits}
          sub="cerradas a mano o pendientes"
        />
        <Metric
          icon={ClipboardCheck}
          label="Revisiones"
          value={data.decisions_total}
          sub="decisiones auditadas"
        />
      </div>

      <OperatorTable data={data} />

      <p className="text-[11px] text-subtle">
        Una placa ilegible no es un error del operador ni una coincidencia: es un
        evento que el sistema registró sin identificador. La tasa de corrección
        solo cuenta lo que alguien revisó.
      </p>
    </section>
  );
}
