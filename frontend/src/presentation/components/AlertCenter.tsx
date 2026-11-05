// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

/** Operational alerts the operator should not have to go looking for.

 * Three conditions, one surface: an unauthorized vehicle at the gate, a session
 * that has been open longer than the configured threshold, and a camera that
 * stopped reporting. All three are derived from queries the app already makes,
 * so this panel adds no endpoint and no polling of its own.
 */
import { useState } from "react";
import { Bell, ShieldAlert, Timer, VideoOff } from "lucide-react";
import { Link } from "react-router-dom";
import {
  useActiveSessions,
  useCameras,
  usePendingValidations,
} from "../../application/queries";
import { useSession } from "../../application/session";

interface Alert {
  id: string;
  severity: "danger" | "warn";
  icon: typeof ShieldAlert;
  title: string;
  detail: string;
  to?: string;
  hint?: string;
}

export function AlertCenter() {
  const { user } = useSession();
  const pending = usePendingValidations();
  const active = useActiveSessions();
  const cameras = useCameras();
  const [open, setOpen] = useState(false);
  const canOperate = user?.role === "admin" || user?.role === "operator";

  const alerts: Alert[] = [];

  for (const item of pending.data ?? []) {
    if (item.authorized !== false) continue;
    alerts.push({
      id: `unauthorized-${item.vehicle_id}`,
      severity: "danger",
      icon: ShieldAlert,
      title: "Vehículo no autorizado",
      detail: `${item.license_plate ?? "sin placa"} · ${item.camera_id ?? "sin cámara"}`,
      hint: canOperate ? "Revisá la bandeja de validación." : undefined,
    });
  }

  for (const session of active.data ?? []) {
    if (!session.stale) continue;
    alerts.push({
      id: `stale-${session.id}`,
      severity: "warn",
      icon: Timer,
      title: "Sesión huérfana",
      detail: `${session.license_plate ?? "sin placa"} · ${session.age_minutes} min dentro`,
      hint: canOperate
        ? "Cerrá la sesión desde el panel si el vehículo ya salió."
        : undefined,
    });
  }

  for (const camera of cameras.data?.cameras ?? []) {
    if (camera.online) continue;
    alerts.push({
      id: `camera-${camera.camera_id}`,
      severity: "warn",
      icon: VideoOff,
      title: "Cámara sin señal",
      detail: camera.label ?? camera.camera_id,
      to: user?.role === "admin" ? "/cameras" : undefined,
      hint: user?.role === "admin" ? "Ver configuración de cámaras." : undefined,
    });
  }

  const dangerous = alerts.filter((alert) => alert.severity === "danger").length;
  const tone = dangerous > 0 ? "danger" : alerts.length > 0 ? "warn" : "ok";

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className={`relative rounded-lg p-2 transition-colors hover:bg-elevated ${
          tone === "danger"
            ? "pulse-ring text-danger"
            : tone === "warn"
              ? "text-warn"
              : "text-muted hover:text-ink"
        }`}
        aria-label={`Alertas operativas (${alerts.length})`}
        aria-expanded={open}
      >
        <Bell className="h-4 w-4" />
        {alerts.length > 0 && (
          <span
            className={`absolute -right-0.5 -top-0.5 inline-flex min-w-[16px] items-center justify-center rounded-full px-1 text-[10px] font-bold text-white ${
              dangerous > 0 ? "bg-danger" : "bg-warn"
            }`}
          >
            {alerts.length}
          </span>
        )}
      </button>

      {open && (
        <>
          <button
            type="button"
            aria-label="Cerrar alertas"
            className="fixed inset-0 z-40 cursor-default"
            onClick={() => setOpen(false)}
          />
          <div
            role="dialog"
            aria-label="Alertas operativas"
            className="surface-glass dropdown-in absolute right-0 top-11 z-50 w-80 overflow-hidden rounded-xl border border-border shadow-2xl"
          >
            <header className="flex items-center justify-between border-b border-border px-4 py-3">
              <h3 className="text-sm font-semibold text-ink">Alertas operativas</h3>
              <span className="font-mono text-xs tabular-nums text-subtle">
                {alerts.length}
              </span>
            </header>

            <div className="max-h-80 overflow-y-auto">
              {alerts.length === 0 ? (
                <p className="px-4 py-10 text-center text-sm text-subtle">
                  Nada requiere atención ahora.
                </p>
              ) : (
                <ul className="divide-y divide-border">
                  {alerts.map((alert) => {
                    const Icon = alert.icon;
                    const body = (
                      <div className="flex items-start gap-3 px-4 py-3 text-left">
                        <Icon
                          className={`mt-0.5 h-4 w-4 shrink-0 ${
                            alert.severity === "danger" ? "text-danger" : "text-warn"
                          }`}
                        />
                        <div className="min-w-0">
                          <p className="text-sm font-medium text-ink">{alert.title}</p>
                          <p className="truncate text-xs text-muted">{alert.detail}</p>
                          {alert.hint && (
                            <p className="mt-0.5 text-[11px] text-subtle">{alert.hint}</p>
                          )}
                        </div>
                      </div>
                    );

                    return (
                      <li key={alert.id}>
                        {alert.to ? (
                          <Link
                            to={alert.to}
                            onClick={() => setOpen(false)}
                            className="block hover:bg-elevated"
                          >
                            {body}
                          </Link>
                        ) : (
                          body
                        )}
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
