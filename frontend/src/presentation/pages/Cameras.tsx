// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

import { useState, type FormEvent } from "react";
import { Pencil, Plus, Trash2 } from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useApi } from "../../application/context";
import { useCameras } from "../../application/queries";
import { formatDateTime } from "../../domain/incident";
import type { CameraGate, CameraReading } from "../../domain/types";

interface FormState {
  camera_id: string;
  gate: CameraGate;
  label: string;
  rtsp_url: string;
  is_active?: boolean;
}

const EMPTY: FormState = {
  camera_id: "",
  gate: "entry",
  label: "",
  rtsp_url: "",
  is_active: true,
};

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : "No se pudo guardar la cámara";
}

export function Cameras() {
  const api = useApi();
  const queryClient = useQueryClient();
  const status = useCameras();
  const [form, setForm] = useState<FormState | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = () =>
    void queryClient.invalidateQueries({ queryKey: ["cameras"] });

  const create = useMutation({
    mutationFn: () =>
      api.createCamera({
        camera_id: form?.camera_id.trim() ?? "",
        gate: form?.gate ?? "entry",
        label: form?.label.trim() || undefined,
        rtsp_url: form?.rtsp_url.trim() ?? "",
        is_active: form?.is_active ?? true,
      }),    onSuccess: () => {
      refresh();
      setForm(null);
    },
    onError: (cause) => setError(errorText(cause)),
  });

  const update = useMutation({
    mutationFn: () =>
      api.updateCamera(editing as string, {
        gate: form?.gate ?? "entry",
        label: form?.label.trim() || undefined,
        rtsp_url: form?.rtsp_url.trim() || undefined,
        is_active: form?.is_active,
      }),
    onSuccess: () => {
      refresh();
      setForm(null);
      setEditing(null);
    },
    onError: (cause) => setError(errorText(cause)),
  });

  const remove = useMutation({
    mutationFn: (cameraId: string) => api.deleteCamera(cameraId),
    onSuccess: refresh,
    onError: (cause) => setError(errorText(cause)),
  });

  function startCreate(camera?: CameraReading) {
    setEditing(null);
    setError(null);
    setForm({
      ...EMPTY,
      camera_id: camera?.camera_id ?? "",
      gate: camera?.gate === "exit" ? "exit" : "entry",
    });
  }

  function startEdit(camera: CameraReading) {
    setEditing(camera.camera_id);
    setError(null);
    setForm({
      camera_id: camera.camera_id,
      gate: camera.gate === "exit" ? "exit" : "entry",
      label: camera.label ?? "",
      rtsp_url: "",
      is_active: undefined,
    });
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    if (editing) update.mutate();
    else create.mutate();
  }

  const cameras = status.data?.cameras ?? [];
  const busy = create.isPending || update.isPending || remove.isPending;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-subtle">
            Administración
          </p>
          <h2 className="font-display text-2xl font-bold tracking-tight text-ink">
            Cámaras
          </h2>
          <p className="mt-1 text-sm text-muted">
            Registrá el portón y la URL RTSP. El capturador publica el heartbeat;
            acá se administra la configuración.
          </p>
        </div>
        <button type="button" onClick={() => startCreate()} className="btn-primary">
          <Plus className="h-4 w-4" />
          Nueva cámara
        </button>
      </div>

      {form && (
        <form onSubmit={submit} className="card space-y-4">
          <div className="flex items-start justify-between">
            <h3 className="text-sm font-semibold text-ink">
              {editing ? `Editar ${editing}` : "Registrar cámara"}
            </h3>
            <button
              type="button"
              onClick={() => {
                setForm(null);
                setEditing(null);
              }}
              className="text-xs text-muted hover:text-ink"
            >
              Cancelar
            </button>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <label className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-muted">Identificador</span>
              <input
                value={form.camera_id}
                onChange={(event) =>
                  setForm({ ...form, camera_id: event.target.value })
                }
                disabled={Boolean(editing)}
                placeholder="gate-1"
                className="field font-mono disabled:opacity-60"
                required
              />
            </label>

            <label className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-muted">Portón</span>
              <select
                value={form.gate}
                onChange={(event) =>
                  setForm({ ...form, gate: event.target.value as CameraGate })
                }
                className="field"
              >
                <option value="entry">Entrada</option>
                <option value="exit">Salida</option>
              </select>
            </label>

            <label className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-muted">Etiqueta</span>
              <input
                value={form.label}
                onChange={(event) => setForm({ ...form, label: event.target.value })}
                placeholder="Portón principal"
                className="field"
              />
            </label>

            <label className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-muted">URL RTSP</span>
              <input
                value={form.rtsp_url}
                onChange={(event) =>
                  setForm({ ...form, rtsp_url: event.target.value })
                }
                placeholder={
                  editing ? "Dejar vacío para conservar la actual" : "rtsp://..."
                }
                className="field font-mono"
                required={!editing}
              />
            </label>

          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-muted">Estado</span>
            <select
              value={
                form.is_active === undefined ? "" : form.is_active ? "true" : "false"
              }
              onChange={(event) =>
                setForm({
                  ...form,
                  is_active:
                    event.target.value === ""
                      ? undefined
                      : event.target.value === "true",
                })
              }
              className="field"
            >
              {editing && <option value="">Sin cambios</option>}
              <option value="true">Activa</option>
              <option value="false">Inactiva</option>
            </select>
          </label>
        </div>

        {error && <p className="text-sm text-danger">{error}</p>}

          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => {
                setForm(null);
                setEditing(null);
              }}
              className="btn-secondary"
            >
              Cancelar
            </button>
            <button type="submit" disabled={busy} className="btn-primary">
              {busy ? "Guardando…" : editing ? "Guardar cambios" : "Registrar"}
            </button>
          </div>
        </form>
      )}

      <div className="card overflow-hidden p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-elevated">
                <th className="text-left py-3 px-4 font-medium text-muted">Cámara</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Portón</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Estado</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Última señal</th>
                <th className="text-right py-3 px-4 font-medium text-muted">Acciones</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {status.isLoading ? (
                <tr>
                  <td colSpan={5} className="py-12 text-center text-subtle">
                    Cargando…
                  </td>
                </tr>
              ) : status.isError ? (
                <tr>
                  <td colSpan={5} className="py-12 text-center text-danger">
                    No se pudo leer el estado de las cámaras
                  </td>
                </tr>
              ) : cameras.length > 0 ? (
                cameras.map((camera) => (
                  <tr key={camera.camera_id} className="hover:bg-elevated">
                    <td className="py-3 px-4">
                      <span className="font-mono text-xs font-medium text-ink">
                        {camera.camera_id}
                      </span>
                      {camera.label && (
                        <span className="block text-xs text-subtle">
                          {camera.label}
                        </span>
                      )}
                      {!camera.registered && (
                        <span className="ml-2 badge-stale">Sin registrar</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-muted">
                      {camera.gate === "exit" ? "Salida" : camera.gate === "entry" ? "Entrada" : "—"}
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`inline-flex items-center gap-2 text-xs font-medium ${
                          camera.online ? "text-live" : "text-danger"
                        }`}
                      >
                        <span
                          className={`h-2 w-2 rounded-full ${
                            camera.online ? "bg-live status-glow" : "bg-danger"
                          }`}
                        />
                        {camera.online ? "En línea" : "Sin señal"}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-xs text-muted">
                      {camera.last_seen ? formatDateTime(camera.last_seen) : "—"}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center justify-end gap-1">
                        {!camera.registered ? (
                          <button
                            type="button"
                            onClick={() => startCreate(camera)}
                            className="btn-secondary px-2.5 py-1.5 text-xs"
                          >
                            <Plus className="h-3 w-3" />
                            Registrar
                          </button>
                        ) : (
                          <>
                            <button
                              type="button"
                              onClick={() => startEdit(camera)}
                              className="p-1.5 rounded-md text-muted hover:bg-elevated hover:text-ink"
                              aria-label={`Editar ${camera.camera_id}`}
                            >
                              <Pencil className="h-4 w-4" />
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                if (window.confirm(`¿Borrar ${camera.camera_id}?`)) {
                                  remove.mutate(camera.camera_id);
                                }
                              }}
                              className="p-1.5 rounded-md text-muted hover:bg-danger/10 hover:text-danger"
                              aria-label={`Borrar ${camera.camera_id}`}
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="py-12 text-center text-subtle">
                    No hay cámaras registradas ni reportando.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {status.data?.redis_available === false && (
        <p className="text-xs text-warn">
          Redis no responde: no se puede ver el heartbeat. La configuración
          registrada sigue disponible.
        </p>
      )}

      <p className="text-xs text-subtle">
        El identificador no se puede cambiar una vez registrado. El portón sí se
        puede reasignar; si editás la URL RTSP y dejás el campo vacío, se conserva
        la actual.
      </p>
    </div>
  );
}
