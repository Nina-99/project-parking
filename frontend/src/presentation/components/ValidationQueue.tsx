// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

/** The human half of RF-10: review what the gate just decided.

 * The machine confirms a crossing; the operator confirms the machine. Entries,
 * unmatched exits and unauthorized plates all land here. The queue is fed by the
 * server (so a reload does not lose work) and refreshed by the socket (so it
 * does not need a reload).
 */
import { useEffect, useRef, useState } from "react";
import {
  Check,
  ClipboardCheck,
  ImageOff,
  Pencil,
  ShieldAlert,
  ShieldCheck,
  X,
} from "lucide-react";
import {
  useActiveSessions,
  useCloseSession,
  usePendingValidations,
  useSubmitValidationDecision,
} from "../../application/queries";
import { useLiveEvents } from "../../application/liveEvents";
import { useSession } from "../../application/session";
import { formatConfidence, formatDateTime } from "../../domain/incident";
import type {
  PendingValidation,
  ValidationDecisionKind,
} from "../../domain/types";
import { PlateCharacters } from "./PlateCharacters";

function directionLabel(direction: PendingValidation["direction"]): string {
  if (direction === "entry") return "Ingreso";
  if (direction === "exit") return "Salida";
  return "Sin geometría";
}

function AuthorizationBadge({ authorized }: { authorized: boolean | null }) {
  if (authorized === null) {
    return <span className="badge-completed text-[10px]">Sin placa</span>;
  }
  if (authorized) {
    return (
      <span className="badge-active inline-flex items-center gap-1 text-[10px]">
        <ShieldCheck className="h-3 w-3" /> En padrón
      </span>
    );
  }
  return (
    <span className="badge-danger inline-flex items-center gap-1 text-[10px]">
      <ShieldAlert className="h-3 w-3" /> No autorizado
    </span>
  );
}

function ValidationCard({ item }: { item: PendingValidation }) {
  const submit = useSubmitValidationDecision();
  const closeSession = useCloseSession();
  const active = useActiveSessions();

  const [correcting, setCorrecting] = useState(false);
  const [plate, setPlate] = useState(item.license_plate ?? "");
  const [selected, setSelected] = useState<number | null>(null);
  const [closing, setClosing] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [error, setError] = useState<string | null>(null);

  const characters = item.plate_characters ?? [];
  // Only aligned when the edition still has one entry per decoded character:
  // after a manual edit the positions no longer describe what the model read.
  const charactersAligned =
    characters.length > 0 && characters.length === plate.length;

  function pickAlternative(index: number, char: string) {
    setPlate((current) => current.slice(0, index) + char + current.slice(index + 1));
    setSelected(null);
  }

  const busy = submit.isPending || closeSession.isPending;

  async function decide(
    decision: ValidationDecisionKind,
    extra?: { corrected_plate?: string; session_id?: string },
  ) {
    setError(null);
    try {
      if (decision === "session_closed") {
        if (!sessionId) {
          setError("Elegí la sesión que corresponde cerrar.");
          return;
        }
        await closeSession.mutateAsync({ id: sessionId, reason: "manual" });
      }
      await submit.mutateAsync({
        vehicleId: item.vehicle_id,
        input: {
          decision,
          ...extra,
          session_id: decision === "session_closed" ? sessionId : undefined,
        },
      });
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "No se pudo registrar la decisión",
      );
    }
  }

  const isUnmatchedExit = item.direction === "exit" && item.session_id === null;

  return (
    <article className="space-y-3 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="font-mono text-lg font-bold tracking-tight text-ink">
            {item.license_plate ?? "Sin lectura"}
          </p>
          <p className="text-xs text-muted">
            {item.vehicle_type ?? "tipo desconocido"} ·{" "}
            {item.color ?? "color desconocido"} · placa{" "}
            {formatConfidence(item.plate_confidence)}
          </p>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1">
          <span
            className={`text-[10px] font-semibold uppercase tracking-wider ${
              item.direction === "exit" ? "text-accent" : "text-live"
            }`}
          >
            {directionLabel(item.direction)}
          </span>
          <AuthorizationBadge authorized={item.authorized} />
        </div>
      </div>

      <div className="overflow-hidden rounded-lg border border-border bg-elevated">
        {item.clip_url ? (
          <video
            src={item.clip_url}
            poster={item.image_url ?? undefined}
            autoPlay
            muted
            loop
            playsInline
            controls
            className="h-32 w-full bg-black object-cover"
          >
            Tu navegador no puede reproducir el clip del evento.
          </video>
        ) : item.image_url ? (
          <img
            src={item.image_url}
            alt={`Evidencia ${item.license_plate ?? "sin placa"}`}
            loading="lazy"
            className="h-32 w-full object-cover"
          />
        ) : (
          <div className="flex h-32 items-center justify-center text-subtle">
            <ImageOff className="h-7 w-7" />
          </div>
        )}
      </div>

      <p className="text-[11px] text-subtle">
        {formatDateTime(item.observed_at ?? item.created_at)} ·{" "}
        {item.camera_id ?? "sin cámara"}
      </p>

      {correcting ? (
        <div className="space-y-2">
          <form
            className="flex items-center gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              void decide("plate_corrected", { corrected_plate: plate });
            }}
          >
            <input
              value={plate}
              onChange={(event) => {
                setPlate(event.target.value);
                setSelected(null);
              }}
              className="field font-mono"
              placeholder="1234-ABC"
              autoFocus
              required
              aria-label="Placa corregida"
            />
            <button type="submit" disabled={busy} className="btn-primary px-3 py-2 text-xs">
              Guardar
            </button>
            <button
              type="button"
              onClick={() => setCorrecting(false)}
              className="btn-secondary px-2 py-2"
              aria-label="Cancelar corrección"
            >
              <X className="h-4 w-4" />
            </button>
          </form>

          {charactersAligned && (
            <div className="space-y-1">
              <PlateCharacters
                characters={characters}
                selected={selected}
                onSelect={setSelected}
              />
              {selected !== null &&
                characters[selected] &&
                characters[selected].alternatives.length > 0 && (
                  <div className="flex flex-wrap items-center gap-1">
                    <span className="text-[11px] text-subtle">
                      Posición {selected + 1}:
                    </span>
                    {characters[selected].alternatives.map((alternative) => (
                      <button
                        key={alternative.char}
                        type="button"
                        onClick={() => {
                          if (selected !== null) {
                            pickAlternative(selected, alternative.char);
                          }
                        }}
                        className="rounded bg-elevated px-1.5 py-0.5 font-mono text-xs text-ink hover:bg-accent-soft"
                      >
                        {alternative.char}{" "}
                        <span className="text-subtle">
                          {Math.round(alternative.confidence * 100)}%
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              <p className="text-[11px] text-subtle">
                Tocá un carácter para ver las alternativas del modelo.
              </p>
            </div>
          )}
        </div>
      ) : closing ? (
        <div className="space-y-2">
          <select
            value={sessionId}
            onChange={(event) => setSessionId(event.target.value)}
            className="field"
            aria-label="Sesión a cerrar"
          >
            <option value="">Elegí la sesión que corresponde…</option>
            {(active.data ?? []).map((session) => (
              <option key={session.id} value={session.id}>
                {session.license_plate ?? "sin placa"} ·{" "}
                {formatDateTime(session.entry_time)}
              </option>
            ))}
          </select>
          <div className="flex gap-2">
            <button
              type="button"
              disabled={busy}
              onClick={() => void decide("session_closed")}
              className="btn-primary px-3 py-2 text-xs"
            >
              Cerrar sesión y marcar
            </button>
            <button
              type="button"
              onClick={() => setClosing(false)}
              className="btn-secondary px-3 py-2 text-xs"
            >
              Cancelar
            </button>
          </div>
        </div>
      ) : (
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            disabled={busy}
            onClick={() => void decide("confirmed")}
            className="btn-primary px-3 py-2 text-xs"
          >
            <Check className="h-3.5 w-3.5" />
            Confirmar
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => setCorrecting(true)}
            className="btn-secondary px-3 py-2 text-xs"
          >
            <Pencil className="h-3.5 w-3.5" />
            Corregir placa
          </button>
          {isUnmatchedExit && (
            <button
              type="button"
              disabled={busy}
              onClick={() => setClosing(true)}
              className="btn-secondary px-3 py-2 text-xs"
            >
              <ClipboardCheck className="h-3.5 w-3.5" />
              Cerrar sesión
            </button>
          )}
          <button
            type="button"
            disabled={busy}
            onClick={() => void decide("dismissed")}
            className="btn-secondary px-3 py-2 text-xs text-muted"
          >
            Descartar
          </button>
        </div>
      )}

      {error && <p className="text-xs text-danger">{error}</p>}
    </article>
  );
}

export function ValidationQueue() {
  const { user } = useSession();
  const canDecide = user?.role === "admin" || user?.role === "operator";
  const pending = usePendingValidations();
  const events = useLiveEvents();
  const [open, setOpen] = useState(false);
  const submit = useSubmitValidationDecision();

  const items = pending.data ?? [];
  const unauthorized = items.filter((item) => item.authorized === false).length;

  // The queue opens itself when someone unauthorized comes through. That is the
  // one event an operator cannot afford to notice late.
  const lastEvent = useRef<string | null>(null);
  useEffect(() => {
    const latest = events[0];
    if (!latest || latest.type === "ping" || latest.type === "session_closed") {
      return;
    }
    const key = `${latest.type}-${latest.timestamp}`;
    if (lastEvent.current === key) return;
    lastEvent.current = key;
    if (latest.authorized === false) setOpen(true);
  }, [events]);

  // Shortcuts operate on the newest pending event and never while typing.
  useEffect(() => {
    if (!open || !canDecide) return;
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const first = items[0];
      if (!first) return;
      if (event.key === "c") {
        event.preventDefault();
        void submit.mutateAsync({
          vehicleId: first.vehicle_id,
          input: { decision: "confirmed" },
        });
      } else if (event.key === "x") {
        event.preventDefault();
        void submit.mutateAsync({
          vehicleId: first.vehicle_id,
          input: { decision: "dismissed" },
        });
      } else if (event.key === "Escape") {
        setOpen(false);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, canDecide, items, submit]);

  if (!canDecide) return null;

  return (
    <>
      {(open || items.length > 0) && (
        <button
          type="button"
          onClick={() => setOpen((value) => !value)}
          className={`fixed bottom-5 right-5 z-40 inline-flex items-center gap-2 rounded-full px-4 py-3 text-sm font-semibold shadow-lg transition-colors ${
            unauthorized > 0
              ? "pulse-ring bg-danger text-white hover:bg-danger/90"
              : "bg-accent text-on-accent hover:bg-accent/90"
          }`}
          aria-label="Abrir bandeja de validación"
        >
          {unauthorized > 0 ? (
            <ShieldAlert className="h-4 w-4" />
          ) : (
            <ClipboardCheck className="h-4 w-4" />
          )}
          {items.length} pendiente{items.length === 1 ? "" : "s"}
          {unauthorized > 0 && (
            <span className="rounded-full bg-white/20 px-1.5 text-[11px]">
              {unauthorized} sin padrón
            </span>
          )}
        </button>
      )}

      {open && (
        <>
          <button
            type="button"
            aria-label="Cerrar bandeja de validación"
            onClick={() => setOpen(false)}
            className="backdrop-in fixed inset-0 z-40 cursor-default bg-black/40 backdrop-blur-sm"
          />
          <aside
            role="dialog"
            aria-label="Bandeja de validación"
            className="surface-glass drawer-in fixed inset-y-0 right-0 z-50 flex w-full max-w-md flex-col border-l border-border shadow-2xl"
          >
          <header className="flex items-center justify-between border-b border-border px-4 py-3">
            <div>
              <h3 className="font-display text-base font-bold text-ink">
                Bandeja de validación
              </h3>
              <p className="text-xs text-muted">
                Confirmá o corregí lo que registró el portón
              </p>
            </div>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded p-1 text-subtle hover:text-ink"
              aria-label="Cerrar bandeja"
            >
              <X className="h-5 w-5" />
            </button>
          </header>

          <div className="flex-1 divide-y divide-border overflow-y-auto">
            {pending.isLoading ? (
              <p className="py-12 text-center text-sm text-subtle">Cargando…</p>
            ) : pending.isError ? (
              <p className="py-12 text-center text-sm text-danger">
                No se pudo leer la bandeja.
              </p>
            ) : items.length === 0 ? (
              <div className="flex h-48 flex-col items-center justify-center gap-2 text-center">
                <ShieldCheck className="h-6 w-6 text-live" />
                <p className="text-sm text-muted">No hay eventos pendientes</p>
                <p className="text-xs text-subtle">
                  Todo lo que cruzó el portón ya fue revisado.
                </p>
              </div>
            ) : (
              items.map((item) => (
                <ValidationCard key={item.vehicle_id} item={item} />
              ))
            )}
          </div>

          <footer className="border-t border-border px-4 py-2 text-[11px] text-subtle">
            Atajos: <kbd className="font-mono">c</kbd> confirmar el más reciente ·{" "}
            <kbd className="font-mono">x</kbd> descartar
          </footer>
        </aside>
        </>
      )}
    </>
  );
}
