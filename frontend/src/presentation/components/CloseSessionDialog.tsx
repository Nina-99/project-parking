// SPDX-FileCopyrightText: 2026 Nina
// SPDX-License-Identifier: LicenseRef-Proprietary

import { useState, type FormEvent } from "react";
import { X } from "lucide-react";
import { useCloseSession } from "../../application/queries";
import { formatTime } from "../../domain/incident";
import type { ActiveSession } from "../../domain/types";

export function CloseSessionDialog({
  session,
  onDismiss,
}: {
  session: ActiveSession;
  onDismiss: () => void;
}) {
  const close = useCloseSession();
  const [reason, setReason] = useState("salió sin lectura");

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    close.mutate(
      { id: session.id, reason },
      { onSuccess: () => onDismiss() },
    );
  }

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center px-4">
      <form
        onSubmit={submit}
        role="dialog"
        aria-modal="true"
        aria-label="Cerrar sesión"
        className="card w-full max-w-sm space-y-4"
      >
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-lg font-semibold text-ink">Cerrar sesión</h3>
            <p className="text-xs text-muted">
              {session.license_plate || "Sin placa"} · ingresó a las{" "}
              {formatTime(session.entry_time)} · {session.age_minutes} min
            </p>
          </div>
          <button
            type="button"
            onClick={onDismiss}
            className="p-1 rounded text-subtle hover:text-ink"
            aria-label="Cancelar"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <label className="flex flex-col gap-1 text-sm">
          <span className="text-muted font-medium">Motivo</span>
          <input
            value={reason}
            onChange={(event) => setReason(event.target.value)}
            maxLength={50}
            required
            className="border border-border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-accent/30 focus:border-accent outline-none"
          />
        </label>

        {close.error && (
          <p className="text-sm text-danger">
            {close.error instanceof Error
              ? close.error.message
              : "No se pudo cerrar la sesión"}
          </p>
        )}

        <div className="flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onDismiss}
            className="px-3 py-2 rounded-lg text-sm text-muted hover:bg-elevated"
          >
            Cancelar
          </button>
          <button
            type="submit"
            disabled={close.isPending}
            className="px-3 py-2 rounded-lg bg-accent text-on-accent text-sm font-medium hover:bg-accent/90 disabled:opacity-50"
          >
            {close.isPending ? "Cerrando…" : "Cerrar sesión"}
          </button>
        </div>
      </form>
    </div>
  );
}
