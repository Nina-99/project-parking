// SPDX-FileCopyrightText: 2026 Nina
// SPDX-License-Identifier: LicenseRef-Proprietary

import { useMemo, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { Clock, ImageOff, Search } from "lucide-react";
import { useVehicleSearch } from "../../application/queries";
import {
  describeMatch,
  formatConfidence,
  formatDateTime,
  matchBadgeClass,
} from "../../domain/incident";
import {
  REQUESTABLE_COLOURS,
  type VehicleSearchHit,
  type VehicleSearchQuery,
} from "../../domain/types";

const SWATCHES: Record<string, string> = {
  blanco: "#f8fafc",
  plata: "#cbd5e1",
  gris: "#64748b",
  negro: "#0f172a",
  azul: "#1d4ed8",
  rojo: "#b91c1c",
};

function toLocalInput(date: Date): string {
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(
    date.getHours(),
  )}:${pad(date.getMinutes())}`;
}

function defaultWindow(): { from: string; to: string } {
  const to = new Date();
  const from = new Date(to.getTime() - 24 * 60 * 60 * 1000);
  return { from: toLocalInput(from), to: toLocalInput(to) };
}

function ResultCard({
  hit,
  requestedColour,
}: {
  hit: VehicleSearchHit;
  requestedColour?: string;
}) {
  return (
    <article className="card h-full overflow-hidden p-0 flex flex-col">
      {hit.image_url ? (
        <img
          src={hit.image_url}
          alt={`Vehículo ${hit.license_plate ?? "sin placa"}`}
          loading="lazy"
          className="h-44 w-full object-cover bg-elevated"
        />
      ) : (
        <div className="h-44 w-full bg-elevated flex items-center justify-center text-subtle">
          <ImageOff className="w-8 h-8" />
        </div>
      )}

      <div className="flex flex-1 flex-col gap-2 p-4">
        <div className="flex items-center justify-between gap-2">
          <span className="font-mono text-base font-semibold tracking-tight text-ink">
            {hit.license_plate ?? "Sin placa"}
          </span>
          {requestedColour && (
            <span
              className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${matchBadgeClass(
                hit.match,
              )}`}
            >
              {hit.match === "exact" ? "Color exacto" : "Vecino medido"}
            </span>
          )}
        </div>

        <p className="text-xs text-muted">
          {hit.vehicle_type ?? "tipo desconocido"} ·{" "}
          {hit.color ?? "color desconocido"} ·{" "}
          {formatConfidence(hit.color_confidence)}
        </p>

        <p className="text-xs text-subtle flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {formatDateTime(hit.created_at)}
        </p>

        <p className="mt-auto border-t border-border pt-2 text-xs text-muted">
          {describeMatch(hit, requestedColour)}
        </p>
      </div>
    </article>
  );
}

export function Incidents() {
  const initial = useMemo(defaultWindow, []);
  const [from, setFrom] = useState(initial.from);
  const [to, setTo] = useState(initial.to);
  const [colour, setColour] = useState("");
  const [query, setQuery] = useState<VehicleSearchQuery | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const results = useVehicleSearch(query);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const start = new Date(from);
    const end = new Date(to);
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
      setFormError("Completá las dos fechas.");
      return;
    }
    if (start > end) {
      setFormError("La fecha de inicio no puede ser posterior a la de fin.");
      return;
    }
    setFormError(null);
    setQuery({
      from: start.toISOString(),
      to: end.toISOString(),
      color: colour || undefined,
    });
  }

  const hits = results.data ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="font-display text-2xl font-bold text-ink">Búsqueda de incidentes</h2>
        <p className="text-sm text-muted mt-1">
          Recordá una ventana horaria y, si la tenés, el color. La lista viene
          ordenada por cercanía al momento recordado.
        </p>
      </div>

      <form onSubmit={submit} className="card space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-muted font-medium">Desde</span>
            <input
              type="datetime-local"
              value={from}
              onChange={(event) => setFrom(event.target.value)}
              className="field"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-muted font-medium">Hasta</span>
            <input
              type="datetime-local"
              value={to}
              onChange={(event) => setTo(event.target.value)}
              className="field"
            />
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm text-muted font-medium mr-1">Color:</span>
          <button
            type="button"
            aria-pressed={colour === ""}
            onClick={() => setColour("")}
            className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
              colour === ""
                ? "bg-accent text-on-accent border-accent"
                : "bg-surface text-muted border-border hover:border-accent"
            }`}
          >
            <span className="h-3 w-3 rounded-full bg-gradient-to-br from-red-500 via-amber-400 to-teal-500" />
            Todos
          </button>
          {REQUESTABLE_COLOURS.map((option) => (
            <button
              key={option}
              type="button"
              aria-pressed={colour === option}
              onClick={() => setColour(option)}
              className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
                colour === option
                  ? "bg-accent text-on-accent border-accent"
                  : "bg-surface text-muted border-border hover:border-accent"
              }`}
            >
              <span
                className="h-3 w-3 rounded-full border border-black/10"
                style={{ background: SWATCHES[option] }}
              />
              {option}
            </button>
          ))}
        </div>

        <div className="flex items-center justify-between gap-4">
          {formError ? (
            <p className="text-sm text-danger">{formError}</p>
          ) : (
            <p className="text-xs text-subtle">
              El color es un filtro blando: se incluyen los vecinos que el
              clasificador confunde con el pedido.
            </p>
          )}
          <button
            type="submit"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-accent text-on-accent text-sm font-medium hover:bg-accent/90 transition-colors"
          >
            <Search className="w-4 h-4" />
            Buscar
          </button>
        </div>
      </form>

      {query === null ? (
        <p className="text-sm text-subtle text-center py-12">
          Elegí una ventana y buscá para ver candidatos.
        </p>
      ) : results.isFetching ? (
        <p className="text-sm text-subtle text-center py-12">Buscando…</p>
      ) : results.isError ? (
        <p className="text-sm text-danger text-center py-12">
          {results.error instanceof Error
            ? results.error.message
            : "No se pudo completar la búsqueda"}
        </p>
      ) : hits.length === 0 ? (
        <div className="text-center py-12 space-y-1">
          <p className="text-sm text-muted">No hay candidatos en esa ventana.</p>
          <p className="text-xs text-subtle">
            Probá ampliar el rango o quitar el filtro de color.
          </p>
        </div>
      ) : (
        <>
          <p className="text-sm text-muted">
            {hits.length} candidato{hits.length === 1 ? "" : "s"}
            {query.color ? ", mejor match primero" : " en la ventana"}
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {hits.map((hit) => (
              <Link
                key={hit.id}
                to={`/vehicles/${hit.id}`}
                className="block rounded-xl transition-shadow hover:shadow-md focus:outline-none focus:ring-2 focus:ring-accent/30"
              >
                <ResultCard hit={hit} requestedColour={query.color} />
              </Link>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
