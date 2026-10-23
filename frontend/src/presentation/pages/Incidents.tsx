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
    <article className="card p-0 overflow-hidden flex flex-col">
      {hit.image_url ? (
        <img
          src={hit.image_url}
          alt={`Vehículo ${hit.license_plate ?? "sin placa"}`}
          loading="lazy"
          className="h-40 w-full object-cover bg-gray-100"
        />
      ) : (
        <div className="h-40 w-full bg-gray-100 flex items-center justify-center text-gray-300">
          <ImageOff className="w-8 h-8" />
        </div>
      )}

      <div className="p-4 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <span className="font-mono text-sm font-semibold text-gray-900">
            {hit.license_plate ?? "Sin placa"}
          </span>
          <span
            className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${matchBadgeClass(
              hit.match,
            )}`}
          >
            {hit.match === "exact" ? "Color exacto" : "Vecino medido"}
          </span>
        </div>

        <p className="text-xs text-gray-500">
          {hit.vehicle_type ?? "tipo desconocido"} · {hit.color ?? "color desconocido"} (
          {formatConfidence(hit.color_confidence)})
        </p>

        <p className="text-xs text-gray-400 flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {formatDateTime(hit.created_at)}
        </p>

        <p className="text-xs text-gray-500 border-t border-gray-100 pt-2">
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
        <h2 className="text-2xl font-bold text-gray-900">Búsqueda de incidentes</h2>
        <p className="text-sm text-gray-500 mt-1">
          Recordá una ventana horaria y, si la tenés, el color. La lista viene
          ordenada por cercanía al momento recordado.
        </p>
      </div>

      <form onSubmit={submit} className="card space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-gray-600 font-medium">Desde</span>
            <input
              type="datetime-local"
              value={from}
              onChange={(event) => setFrom(event.target.value)}
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-gray-600 font-medium">Hasta</span>
            <input
              type="datetime-local"
              value={to}
              onChange={(event) => setTo(event.target.value)}
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
            />
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm text-gray-600 font-medium mr-1">Color:</span>
          <button
            type="button"
            aria-pressed={colour === ""}
            onClick={() => setColour("")}
            className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
              colour === ""
                ? "bg-gray-900 text-white border-gray-900"
                : "bg-white text-gray-600 border-gray-300 hover:border-gray-400"
            }`}
          >
            Todos
          </button>
          {REQUESTABLE_COLOURS.map((option) => (
            <button
              key={option}
              type="button"
              aria-pressed={colour === option}
              onClick={() => setColour(option)}
              className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
                colour === option
                  ? "bg-gray-900 text-white border-gray-900"
                  : "bg-white text-gray-600 border-gray-300 hover:border-gray-400"
              }`}
            >
              {option}
            </button>
          ))}
        </div>

        <div className="flex items-center justify-between gap-4">
          {formError ? (
            <p className="text-sm text-red-600">{formError}</p>
          ) : (
            <p className="text-xs text-gray-400">
              El color es un filtro blando: se incluyen los vecinos que el
              clasificador confunde con el pedido.
            </p>
          )}
          <button
            type="submit"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-900 text-white text-sm font-medium hover:bg-gray-800 transition-colors"
          >
            <Search className="w-4 h-4" />
            Buscar
          </button>
        </div>
      </form>

      {query === null ? (
        <p className="text-sm text-gray-400 text-center py-12">
          Elegí una ventana y buscá para ver candidatos.
        </p>
      ) : results.isFetching ? (
        <p className="text-sm text-gray-400 text-center py-12">Buscando…</p>
      ) : results.isError ? (
        <p className="text-sm text-red-500 text-center py-12">
          {results.error instanceof Error
            ? results.error.message
            : "No se pudo completar la búsqueda"}
        </p>
      ) : hits.length === 0 ? (
        <div className="text-center py-12 space-y-1">
          <p className="text-sm text-gray-500">No hay candidatos en esa ventana.</p>
          <p className="text-xs text-gray-400">
            Probá ampliar el rango o quitar el filtro de color.
          </p>
        </div>
      ) : (
        <>
          <p className="text-sm text-gray-500">
            {hits.length} candidato{hits.length === 1 ? "" : "s"}, mejor match primero.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {hits.map((hit) => (
              <Link
                key={hit.id}
                to={`/vehicles/${hit.id}`}
                className="block rounded-xl transition-shadow hover:shadow-md focus:outline-none focus:ring-2 focus:ring-primary-500"
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
