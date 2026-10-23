import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Car, ImageOff } from "lucide-react";
import { useVehicle } from "../../application/queries";
import {
  formatConfidence,
  formatDateTime,
  formatDuration,
} from "../../domain/incident";
import type { VehicleSession } from "../../domain/types";

function closureLabel(session: VehicleSession): string {
  if (session.closed_by) {
    return `${session.closed_by}${session.close_reason ? `: ${session.close_reason}` : ""}`;
  }
  if (session.close_reason === "plate_match") return "Automático (patente)";
  return "—";
}

export function VehicleDetail() {
  const { id } = useParams<{ id: string }>();
  const vehicle = useVehicle(id);

  if (vehicle.isLoading) {
    return <p className="text-sm text-gray-400 py-12 text-center">Cargando…</p>;
  }
  if (vehicle.isError || !vehicle.data) {
    return (
      <div className="space-y-4">
        <Link to="/incidents" className="inline-flex items-center gap-1 text-sm text-gray-600 hover:text-gray-900">
          <ArrowLeft className="w-4 h-4" /> Volver a incidentes
        </Link>
        <p className="text-sm text-red-600">No se encontró el vehículo.</p>
      </div>
    );
  }

  const data = vehicle.data;

  return (
    <div className="space-y-6">
      <Link
        to="/incidents"
        className="inline-flex items-center gap-1 text-sm text-gray-600 hover:text-gray-900"
      >
        <ArrowLeft className="w-4 h-4" /> Volver a incidentes
      </Link>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card p-0 overflow-hidden">
          {data.image_url ? (
            <img
              src={data.image_url}
              alt={`Vehículo ${data.license_plate ?? "sin placa"}`}
              className="w-full h-80 object-cover bg-gray-100"
            />
          ) : (
            <div className="h-80 bg-gray-100 flex items-center justify-center text-gray-300">
              <ImageOff className="w-10 h-10" />
            </div>
          )}
        </div>

        <div className="card space-y-4">
          <div>
            <p className="font-mono text-2xl font-semibold text-gray-900">
              {data.license_plate ?? "Sin placa"}
            </p>
            <p className="text-sm text-gray-500">
              Registrado el {formatDateTime(data.created_at)}
            </p>
          </div>

          <dl className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <dt className="text-gray-500">Tipo</dt>
              <dd className="text-gray-900">
                {data.vehicle_type ?? "—"} · {formatConfidence(data.vehicle_type_confidence)}
              </dd>
            </div>
            <div>
              <dt className="text-gray-500">Color</dt>
              <dd className="text-gray-900">
                {data.color ?? "—"} · {formatConfidence(data.color_confidence)}
              </dd>
            </div>
            <div>
              <dt className="text-gray-500">Sesiones</dt>
              <dd className="text-gray-900">{data.sessions.length}</dd>
            </div>
            <div>
              <dt className="text-gray-500">Id</dt>
              <dd className="text-gray-500 font-mono text-xs truncate">{data.id}</dd>
            </div>
          </dl>

          <p className="text-xs text-gray-400 flex items-center gap-1">
            <Car className="w-3 h-3" />
            El thumbnail corresponde a la vista más grande del vehículo.
          </p>
        </div>
      </div>

      <div className="card overflow-hidden p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 bg-gray-50">
                <th className="text-left py-3 px-4 font-medium text-gray-500">Ingreso</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">Salida</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">Duración</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">Estado</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500">Cierre</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {data.sessions.length > 0 ? (
                data.sessions.map((session) => (
                  <tr key={session.id} className="hover:bg-gray-50">
                    <td className="py-3 px-4 text-gray-600">
                      {formatDateTime(session.entry_time)}
                    </td>
                    <td className="py-3 px-4 text-gray-600">
                      {session.exit_time ? formatDateTime(session.exit_time) : "—"}
                    </td>
                    <td className="py-3 px-4 text-gray-600">
                      {formatDuration(session.duration_seconds)}
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={
                          session.status === "active" ? "badge-active" : "badge-completed"
                        }
                      >
                        {session.status === "active" ? "Activo" : "Completado"}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-gray-600">
                      {closureLabel(session)}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="py-12 text-center text-gray-400">
                    Sin sesiones registradas
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
