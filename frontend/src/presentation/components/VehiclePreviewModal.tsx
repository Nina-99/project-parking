// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

import { CheckCircle2, ExternalLink, ImageOff, ShieldAlert, X } from "lucide-react";
import { Link } from "react-router-dom";
import { useApi } from "../../application/context";
import { useVehicle } from "../../application/queries";
import { useSession } from "../../application/session";
import { useQuery } from "@tanstack/react-query";
import {
  formatConfidence,
  formatDateTime,
} from "../../domain/incident";
import { PlateCharacters } from "./PlateCharacters";

export function VehiclePreviewModal({
  vehicleId,
  onDismiss,
}: {
  vehicleId: string;
  onDismiss: () => void;
}) {
  const api = useApi();
  const vehicle = useVehicle(vehicleId);
  const { user } = useSession();
  const canCheckAuthorization =
    user?.role === "admin" || user?.role === "operator";

  const plate = vehicle.data?.license_plate;
  const authQuery = useQuery({
    queryKey: ["whitelist-check", plate],
    queryFn: () => (plate ? api.checkAuthorization(plate) : Promise.resolve(null)),
    // The owner reads reports, not the registry; the check is operator/admin on
    // the server, so asking for it with another role is a guaranteed 403.
    enabled: Boolean(plate) && canCheckAuthorization,
  });

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4 backdrop-blur-sm"
      onClick={onDismiss}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Evidencia del vehículo"
        className="card w-full max-w-lg space-y-4 p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between">
          <div>
            <h3 className="font-display text-lg font-bold text-ink">
              Evidencia de detección
            </h3>
            <p className="text-xs text-muted">Recorte confirmado por el motor de visión</p>
          </div>
          <button
            type="button"
            onClick={onDismiss}
            className="rounded p-1 text-subtle hover:text-ink"
            aria-label="Cerrar modal"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {vehicle.isLoading ? (
          <div className="flex h-60 items-center justify-center text-sm text-subtle">
            Cargando evidencia visual…
          </div>
        ) : vehicle.isError || !vehicle.data ? (
          <div className="flex h-60 flex-col items-center justify-center text-center text-sm text-danger">
            <p>No se pudo cargar la evidencia del vehículo.</p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="overflow-hidden rounded-lg border border-border bg-elevated">
              {vehicle.data.clip_url ? (
                <video
                  src={vehicle.data.clip_url}
                  poster={vehicle.data.image_url ?? undefined}
                  controls
                  autoPlay
                  muted
                  loop
                  playsInline
                  className="h-64 w-full bg-black object-contain"
                >
                  Tu navegador no puede reproducir el clip del evento.
                </video>
              ) : vehicle.data.image_url ? (
                <img
                  src={vehicle.data.image_url}
                  alt={`Evidencia ${vehicle.data.license_plate ?? "sin placa"}`}
                  className="h-64 w-full object-cover"
                />
              ) : (
                <div className="flex h-64 items-center justify-center text-subtle">
                  <ImageOff className="h-10 w-10" />
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3 rounded-lg border border-border bg-canvas/50 p-3 text-xs">
              <div>
                <span className="text-muted">Placa detectada:</span>
                <p className="font-mono text-base font-bold text-ink">
                  {vehicle.data.license_plate ?? "Sin lectura"}
                </p>
                {vehicle.data.plate_characters &&
                  vehicle.data.plate_characters.length > 0 && (
                    <div className="mt-1">
                      <PlateCharacters characters={vehicle.data.plate_characters} />
                    </div>
                  )}
                {plate && authQuery.data && (
                  <div className="mt-1">
                    {authQuery.data.authorized ? (
                      <span className="badge-active inline-flex items-center gap-1 text-[10px]">
                        <CheckCircle2 className="h-3 w-3" /> Padrón:{" "}
                        {authQuery.data.vehicle?.owner_name || "Autorizado"}
                      </span>
                    ) : (
                      <span className="badge-stale inline-flex items-center gap-1 text-[10px]">
                        <ShieldAlert className="h-3 w-3" /> No autorizado
                      </span>
                    )}
                  </div>
                )}
              </div>
              <div>
                <span className="text-muted">Registro:</span>
                <p className="font-mono text-ink">
                  {formatDateTime(vehicle.data.created_at)}
                </p>
              </div>
              <div>
                <span className="text-muted">Tipo:</span>
                <p className="text-ink">
                  {vehicle.data.vehicle_type ?? "—"}{" "}
                  <span className="text-subtle">
                    ({formatConfidence(vehicle.data.vehicle_type_confidence)})
                  </span>
                </p>
              </div>
              <div>
                <span className="text-muted">Color:</span>
                <p className="text-ink">
                  {vehicle.data.color ?? "—"}{" "}
                  <span className="text-subtle">
                    ({formatConfidence(vehicle.data.color_confidence)})
                  </span>
                </p>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2">
              <Link
                to={`/vehicles/${vehicleId}`}
                className="btn-secondary inline-flex items-center gap-1.5 text-xs"
              >
                <ExternalLink className="h-3.5 w-3.5" />
                Ver expediente completo
              </Link>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
