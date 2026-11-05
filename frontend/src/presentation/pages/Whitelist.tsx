// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

import { useState, type FormEvent } from "react";
import {
  CheckCircle2,
  Edit2,
  Plus,
  Search,
  Trash2,
  X,
  XCircle,
} from "lucide-react";
import {
  useAuthorizedVehicles,
  useCreateAuthorizedVehicle,
  useDeleteAuthorizedVehicle,
  useUpdateAuthorizedVehicle,
} from "../../application/queries";
import { formatDateTime } from "../../domain/incident";
import type { AuthorizedVehicle } from "../../domain/types";

export function Whitelist() {
  const [search, setSearch] = useState("");
  const [activeOnly, setActiveOnly] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingVehicle, setEditingVehicle] = useState<AuthorizedVehicle | null>(null);

  const [plate, setPlate] = useState("");
  const [owner, setOwner] = useState("");
  const [vehicleType, setVehicleType] = useState("");
  const [color, setColor] = useState("");
  const [notes, setNotes] = useState("");
  const [isActive, setIsActive] = useState(true);

  const { data: vehicles, isLoading, isError } = useAuthorizedVehicles({
    search: search.trim() || undefined,
    active_only: activeOnly,
  });

  const createMutation = useCreateAuthorizedVehicle();
  const updateMutation = useUpdateAuthorizedVehicle();
  const deleteMutation = useDeleteAuthorizedVehicle();

  function openCreateModal() {
    setEditingVehicle(null);
    setPlate("");
    setOwner("");
    setVehicleType("Sedán");
    setColor("Blanco");
    setNotes("");
    setIsActive(true);
    setIsModalOpen(true);
  }

  function openEditModal(v: AuthorizedVehicle) {
    setEditingVehicle(v);
    setPlate(v.license_plate);
    setOwner(v.owner_name || "");
    setVehicleType(v.vehicle_type || "");
    setColor(v.color || "");
    setNotes(v.notes || "");
    setIsActive(v.is_active);
    setIsModalOpen(true);
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (editingVehicle) {
      updateMutation.mutate(
        {
          id: editingVehicle.id,
          input: {
            owner_name: owner || undefined,
            vehicle_type: vehicleType || undefined,
            color: color || undefined,
            notes: notes || undefined,
            is_active: isActive,
          },
        },
        { onSuccess: () => setIsModalOpen(false) },
      );
    } else {
      createMutation.mutate(
        {
          license_plate: plate.trim().toUpperCase(),
          owner_name: owner || undefined,
          vehicle_type: vehicleType || undefined,
          color: color || undefined,
          notes: notes || undefined,
          is_active: isActive,
        },
        { onSuccess: () => setIsModalOpen(false) },
      );
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-subtle">
            Control de Acceso
          </p>
          <h2 className="font-display text-2xl font-bold tracking-tight text-ink">
            Padrón de Vehículos Autorizados
          </h2>
          <p className="text-sm text-muted">
            Registro institucional para verificación y alertas de acceso no autorizado (RF-10)
          </p>
        </div>

        <button
          type="button"
          onClick={openCreateModal}
          className="btn-accent inline-flex items-center gap-2 self-start sm:self-auto"
        >
          <Plus className="h-4 w-4" />
          Nuevo vehículo
        </button>
      </div>

      {/* Filter toolbar */}
      <div className="card flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-subtle" />
          <input
            type="text"
            placeholder="Buscar por placa o propietario..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-lg border border-border bg-canvas pl-9 pr-4 py-2 text-sm text-ink outline-none focus:border-accent focus:ring-2 focus:ring-accent/30"
          />
        </div>

        <label className="flex items-center gap-2 text-sm text-muted cursor-pointer select-none">
          <input
            type="checkbox"
            checked={activeOnly}
            onChange={(e) => setActiveOnly(e.target.checked)}
            className="rounded border-border text-accent focus:ring-accent"
          />
          Mostrar solo activos
        </label>
      </div>

      {/* Table matching Fig 11c */}
      <div className="card overflow-hidden p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-elevated">
                <th className="text-left py-3 px-4 font-medium text-muted">Matrícula</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Propietario / Área</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Tipo</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Color</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Estado</th>
                <th className="text-left py-3 px-4 font-medium text-muted">Registro</th>
                <th className="text-right py-3 px-4 font-medium text-muted">Acciones</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {isLoading ? (
                <tr>
                  <td colSpan={7} className="py-12 text-center text-subtle">
                    Cargando padrón…
                  </td>
                </tr>
              ) : isError ? (
                <tr>
                  <td colSpan={7} className="py-12 text-center text-danger">
                    Error al cargar los vehículos autorizados
                  </td>
                </tr>
              ) : vehicles && vehicles.length > 0 ? (
                vehicles.map((v) => (
                  <tr key={v.id} className="hover:bg-elevated">
                    <td className="py-3 px-4 font-medium font-mono text-ink">
                      {v.license_plate}
                    </td>
                    <td className="py-3 px-4 text-ink font-medium">
                      {v.owner_name || <span className="text-subtle italic">Sin asignar</span>}
                      {v.notes && <p className="text-xs text-subtle font-normal">{v.notes}</p>}
                    </td>
                    <td className="py-3 px-4 text-muted">{v.vehicle_type || "—"}</td>
                    <td className="py-3 px-4 text-muted">{v.color || "—"}</td>
                    <td className="py-3 px-4">
                      {v.is_active ? (
                        <span className="badge-active inline-flex items-center gap-1">
                          <CheckCircle2 className="h-3 w-3" /> Habilitado
                        </span>
                      ) : (
                        <span className="badge-completed inline-flex items-center gap-1 opacity-60">
                          <XCircle className="h-3 w-3" /> Inactivo
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-muted text-xs">
                      {formatDateTime(v.created_at)}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <div className="inline-flex items-center gap-1">
                        <button
                          type="button"
                          onClick={() => openEditModal(v)}
                          title="Editar"
                          className="btn-ghost p-1.5 text-muted hover:text-ink"
                        >
                          <Edit2 className="h-4 w-4" />
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            if (confirm(`¿Eliminar la placa ${v.license_plate} del padrón?`)) {
                              deleteMutation.mutate(v.id);
                            }
                          }}
                          title="Eliminar"
                          className="btn-ghost p-1.5 text-muted hover:text-danger"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} className="py-12 text-center text-subtle">
                    {search ? "No se encontraron vehículos en el padrón" : "El padrón de vehículos está vacío"}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Modal for create/edit */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4 backdrop-blur-sm">
          <form
            onSubmit={handleSubmit}
            className="card w-full max-w-md space-y-4 p-6"
          >
            <div className="flex items-start justify-between">
              <div>
                <h3 className="font-display text-lg font-bold text-ink">
                  {editingVehicle ? "Editar vehículo autorizado" : "Registrar nuevo vehículo"}
                </h3>
                <p className="text-xs text-muted">
                  Datos de identificación en el padrón
                </p>
              </div>
              <button
                type="button"
                onClick={() => setIsModalOpen(false)}
                className="rounded p-1 text-subtle hover:text-ink"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="space-y-3">
              <div>
                <label className="text-xs font-medium text-ink">Placa / Matrícula *</label>
                <input
                  type="text"
                  required
                  disabled={Boolean(editingVehicle)}
                  value={plate}
                  onChange={(e) => setPlate(e.target.value.toUpperCase())}
                  placeholder="Ej. 1234ABC"
                  className="mt-1 w-full rounded-lg border border-border bg-canvas px-3 py-2 font-mono text-sm uppercase text-ink outline-none focus:border-accent disabled:opacity-50"
                />
              </div>

              <div>
                <label className="text-xs font-medium text-ink">Propietario / Funcionario / Docente</label>
                <input
                  type="text"
                  value={owner}
                  onChange={(e) => setOwner(e.target.value)}
                  placeholder="Ej. Ing. Juan Pérez (Docente)"
                  className="mt-1 w-full rounded-lg border border-border bg-canvas px-3 py-2 text-sm text-ink outline-none focus:border-accent"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs font-medium text-ink">Tipo de vehículo</label>
                  <input
                    type="text"
                    value={vehicleType}
                    onChange={(e) => setVehicleType(e.target.value)}
                    placeholder="Ej. Sedán, Camioneta"
                    className="mt-1 w-full rounded-lg border border-border bg-canvas px-3 py-2 text-sm text-ink outline-none focus:border-accent"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-ink">Color</label>
                  <input
                    type="text"
                    value={color}
                    onChange={(e) => setColor(e.target.value)}
                    placeholder="Ej. Blanco, Gris"
                    className="mt-1 w-full rounded-lg border border-border bg-canvas px-3 py-2 text-sm text-ink outline-none focus:border-accent"
                  />
                </div>
              </div>

              <div>
                <label className="text-xs font-medium text-ink">Observaciones / Notas</label>
                <textarea
                  rows={2}
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Observaciones de ingreso, cargo, o autorización especial..."
                  className="mt-1 w-full rounded-lg border border-border bg-canvas px-3 py-2 text-sm text-ink outline-none focus:border-accent"
                />
              </div>

              <label className="flex items-center gap-2 pt-1 text-sm text-ink cursor-pointer">
                <input
                  type="checkbox"
                  checked={isActive}
                  onChange={(e) => setIsActive(e.target.checked)}
                  className="rounded border-border text-accent focus:ring-accent"
                />
                Autorización activa
              </label>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setIsModalOpen(false)}
                className="btn-ghost text-xs"
              >
                Cancelar
              </button>
              <button
                type="submit"
                disabled={createMutation.isPending || updateMutation.isPending}
                className="btn-accent text-xs"
              >
                {editingVehicle ? "Guardar cambios" : "Registrar"}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
