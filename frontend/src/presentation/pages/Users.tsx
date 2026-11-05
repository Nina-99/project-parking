// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

/**
 * Who can enter the system.
 *
 * Three invariants are the server's, not this page's: you cannot delete
 * yourself, the last active admin cannot be demoted or removed, and a password
 * needs eight characters. The UI shows the server's answer instead of guessing,
 * because a disabled button that looks right for the wrong reason is worse than
 * an error message.
 */
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { KeyRound, Plus, Trash2, UserCheck, UserX } from "lucide-react";
import { useApi } from "../../application/context";
import { useSession } from "../../application/session";
import {
  ROLE_DESCRIPTIONS,
  ROLE_LABELS,
  type ManagedUser,
  type UserRole,
} from "../../domain/types";

const ROLES: UserRole[] = ["admin", "operator", "owner"];
const MIN_PASSWORD = 8;

function when(value: string | null): string {
  if (!value) return "nunca";
  return new Date(value).toLocaleString("es-BO", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function Users() {
  const api = useApi();
  const queryClient = useQueryClient();
  const { user: me } = useSession();

  const users = useQuery({ queryKey: ["users"], queryFn: () => api.listUsers() });

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<UserRole>("operator");
  const [notice, setNotice] = useState<{ kind: "ok" | "error"; text: string } | null>(null);

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ["users"] });
  const fail = (cause: unknown) =>
    setNotice({ kind: "error", text: cause instanceof Error ? cause.message : "Falló la operación" });

  const create = useMutation({
    mutationFn: () => api.createUser({ username: username.trim(), password, role }),
    onSuccess: () => {
      setUsername("");
      setPassword("");
      setRole("operator");
      setNotice({ kind: "ok", text: "Usuario creado" });
      void invalidate();
    },
    onError: fail,
  });

  const update = useMutation({
    mutationFn: (input: { id: string; body: Parameters<typeof api.updateUser>[1] }) =>
      api.updateUser(input.id, input.body),
    onSuccess: () => {
      setNotice({ kind: "ok", text: "Usuario actualizado" });
      void invalidate();
    },
    onError: fail,
  });

  const remove = useMutation({
    mutationFn: (id: string) => api.deleteUser(id),
    onSuccess: () => {
      setNotice({ kind: "ok", text: "Usuario eliminado" });
      void invalidate();
    },
    onError: fail,
  });

  const resetPassword = (target: ManagedUser) => {
    const value = window.prompt(`Nueva contraseña para ${target.username} (mínimo ${MIN_PASSWORD})`);
    if (value === null) return;
    if (value.length < MIN_PASSWORD) {
      setNotice({ kind: "error", text: `La contraseña necesita ${MIN_PASSWORD} caracteres` });
      return;
    }
    update.mutate({ id: target.id, body: { password: value } });
  };

  const rows = users.data ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="font-display text-2xl font-bold text-ink">Usuarios</h2>
        <p className="text-sm text-muted mt-1">
          Quién entra al sistema y con qué rol. El administrador del parqueo es una
          persona, no una constante de configuración.
        </p>
      </div>

      {notice && (
        <div
          className={`rounded-xl border px-4 py-3 text-sm ${
            notice.kind === "ok"
              ? "border-accent/40 bg-accent-soft text-accent"
              : "border-danger/40 bg-danger/10 text-danger"
          }`}
          role="status"
        >
          {notice.text}
        </div>
      )}

      <form
        className="rounded-2xl border border-border bg-surface p-4"
        onSubmit={(event) => {
          event.preventDefault();
          setNotice(null);
          create.mutate();
        }}
      >
        <p className="mb-3 flex items-center gap-2 text-sm font-semibold text-ink">
          <Plus className="h-4 w-4" /> Nuevo usuario
        </p>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
          <label className="text-xs text-muted">
            Usuario
            <input
              className="mt-1 w-full rounded-lg border border-border bg-canvas px-3 py-2 text-sm text-ink"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              minLength={3}
              maxLength={64}
              required
            />
          </label>
          <label className="text-xs text-muted">
            Contraseña
            <input
              className="mt-1 w-full rounded-lg border border-border bg-canvas px-3 py-2 text-sm text-ink"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              minLength={MIN_PASSWORD}
              required
            />
          </label>
          <label className="text-xs text-muted">
            Rol
            <select
              className="mt-1 w-full rounded-lg border border-border bg-canvas px-3 py-2 text-sm text-ink"
              value={role}
              onChange={(event) => setRole(event.target.value as UserRole)}
            >
              {ROLES.map((value) => (
                <option key={value} value={value}>
                  {ROLE_LABELS[value]}
                </option>
              ))}
            </select>
          </label>
          <div className="flex items-end">
            <button
              type="submit"
              disabled={create.isPending}
              className="w-full rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
            >
              {create.isPending ? "Creando…" : "Crear"}
            </button>
          </div>
        </div>
        <p className="mt-2 text-[11px] text-subtle">{ROLE_DESCRIPTIONS[role]}</p>
      </form>

      <div className="overflow-x-auto rounded-2xl border border-border bg-surface">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-border text-[11px] uppercase tracking-[0.12em] text-subtle">
            <tr>
              <th className="px-4 py-3">Usuario</th>
              <th className="px-4 py-3">Rol</th>
              <th className="px-4 py-3">Estado</th>
              <th className="px-4 py-3">Último acceso</th>
              <th className="px-4 py-3 text-right">Acciones</th>
            </tr>
          </thead>
          <tbody>
            {users.isLoading && (
              <tr>
                <td className="px-4 py-4 text-muted" colSpan={5}>
                  Consultando…
                </td>
              </tr>
            )}
            {users.isError && (
              <tr>
                <td className="px-4 py-4 text-danger" colSpan={5}>
                  {users.error instanceof Error ? users.error.message : "No se pudo leer"}
                </td>
              </tr>
            )}
            {rows.map((row) => {
              const self = row.username === me?.username;
              return (
                <tr key={row.id} className="border-b border-border/60 last:border-0">
                  <td className="px-4 py-3 font-medium text-ink">
                    {row.username}
                    {self && <span className="ml-2 text-[11px] text-subtle">(vos)</span>}
                  </td>
                  <td className="px-4 py-3">
                    <select
                      className="rounded-lg border border-border bg-canvas px-2 py-1 text-xs text-ink"
                      value={row.role}
                      onChange={(event) =>
                        update.mutate({
                          id: row.id,
                          body: { role: event.target.value as UserRole },
                        })
                      }
                    >
                      {ROLES.map((value) => (
                        <option key={value} value={value}>
                          {ROLE_LABELS[value]}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${
                        row.is_active
                          ? "bg-accent-soft text-accent"
                          : "bg-elevated text-subtle"
                      }`}
                    >
                      {row.is_active ? "activo" : "inactivo"}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-muted">{when(row.last_login_at)}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end gap-1">
                      <button
                        type="button"
                        title="Cambiar contraseña"
                        onClick={() => resetPassword(row)}
                        className="rounded-lg p-2 text-muted hover:bg-elevated hover:text-ink"
                      >
                        <KeyRound className="h-4 w-4" />
                      </button>
                      <button
                        type="button"
                        title={row.is_active ? "Desactivar" : "Activar"}
                        onClick={() =>
                          update.mutate({ id: row.id, body: { is_active: !row.is_active } })
                        }
                        className="rounded-lg p-2 text-muted hover:bg-elevated hover:text-ink"
                      >
                        {row.is_active ? (
                          <UserX className="h-4 w-4" />
                        ) : (
                          <UserCheck className="h-4 w-4" />
                        )}
                      </button>
                      <button
                        type="button"
                        title="Eliminar"
                        disabled={self}
                        onClick={() => {
                          if (window.confirm(`¿Eliminar a ${row.username}?`)) {
                            remove.mutate(row.id);
                          }
                        }}
                        className="rounded-lg p-2 text-muted hover:bg-danger/10 hover:text-danger disabled:opacity-30"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
            {!users.isLoading && !users.isError && rows.length === 0 && (
              <tr>
                <td className="px-4 py-4 text-muted" colSpan={5}>
                  No hay usuarios.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <p className="text-[11px] text-subtle">
        No podés eliminarte a vos mismo, ni degradar o desactivar al último
        administrador activo. El servidor rechaza ambas operaciones.
      </p>
    </div>
  );
}
