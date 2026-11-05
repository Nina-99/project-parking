// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

import { useState, type FormEvent } from "react";
import {
  Camera,
  Car,
  LogIn,
  ScanSearch,
  ShieldCheck,
} from "lucide-react";
import { useSession } from "../../application/session";
import { SettingsMenu } from "../components/SettingsMenu";

const highlights = [
  {
    icon: Camera,
    title: "Dos portones, una ocupación",
    text: "Entrada y salida alimentan el mismo estado, en vivo.",
  },
  {
    icon: ScanSearch,
    title: "Evidencia, no adivinanzas",
    text: "Candidatos por ventana, color y confianza medida.",
  },
  {
    icon: ShieldCheck,
    title: "Cierre auditado",
    text: "Cada corrección queda con operador y motivo.",
  },
];

export function Login() {
  const { login, error } = useSession();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    await login(username, password);
    setBusy(false);
  }

  return (
    <div className="grid min-h-screen bg-canvas lg:grid-cols-[1.05fr_1fr]">
      <section className="relative hidden overflow-hidden bg-brand p-12 text-brand-ink lg:flex lg:flex-col lg:justify-between">
        <div className="absolute inset-0 blueprint opacity-70" />
        <div className="absolute -left-24 top-1/4 h-72 w-72 rounded-full bg-accent/20 blur-3xl" />
        <div className="absolute -bottom-24 right-0 h-80 w-80 rounded-full bg-warn/10 blur-3xl" />

        <div className="relative">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-white/15 bg-white/5">
              <Car className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-semibold">Parking Intelligence</p>
              <p className="text-[11px] uppercase tracking-[0.18em] text-brand-ink/60">
                Control de acceso
              </p>
            </div>
          </div>

          <h1 className="font-display mt-16 max-w-md bg-gradient-to-br from-white via-white to-accent bg-clip-text text-4xl font-semibold leading-[1.08] tracking-tight text-transparent xl:text-5xl">
            Cada vehículo deja un rastro.
          </h1>
          <p className="mt-4 max-w-md text-sm leading-relaxed text-brand-ink/70">
            Ocupación, evidencia y auditoría en un solo panel. Del frame al
            incidente, sin planillas.
          </p>
        </div>

        <div className="relative mt-16 space-y-5">
          {highlights.map((item) => (
            <div key={item.title} className="flex items-start gap-3">
              <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-accent">
                <item.icon className="h-4 w-4" />
              </div>
              <div>
                <p className="text-sm font-medium">{item.title}</p>
                <p className="text-xs leading-relaxed text-brand-ink/60">
                  {item.text}
                </p>
              </div>
            </div>
          ))}
        </div>

        <p className="relative mt-12 text-[11px] uppercase tracking-[0.18em] text-brand-ink/40">
          © 2026 Nina · Visión por computadora
        </p>
      </section>

      <section className="relative flex items-center justify-center overflow-hidden px-6 py-12">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 flex items-center justify-center"
        >
          <div className="radar h-[34rem] w-[34rem] opacity-80" />
        </div>
        <div className="absolute right-6 top-6">
          <SettingsMenu />
        </div>

        <form
          onSubmit={submit}
          className="page-enter surface-glass relative z-10 w-full max-w-sm space-y-5 rounded-2xl border border-border p-6 shadow-2xl"
        >
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-subtle">
              Acceso de operador
            </p>
            <h2 className="font-display mt-2 text-2xl font-semibold tracking-tight text-ink">
              Entrá al panel
            </h2>
            <p className="mt-1 text-sm text-muted">
              Las patentes son datos personales. La sesión es obligatoria.
            </p>
          </div>

          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-muted">Usuario</span>
            <input
              autoComplete="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              className="field"
            />
          </label>

          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium text-muted">Contraseña</span>
            <input
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="field"
            />
          </label>

          {error && <p className="text-sm text-danger">{error}</p>}

          <button type="submit" disabled={busy} className="btn-primary w-full">
            <LogIn className="h-4 w-4" />
            {busy ? "Entrando…" : "Entrar"}
          </button>

          <p className="text-xs text-subtle">
            Si no tenés credenciales, pedilas al administrador del sistema.
          </p>
        </form>
      </section>
    </div>
  );
}
