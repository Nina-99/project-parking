// SPDX-FileCopyrightText: 2026 Nina
// SPDX-License-Identifier: LicenseRef-Proprietary

import { useEffect, useRef, useState } from "react";
import { Check, Monitor, Moon, Settings, Sun } from "lucide-react";
import { useTheme, type Theme } from "../theme/ThemeProvider";

const OPTIONS: { value: Theme; label: string; hint: string; icon: typeof Sun }[] = [
  { value: "light", label: "Claro", hint: "Proyector y aulas iluminadas", icon: Sun },
  { value: "dark", label: "Oscuro", hint: "Salas de control con poca luz", icon: Moon },
  { value: "system", label: "Sistema", hint: "Sigue la preferencia del equipo", icon: Monitor },
];

export function SettingsMenu() {
  const { theme, setTheme } = useTheme();
  const [open, setOpen] = useState(false);
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: MouseEvent) {
      if (container.current && !container.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div className="relative" ref={container}>
      <button
        type="button"
        aria-label="Configuración"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        className="p-2 rounded-lg text-muted hover:text-ink hover:bg-elevated transition-colors"
      >
        <Settings className="w-4 h-4" />
      </button>

      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full mt-2 w-64 rounded-xl border border-border bg-surface p-2 shadow-xl"
        >
          <p className="px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-subtle">
            Configuración
          </p>
          <p className="px-2 pt-3 pb-1 text-xs font-medium text-muted">Tema</p>
          {OPTIONS.map((option) => {
            const Icon = option.icon;
            const active = theme === option.value;
            return (
              <button
                key={option.value}
                type="button"
                role="menuitemradio"
                aria-checked={active}
                onClick={() => {
                  setTheme(option.value);
                  setOpen(false);
                }}
                className={`flex w-full items-start gap-3 rounded-lg px-2 py-2 text-left transition-colors ${
                  active ? "bg-accent-soft text-accent" : "text-ink hover:bg-elevated"
                }`}
              >
                <Icon className="mt-0.5 w-4 h-4 shrink-0" />
                <span className="min-w-0 flex-1">
                  <span className="block text-sm font-medium">{option.label}</span>
                  <span
                    className={`block text-xs ${
                      active ? "text-accent/80" : "text-subtle"
                    }`}
                  >
                    {option.hint}
                  </span>
                </span>
                {active && <Check className="mt-0.5 w-4 h-4 shrink-0" />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
