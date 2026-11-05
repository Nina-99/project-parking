// SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
// SPDX-License-Identifier: LicenseRef-Proprietary

import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import {
  Activity,
  BarChart3,
  Car,
  ChevronDown,
  HistoryIcon,
  LayoutDashboard,
  LogOut,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Users,
  Video,
} from "lucide-react";
import { useSession } from "../../application/session";
import { SettingsMenu } from "./SettingsMenu";
import { ValidationQueue } from "./ValidationQueue";
import { AlertCenter } from "./AlertCenter";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/stats", label: "Estadísticas", icon: BarChart3 },
  { to: "/incidents", label: "Incidentes", icon: Search },
  { to: "/history", label: "Historial", icon: HistoryIcon },
];

// The registry is operational: operators use it to confirm who may enter, and
// the route enforces the same rule. An owner reads reports, not the padrón.
const whitelistItem = {
  to: "/whitelist",
  label: "Padrón",
  icon: ShieldCheck,
};

// Management is not a view: only an admin sees it, and the server enforces the
// same rule. Hiding the link is courtesy, not access control. Inference
// telemetry lives here because it diagnoses the detector, not the lot.
const adminNavItems = [
  {
    to: "/cameras",
    label: "Cámaras",
    hint: "RTSP, portón y latido",
    icon: Video,
  },
  {
    to: "/model",
    label: "Modelo",
    hint: "Latencia y confianza de inferencia",
    icon: Activity,
  },
  {
    to: "/users",
    label: "Usuarios",
    hint: "Roles y acceso al panel",
    icon: Users,
  },
];

function navClass(isActive: boolean): string {
  return `flex whitespace-nowrap items-center gap-1.5 rounded-lg px-2.5 py-2 text-[13px] font-medium transition-all duration-200 ${
    isActive
      ? "bg-accent-soft text-accent ring-1 ring-accent/25 shadow-[0_6px_18px_-14px_var(--color-accent)]"
      : "text-muted hover:bg-elevated hover:text-ink"
  }`;
}

function AdminMenu({ align = "left" }: { align?: "left" | "right" }) {
  const location = useLocation();
  const [open, setOpen] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  const active = adminNavItems.some((item) =>
    location.pathname.startsWith(item.to),
  );

  useEffect(() => {
    setOpen(false);
  }, [location.pathname]);

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
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        className={navClass(active)}
      >
        <SlidersHorizontal className="h-4 w-4" />
        Administración
        <ChevronDown
          className={`h-3.5 w-3.5 transition-transform duration-200 ${
            open ? "rotate-180" : ""
          }`}
        />
      </button>

      {open && (
        <div
          role="menu"
          aria-label="Administración"
          className={`surface-glass dropdown-in absolute ${
            align === "right" ? "right-0" : "left-0"
          } top-full z-50 mt-2 w-72 max-w-[calc(100vw-2rem)] rounded-xl border border-border p-2 shadow-2xl`}
        >
          <p className="px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-subtle">
            Administración
          </p>
          {adminNavItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              role="menuitem"
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `flex items-start gap-3 rounded-lg px-2 py-2 transition-colors ${
                  isActive
                    ? "bg-accent-soft text-accent"
                    : "text-ink hover:bg-elevated"
                }`
              }
            >
              <span className="icon-chip mt-0.5 h-7 w-7 shrink-0 rounded-lg">
                <item.icon className="h-3.5 w-3.5" />
              </span>
              <span className="min-w-0">
                <span className="block text-sm font-medium">{item.label}</span>
                <span className="block text-xs text-subtle">{item.hint}</span>
              </span>
            </NavLink>
          ))}
        </div>
      )}
    </div>
  );
}

export function Layout() {
  const { user, logout } = useSession();
  const location = useLocation();
  const isAdmin = user?.role === "admin";
  const canReadWhitelist = isAdmin || user?.role === "operator";
  const items = canReadWhitelist ? [...navItems, whitelistItem] : navItems;

  return (
    <div className="relative min-h-screen">
      <div aria-hidden className="ambient-layer">
        <div className="ambient-blob ambient-blob--teal" />
        <div className="ambient-blob ambient-blob--amber" />
        <div className="ambient-grid" />
      </div>

      <header className="surface-glass sticky top-0 z-50 border-b border-border/70 shadow-[0_18px_40px_-32px_rgba(2,6,23,0.7)]">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
          <div className="flex shrink-0 items-center gap-3">
            <div className="icon-chip flex h-9 w-9 items-center justify-center rounded-xl">
              <Car className="h-5 w-5" />
            </div>
            <div className="leading-tight whitespace-nowrap">
              <p className="font-display whitespace-nowrap text-sm font-semibold tracking-tight text-ink">
                Parking Intelligence
              </p>
              <p className="whitespace-nowrap text-[11px] uppercase tracking-[0.16em] text-subtle">
                Control de acceso
              </p>
            </div>
          </div>

          <nav className="hidden min-w-0 items-center gap-1 xl:flex">
            {items.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === "/"}
                className={({ isActive }) => navClass(isActive)}
              >
                <item.icon className="h-4 w-4" />
                {item.label}
              </NavLink>
            ))}
            {isAdmin && <AdminMenu />}
          </nav>

          <div className="flex shrink-0 items-center gap-2">
            <span className="hidden text-sm font-medium text-ink lg:inline">
              {user?.username}
            </span>
            <AlertCenter />
            {user?.role && (
              <span className="badge-completed hidden uppercase tracking-wider text-[10px] font-semibold lg:inline-block">
                {user.role === "admin"
                  ? "Admin"
                  : user.role === "operator"
                    ? "Operador"
                    : "Dueño"}
              </span>
            )}
            <SettingsMenu />
            <button
              type="button"
              onClick={() => void logout()}
              title="Cerrar sesión"
              aria-label="Cerrar sesión"
              className="rounded-lg p-2 text-muted transition-colors hover:bg-elevated hover:text-ink"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>

        <nav className="flex items-center gap-1 overflow-x-auto border-t border-border px-4 py-2 xl:hidden">
          {items.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) => navClass(isActive)}
            >
              <item.icon className="h-4 w-4" />
              {item.label}
            </NavLink>
          ))}
          {isAdmin && <AdminMenu align="right" />}
        </nav>
      </header>

      <main
        key={location.pathname}
        className="page-enter relative z-10 mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8"
      >
        <Outlet />
      </main>

      <div className="relative z-10 border-t border-border/60">
        <p className="mx-auto max-w-7xl px-4 py-4 text-[11px] text-subtle sm:px-6 lg:px-8">
          © 2026 Nina · Parking Intelligence
        </p>
      </div>
      <ValidationQueue />
      {/* <footer className="mx-auto max-w-7xl px-4 pb-8 text-xs text-subtle sm:px-6 lg:px-8"> */}
      {/*   © 2026 Nina · Parking Intelligence */}
      {/* </footer> */}
    </div>
  );
}
