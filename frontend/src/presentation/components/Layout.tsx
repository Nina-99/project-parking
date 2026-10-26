import { NavLink, Outlet } from "react-router-dom";
import {
  Activity,
  Car,
  HistoryIcon,
  LayoutDashboard,
  LogOut,
  Search,
} from "lucide-react";
import { useCameras } from "../../application/queries";
import { useSession } from "../../application/session";
import { SettingsMenu } from "./SettingsMenu";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/incidents", label: "Incidentes", icon: Search },
  { to: "/history", label: "Historial", icon: HistoryIcon },
  { to: "/model", label: "Modelo", icon: Activity },
];

function navClass(isActive: boolean): string {
  return `flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
    isActive
      ? "bg-accent-soft text-accent"
      : "text-muted hover:bg-elevated hover:text-ink"
  }`;
}

export function Layout() {
  const { user, logout } = useSession();
  const cameras = useCameras();

  const total = cameras.data?.cameras.length ?? 0;
  const online = cameras.data?.cameras.filter((camera) => camera.online).length ?? 0;
  const camerasOk = !cameras.isError && total > 0 && online === total;
  const cameraLabel = cameras.isLoading
    ? "Consultando…"
    : cameras.isError
      ? "Sin datos"
      : total === 0
        ? "Sin cámaras"
        : `${online}/${total} cámaras`;

  return (
    <div className="min-h-screen bg-canvas">
      <header className="sticky top-0 z-50 border-b border-border bg-surface/90 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent-soft text-accent">
              <Car className="h-5 w-5" />
            </div>
            <div className="leading-tight">
              <p className="font-display text-sm font-semibold tracking-tight text-ink">
                Parking Intelligence
              </p>
              <p className="text-[11px] uppercase tracking-[0.16em] text-subtle">
                Control de acceso
              </p>
            </div>
          </div>

          <nav className="hidden items-center gap-1 md:flex">
            {navItems.map((item) => (
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
          </nav>

          <div className="flex items-center gap-2">
            <span
              className={`hidden items-center gap-2 rounded-full border border-border px-3 py-1 text-xs font-medium lg:inline-flex ${
                camerasOk ? "text-live" : "text-warn"
              }`}
            >
              <span
                className={
                  camerasOk
                    ? "live-dot"
                    : "inline-block h-2 w-2 rounded-full bg-warn"
                }
              />
              {cameraLabel}
            </span>
            <span className="hidden text-sm text-muted sm:inline">{user}</span>
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

        <nav className="flex items-center gap-1 overflow-x-auto border-t border-border px-4 py-2 md:hidden">
          {navItems.map((item) => (
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
        </nav>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <Outlet />
      </main>

      <footer className="mx-auto max-w-7xl px-4 pb-8 text-xs text-subtle sm:px-6 lg:px-8">
        © 2026 Nina · Parking Intelligence
      </footer>
    </div>
  );
}
