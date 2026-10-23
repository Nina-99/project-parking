import { Outlet, NavLink } from "react-router-dom";
import {
  Activity,
  Car,
  HistoryIcon,
  LayoutDashboard,
  LogOut,
  Search,
} from "lucide-react";
import { useLiveEvents } from "../../application/useLiveEvents";
import { useSession } from "../../application/session";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/incidents", label: "Incidentes", icon: Search },
  { to: "/history", label: "Historial", icon: HistoryIcon },
  { to: "/model", label: "Modelo", icon: Activity },
];

export function Layout() {
  useLiveEvents();
  const { user, logout } = useSession();

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <Car className="w-8 h-8 text-primary-600" />
              <h1 className="text-xl font-bold text-gray-900">
                Parking Intelligence
              </h1>
            </div>
            <nav className="flex items-center gap-1">
              {navItems.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === "/"}
                  className={({ isActive }) =>
                    `flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      isActive
                        ? "bg-primary-50 text-primary-700"
                        : "text-gray-600 hover:text-gray-900 hover:bg-gray-100"
                    }`
                  }
                >
                  <item.icon className="w-4 h-4" />
                  {item.label}
                </NavLink>
              ))}
              <span className="mx-2 h-6 w-px bg-gray-200" />
              <span className="text-sm text-gray-500">{user}</span>
              <button
                type="button"
                onClick={() => void logout()}
                title="Cerrar sesión"
                className="p-2 rounded-lg text-gray-500 hover:text-gray-900 hover:bg-gray-100 transition-colors"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>
    </div>
  );
}
