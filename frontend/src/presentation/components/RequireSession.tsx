import type { ReactNode } from "react";
import { useSession } from "../../application/session";
import { Login } from "../pages/Login";

export function RequireSession({ children }: { children: ReactNode }) {
  const { user, loading } = useSession();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-sm text-gray-400">
        Cargando…
      </div>
    );
  }
  if (!user) return <Login />;
  return <>{children}</>;
}
