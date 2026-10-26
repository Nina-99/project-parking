// SPDX-FileCopyrightText: 2026 Nina
// SPDX-License-Identifier: LicenseRef-Proprietary

import type { ReactNode } from "react";
import { LiveEventsProvider } from "../../application/liveEvents";
import { useSession } from "../../application/session";
import { Login } from "../pages/Login";

export function RequireSession({ children }: { children: ReactNode }) {
  const { user, loading } = useSession();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-sm text-subtle">
        Cargando…
      </div>
    );
  }
  if (!user) return <Login />;
  return <LiveEventsProvider>{children}</LiveEventsProvider>;
}
